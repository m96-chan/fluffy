use candle_core::{DType, Result, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};

/// Transformer with optional windowed local attention and AdaLN-Zero.
///
/// Weight naming convention (matching MioCodec safetensors):
/// - Attention: `layers.N.attention.{wq,wk,wv,wo}.weight`
/// - Norms: `layers.N.{attention_norm,ffn_norm}.{weight,bias}` (plain)
///   or `layers.N.{attention_norm,ffn_norm}.condition_proj.1.{weight,bias}` (AdaLN)
/// - FFN (SwiGLU): `layers.N.feed_forward.{w1,w2,w3}.weight`
/// - Final norm: `norm.{weight,bias}` or `norm.condition_proj.1.{weight,bias}`
/// - Output proj: `output_proj.{weight,bias}` (optional)
pub struct Transformer {
    layers: Vec<TransformerLayer>,
    final_norm: NormLayer,
    output_proj: Option<Linear>,
    window_size: usize,
}

impl Transformer {
    pub fn load(
        vb: VarBuilder,
        dim: usize,
        ff_dim: usize,
        output_dim: Option<usize>,
        n_layers: usize,
        n_heads: usize,
        window_size: usize,
        rope_theta: f64,
        adaln_cond_dim: Option<usize>,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let layer = TransformerLayer::load(
                vb.pp(&format!("layers.{i}")),
                dim,
                ff_dim,
                n_heads,
                rope_theta,
                adaln_cond_dim,
            )?;
            layers.push(layer);
        }

        let final_norm = if let Some(cond_dim) = adaln_cond_dim {
            // AdaLN final norm: 2*dim (shift, scale only, no gate)
            NormLayer::load_adaln(vb.pp("norm"), cond_dim, dim, false)?
        } else {
            NormLayer::load_plain(vb.pp("norm"), dim)?
        };

        let output_proj = if let Some(out_d) = output_dim {
            Some(linear(dim, out_d, vb.pp("output_proj"))?)
        } else {
            None
        };

        Ok(Self {
            layers,
            final_norm,
            output_proj,
            window_size,
        })
    }

    /// Forward pass.
    /// - `x`: (B, T, dim)
    /// - `condition`: optional (B, cond_dim) for AdaLN-Zero
    /// Returns: (B, T, output_dim)
    pub fn forward(&self, x: &Tensor, condition: Option<&Tensor>) -> Result<Tensor> {
        let mut h = x.clone();

        for layer in &self.layers {
            h = layer.forward(&h, condition, self.window_size)?;
        }

        h = self.final_norm.forward(&h, condition)?;

        if let Some(proj) = &self.output_proj {
            h = proj.forward(&h)?;
        }

        Ok(h)
    }
}

/// Single transformer layer with optional AdaLN-Zero conditioning.
struct TransformerLayer {
    attention_norm: NormLayer,
    attention: LocalAttention,
    ffn_norm: NormLayer,
    feed_forward: SwiGLUFeedForward,
}

impl TransformerLayer {
    fn load(
        vb: VarBuilder,
        dim: usize,
        ff_dim: usize,
        n_heads: usize,
        rope_theta: f64,
        adaln_cond_dim: Option<usize>,
    ) -> Result<Self> {
        let attention_norm = if let Some(cond_dim) = adaln_cond_dim {
            NormLayer::load_adaln(vb.pp("attention_norm"), cond_dim, dim, true)?
        } else {
            NormLayer::load_plain(vb.pp("attention_norm"), dim)?
        };

        let attention = LocalAttention::load(vb.pp("attention"), dim, n_heads, rope_theta)?;

        let ffn_norm = if let Some(cond_dim) = adaln_cond_dim {
            NormLayer::load_adaln(vb.pp("ffn_norm"), cond_dim, dim, true)?
        } else {
            NormLayer::load_plain(vb.pp("ffn_norm"), dim)?
        };

        let feed_forward = SwiGLUFeedForward::load(vb.pp("feed_forward"), dim, ff_dim)?;

        Ok(Self {
            attention_norm,
            attention,
            ffn_norm,
            feed_forward,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        condition: Option<&Tensor>,
        window_size: usize,
    ) -> Result<Tensor> {
        match &self.attention_norm {
            NormLayer::AdaLN { has_gate: true, .. } => {
                // AdaLN-Zero mode
                let (h, gate1) = self.attention_norm.forward_with_gate(x, condition)?;
                let h = self.attention.forward(&h, window_size)?;
                let h = (x + h.broadcast_mul(&gate1)?)?;

                let residual = &h;
                let (h, gate2) = self.ffn_norm.forward_with_gate(&h, condition)?;
                let h = self.feed_forward.forward(&h)?;
                residual + h.broadcast_mul(&gate2)?
            }
            _ => {
                // Standard pre-norm transformer
                let residual = x;
                let h = self.attention_norm.forward(x, condition)?;
                let h = self.attention.forward(&h, window_size)?;
                let h = (residual + h)?;
                let residual = &h;
                let h = self.ffn_norm.forward(&h, condition)?;
                let h = self.feed_forward.forward(&h)?;
                residual + h
            }
        }
    }
}

/// Norm layer: either plain LayerNorm or AdaLN (no learnable affine, modulated by condition).
enum NormLayer {
    Plain(candle_nn::LayerNorm),
    AdaLN {
        condition_proj: Linear, // loaded from "condition_proj.1"
        dim: usize,
        has_gate: bool, // true for attention_norm/ffn_norm (3*dim), false for final norm (2*dim)
    },
}

impl NormLayer {
    fn load_plain(vb: VarBuilder, dim: usize) -> Result<Self> {
        let norm = candle_nn::layer_norm(dim, 1e-5, vb)?;
        Ok(Self::Plain(norm))
    }

    fn load_adaln(vb: VarBuilder, cond_dim: usize, dim: usize, has_gate: bool) -> Result<Self> {
        let out_dim = if has_gate { 3 * dim } else { 2 * dim };
        let condition_proj = linear(cond_dim, out_dim, vb.pp("condition_proj.1"))?;
        Ok(Self::AdaLN {
            condition_proj,
            dim,
            has_gate,
        })
    }

    /// Forward: normalize and modulate. For AdaLN with gate, the gate is applied
    /// externally, so this returns the modulated output (without gate).
    fn forward(&self, x: &Tensor, condition: Option<&Tensor>) -> Result<Tensor> {
        match self {
            Self::Plain(norm) => norm.forward(x),
            Self::AdaLN {
                condition_proj,
                dim,
                has_gate,
            } => {
                let cond = condition.ok_or_else(|| {
                    candle_core::Error::Msg("AdaLN requires condition".into())
                })?;
                let h = layer_norm_no_affine(x, 1e-5)?;
                let params = condition_proj.forward(&candle_nn::ops::silu(cond)?)?;
                // params: (B, N*dim) where N=3 (gate) or N=2 (no gate)
                let shift = params.narrow(1, 0, *dim)?.unsqueeze(1)?;
                let scale = params.narrow(1, *dim, *dim)?.unsqueeze(1)?;
                if *has_gate {
                    // Return modulated result; gate handled by caller via forward_with_gate
                    adaln_modulate(&h, &shift, &scale)
                } else {
                    adaln_modulate(&h, &shift, &scale)
                }
            }
        }
    }

    /// Forward with gate extraction. Returns (modulated_output, gate).
    fn forward_with_gate(
        &self,
        x: &Tensor,
        condition: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        match self {
            Self::AdaLN {
                condition_proj,
                dim,
                has_gate: true,
            } => {
                let cond = condition.ok_or_else(|| {
                    candle_core::Error::Msg("AdaLN requires condition".into())
                })?;
                let h = layer_norm_no_affine(x, 1e-5)?;
                let params = condition_proj.forward(&candle_nn::ops::silu(cond)?)?;
                let shift = params.narrow(1, 0, *dim)?.unsqueeze(1)?;
                let scale = params.narrow(1, *dim, *dim)?.unsqueeze(1)?;
                let gate = params.narrow(1, 2 * dim, *dim)?.unsqueeze(1)?;
                let modulated = adaln_modulate(&h, &shift, &scale)?;
                Ok((modulated, gate))
            }
            _ => {
                let h = self.forward(x, condition)?;
                // No gate, return ones
                let gate = Tensor::ones_like(&h)?;
                Ok((h, gate))
            }
        }
    }
}

/// LayerNorm without learnable affine parameters.
fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(candle_core::D::Minus1)?;
    let x_centered = x.broadcast_sub(&mean)?;
    let var = x_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let normed = x_centered.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed.to_dtype(dtype)
}

/// Apply AdaLN modulation: x * (1 + scale) + shift
fn adaln_modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let modulated = x.broadcast_mul(&(scale + 1.0)?)?;
    modulated.broadcast_add(shift)
}

/// SwiGLU feed-forward: w2(silu(w1(x)) * w3(x))
struct SwiGLUFeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl SwiGLUFeedForward {
    fn load(vb: VarBuilder, dim: usize, ff_dim: usize) -> Result<Self> {
        let w1 = linear_no_bias(dim, ff_dim, vb.pp("w1"))?;
        let w2 = linear_no_bias(ff_dim, dim, vb.pp("w2"))?;
        let w3 = linear_no_bias(dim, ff_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

/// Local windowed multi-head attention with RoPE.
struct LocalAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

impl LocalAttention {
    fn load(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        rope_theta: f64,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let wq = linear_no_bias(dim, dim, vb.pp("wq"))?;
        let wk = linear_no_bias(dim, dim, vb.pp("wk"))?;
        let wv = linear_no_bias(dim, dim, vb.pp("wv"))?;
        let wo = linear_no_bias(dim, dim, vb.pp("wo"))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            head_dim,
            rope_theta,
        })
    }

    fn forward(&self, x: &Tensor, window_size: usize) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;

        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        // Reshape: (B, T, D) → (B, H, T, Dh), contiguous for matmul
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply RoPE
        let q = apply_rope_local(&q, self.rope_theta)?.contiguous()?;
        let k = apply_rope_local(&k, self.rope_theta)?.contiguous()?;

        // Local attention with window mask.
        // Window ±(window_size/2): positions farther apart don't attend.
        let half_win = window_size / 2;
        let out = if t <= half_win + 1 {
            // All positions within window — no mask needed
            let scale = (self.head_dim as f64).sqrt();
            let attn = (q.matmul(&k.t()?.contiguous()?)? / scale)?;
            let attn = candle_nn::ops::softmax_last_dim(&attn)?;
            attn.matmul(&v)?
        } else {
            // Apply windowed band mask
            windowed_attention(&q, &k, &v, window_size, self.head_dim)?
        };

        // (B, H, T, Dh) → (B, T, D)
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.num_heads * self.head_dim))?;
        self.wo.forward(&out)
    }
}

/// Windowed local attention for long sequences.
fn windowed_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    window_size: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let (_b, _h, t, _d) = q.dims4()?;
    let scale = (head_dim as f64).sqrt();
    let half_win = window_size / 2;

    let attn = (q.matmul(&k.t()?.contiguous()?)? / scale)?;

    // Create a band mask: each position attends only to ±half_win
    let device = q.device();
    let mask: Vec<f32> = (0..t)
        .flat_map(|i| {
            (0..t).map(move |j| {
                let dist = if i >= j { i - j } else { j - i };
                if dist <= half_win {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    let mask = Tensor::from_vec(mask, (1, 1, t, t), device)?.to_dtype(attn.dtype())?;
    let attn = attn.broadcast_add(&mask)?;
    let attn = candle_nn::ops::softmax_last_dim(&attn)?;
    attn.matmul(v)
}

/// Apply RoPE using the interleaved complex-number convention.
///
/// Python: `view_as_complex(x.reshape(..., -1, 2))` treats consecutive pairs
/// (x[0],x[1]), (x[2],x[3]), ... as (real, imag) of complex numbers, then
/// multiplies by `exp(i*theta)`. We replicate this exactly.
fn apply_rope_local(x: &Tensor, theta: f64) -> Result<Tensor> {
    let (b, h, t, d) = x.dims4()?;
    let half_d = d / 2;
    let device = x.device();
    let dtype = x.dtype();

    // freq[i] = 1 / (theta^(2i/d)) for i = 0..half_d
    let freq: Vec<f32> = (0..half_d)
        .map(|i| 1.0f32 / (theta as f32).powf(2.0 * i as f32 / d as f32))
        .collect();
    let freq = Tensor::from_vec(freq, half_d, device)?;

    let positions: Vec<f32> = (0..t).map(|i| i as f32).collect();
    let positions = Tensor::from_vec(positions, t, device)?;

    // angles: (T, half_d) — angle for each position and frequency bin
    let angles = positions.unsqueeze(1)?.matmul(&freq.unsqueeze(0)?)?;
    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;

    // Reshape x from (B, H, T, D) → (B, H, T, D/2, 2) to extract interleaved pairs
    let x_pairs = x.reshape((b, h, t, half_d, 2))?;
    let x_even = x_pairs.narrow(4, 0, 1)?.squeeze(4)?; // (B, H, T, D/2) — real parts
    let x_odd = x_pairs.narrow(4, 1, 1)?.squeeze(4)?;  // (B, H, T, D/2) — imag parts

    // Broadcast cos/sin: (T, D/2) → (1, 1, T, D/2)
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    // Complex multiply: (real + i*imag) * (cos + i*sin)
    let r_even = (x_even.broadcast_mul(&cos)? - x_odd.broadcast_mul(&sin)?)?;
    let r_odd = (x_even.broadcast_mul(&sin)? + x_odd.broadcast_mul(&cos)?)?;

    // Interleave back: (B, H, T, D/2, 1) + (B, H, T, D/2, 1) → cat → (B, H, T, D/2, 2) → flatten
    let r_even = r_even.unsqueeze(4)?;
    let r_odd = r_odd.unsqueeze(4)?;
    Tensor::cat(&[&r_even, &r_odd], 4)?.reshape((b, h, t, d))
}
