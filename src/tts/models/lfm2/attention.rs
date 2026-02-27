use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use super::RmsNorm;
use crate::tts::models::config::Lfm2Config;

/// Grouped Query Attention with RoPE and QK LayerNorm.
pub struct Lfm2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_layernorm: RmsNorm,
    k_layernorm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Pre-computed RoPE inverse frequencies on device, shape (1, half_d)
    rope_freq: Tensor,
}

impl Lfm2Attention {
    pub fn load(vb: VarBuilder, cfg: &Lfm2Config) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let kv_dim = cfg.num_key_value_heads * head_dim;

        let q_proj = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, kv_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, kv_dim, vb.pp("v_proj"))?;
        let out_proj = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("out_proj"))?;

        let q_layernorm =
            RmsNorm::load(vb.pp("q_layernorm"), head_dim, cfg.norm_eps)?;
        let k_layernorm =
            RmsNorm::load(vb.pp("k_layernorm"), head_dim, cfg.norm_eps)?;

        // Pre-compute RoPE frequencies on the model's device (avoids H2D per step)
        let half_d = head_dim / 2;
        let freq: Vec<f32> = (0..half_d)
            .map(|i| 1.0f32 / (cfg.rope_theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let device = q_proj.weight().device();
        let rope_freq = Tensor::from_vec(freq, (1, half_d), device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_layernorm,
            k_layernorm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            rope_freq,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        seq_offset: usize,
    ) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape: (B, T, H*D) → (B, T, H, D)
        let q = q.reshape((b, t, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, t, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, t, self.num_kv_heads, self.head_dim))?;

        // QK LayerNorm (applied per head)
        let q = self.q_layernorm.forward(&q)?;
        let k = self.k_layernorm.forward(&k)?;

        // Transpose: (B, H, T, D), contiguous for matmul
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Apply RoPE (uses cached freq tensor — no H2D)
        let q = apply_rope_cached(&q, seq_offset, &self.rope_freq)?.contiguous()?;
        let k = apply_rope_cached(&k, seq_offset, &self.rope_freq)?.contiguous()?;

        // KV cache handling
        let (k, v) = match kv_cache.take() {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[&prev_k, &k], 2)?;
                let v = Tensor::cat(&[&prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        *kv_cache = Some((k.clone(), v.clone()));

        // GQA: expand KV heads to match Q heads
        let k = expand_kv_heads(&k, self.num_heads, self.num_kv_heads)?;
        let v = expand_kv_heads(&v, self.num_heads, self.num_kv_heads)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.t()?.contiguous()?)? / scale)?;

        // Causal mask (only for prefill; single token needs no mask)
        let kv_len = attn_weights.dim(3)?;
        let q_len = attn_weights.dim(2)?;
        let attn_weights = if q_len > 1 {
            let mask = create_causal_mask(q_len, kv_len, attn_weights.device())?
                .to_dtype(attn_weights.dtype())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let out = attn_weights.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

/// Apply RoPE using a pre-computed frequency tensor (already on device).
/// x shape: (B, H, T, D), freq shape: (1, D/2)
fn apply_rope_cached(x: &Tensor, offset: usize, freq: &Tensor) -> Result<Tensor> {
    let (_b, _h, t, d) = x.dims4()?;
    let half_d = d / 2;
    let device = x.device();
    let dtype = x.dtype();

    // Positions: single scalar for generation (t=1), small vec for prefill
    let positions = Tensor::arange(offset as u32, (offset + t) as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?; // (T, 1)

    // angles: (T, D/2) = positions @ freq
    let angles = positions.matmul(freq)?; // (T, 1) @ (1, D/2) → (T, D/2)
    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;

    // Split x into halves
    let x1 = x.narrow(3, 0, half_d)?;
    let x2 = x.narrow(3, half_d, half_d)?;

    // Broadcast: (T, D/2) → (1, 1, T, D/2)
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

    Tensor::cat(&[&r1, &r2], 3)
}

/// Expand KV heads for Grouped Query Attention.
/// input: (B, KV_H, T, D) → output: (B, Q_H, T, D)
fn expand_kv_heads(
    x: &Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
) -> Result<Tensor> {
    if num_q_heads == num_kv_heads {
        return Ok(x.clone());
    }
    let repeat = num_q_heads / num_kv_heads;
    let (b, _h, t, d) = x.dims4()?;

    // (B, KV_H, T, D) → (B, KV_H, 1, T, D) → expand → (B, KV_H, repeat, T, D) → reshape
    x.unsqueeze(2)?
        .expand((b, num_kv_heads, repeat, t, d))?
        .reshape((b, num_q_heads, t, d))
}

/// Create a causal attention mask.
fn create_causal_mask(q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            let row_start = kv_len - q_len;
            (0..kv_len).map(move |j| {
                if j <= row_start + i {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();

    Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)
}
