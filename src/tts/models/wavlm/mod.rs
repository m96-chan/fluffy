mod global_encoder;

use candle_core::{DType, Result, Tensor};
use candle_nn::{group_norm, Conv1d, Conv1dConfig, GroupNorm, Linear, Module, VarBuilder};

pub use global_encoder::GlobalEncoder;

/// Minimal WavLM-base+ implementation.
///
/// Only loads the convolutional feature extractor + the first few transformer
/// encoder layers needed for global SSL features (layers 1, 2).
/// The rest of the model is NOT loaded to save VRAM.
pub struct WavLM {
    feature_extractor: WavLMFeatureExtractor,
    feature_projection: WavLMFeatureProjection,
    pos_conv_embed: PosConvEmbed,
    encoder_layer_norm: candle_nn::LayerNorm,
    encoder_layers: Vec<WavLMEncoderLayer>,
    /// Which layers to average for global features
    global_ssl_layers: Vec<usize>,
}

impl WavLM {
    /// Load only layers 0..=max(global_ssl_layers) from the WavLM checkpoint.
    pub fn load(
        vb: VarBuilder,
        global_ssl_layers: &[usize],
    ) -> Result<Self> {
        let max_layer = global_ssl_layers.iter().copied().max().unwrap_or(2);

        let feature_extractor =
            WavLMFeatureExtractor::load(vb.pp("feature_extractor"))?;
        let feature_projection =
            WavLMFeatureProjection::load(vb.pp("feature_projection"))?;

        // Convolutional position embedding
        let pos_conv_embed = PosConvEmbed::load(vb.pp("encoder.pos_conv_embed"), 768, 128, 16)?;

        // LayerNorm before encoder layers
        let encoder_layer_norm =
            candle_nn::layer_norm(768, 1e-5, vb.pp("encoder.layer_norm"))?;

        // Load relative position embedding from layer 0 (shared across layers in WavLM)
        let rel_attn_embed = vb
            .pp("encoder.layers.0.attention.rel_attn_embed")
            .get((320, 12), "weight")
            .ok();

        let mut encoder_layers = Vec::with_capacity(max_layer + 1);
        for i in 0..=max_layer {
            let layer = WavLMEncoderLayer::load(
                vb.pp(&format!("encoder.layers.{i}")),
                768, // hidden_size
                12,  // num_attention_heads
                rel_attn_embed.as_ref(),
            )?;
            encoder_layers.push(layer);
        }

        Ok(Self {
            feature_extractor,
            feature_projection,
            pos_conv_embed,
            encoder_layer_norm,
            encoder_layers,
            global_ssl_layers: global_ssl_layers.to_vec(),
        })
    }

    /// Extract global SSL features by averaging the outputs of specified layers.
    ///
    /// Input: raw 16kHz waveform (1, T)
    /// Output: (1, T', 768) where T' = T/320 approximately
    pub fn extract_global_features(&self, waveform: &Tensor) -> Result<Tensor> {
        // Feature extractor: conv stack
        let features = self.feature_extractor.forward(waveform)?;
        // features: (1, T', 512)

        // Feature projection: LayerNorm(512) + Linear(512, 768)
        let hidden = self.feature_projection.forward(&features)?;
        // hidden: (1, T', 768)

        // Convolutional position embedding (added to hidden)
        let pos_emb = self.pos_conv_embed.forward(&hidden)?;
        let hidden = (hidden + pos_emb)?;

        // LayerNorm before encoder
        let hidden = self.encoder_layer_norm.forward(&hidden)?;

        // Run through encoder layers, collecting specified outputs
        let mut x = hidden;
        let mut collected = Vec::new();

        for (i, layer) in self.encoder_layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if self.global_ssl_layers.contains(&(i + 1)) {
                // WavLM layer indexing: layer 1 = first encoder layer output
                collected.push(x.clone());
            }
        }

        if collected.is_empty() {
            return Ok(x);
        }

        // Average the collected layer outputs
        let stacked = Tensor::stack(&collected, 0)?;
        stacked.mean(0)
    }
}

/// Convolutional position embedding (wav2vec2/WavLM).
///
/// Groups=16 Conv1d with weight_norm(dim=2), kernel=128, padding=64.
struct PosConvEmbed {
    conv: Conv1d,
}

impl PosConvEmbed {
    fn load(
        vb: VarBuilder,
        channels: usize,
        kernel_size: usize,
        groups: usize,
    ) -> Result<Self> {
        // weight_norm with dim=2: weight = weight_g * weight_v / ||weight_v||
        // weight_v: (out_channels, in_channels/groups, kernel_size) = (768, 48, 128)
        // weight_g: (1, 1, kernel_size) = (1, 1, 128) — norm over dims 0,1
        let weight_v = vb.pp("conv").get(
            (channels, channels / groups, kernel_size),
            "weight_v",
        )?;
        let weight_g = vb.pp("conv").get((1, 1, kernel_size), "weight_g")?;
        let bias = vb.pp("conv").get(channels, "bias")?;

        // Reconstruct weight from weight_norm: w = g * v / ||v||
        // ||v|| computed over dims 0,1 for each kernel position → shape (1, 1, K)
        let v_norm = weight_v
            .sqr()?
            .sum_keepdim(0)?
            .sum_keepdim(1)?
            .sqrt()?;
        let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&(v_norm + 1e-12)?)?;

        let conv = Conv1d::new(
            weight,
            Some(bias),
            Conv1dConfig {
                padding: kernel_size / 2,
                groups,
                ..Default::default()
            },
        );

        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, T, C) → transpose for conv1d
        let h = x.transpose(1, 2)?; // (B, C, T)
        let h = self.conv.forward(&h)?;
        // Remove one sample from the right if we have padding overshoot
        let target_len = x.dim(1)?;
        let conv_len = h.dim(2)?;
        let h = if conv_len > target_len {
            h.narrow(2, 0, target_len)?
        } else {
            h
        };
        let h = h.gelu_erf()?;
        h.transpose(1, 2) // back to (B, T, C)
    }
}

/// 7-layer convolutional feature extractor from wav2vec2/WavLM.
///
/// All conv layers have NO bias. Only layer 0 has GroupNorm (num_groups=num_channels).
struct WavLMFeatureExtractor {
    conv_layers: Vec<WavLMConvLayer>,
}

impl WavLMFeatureExtractor {
    fn load(vb: VarBuilder) -> Result<Self> {
        let configs = [
            (1, 512, 10, 5),   // layer 0
            (512, 512, 3, 2),  // layer 1
            (512, 512, 3, 2),  // layer 2
            (512, 512, 3, 2),  // layer 3
            (512, 512, 3, 2),  // layer 4
            (512, 512, 2, 2),  // layer 5
            (512, 512, 2, 2),  // layer 6
        ];

        let mut conv_layers = Vec::with_capacity(7);
        for (i, (in_c, out_c, k, s)) in configs.iter().enumerate() {
            let layer = WavLMConvLayer::load(
                vb.pp(&format!("conv_layers.{i}")),
                *in_c,
                *out_c,
                *k,
                *s,
                i == 0,
            )?;
            conv_layers.push(layer);
        }

        Ok(Self { conv_layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, T) → need (B, 1, T) for conv
        let mut h = x.unsqueeze(1)?;
        for layer in &self.conv_layers {
            h = layer.forward(&h)?;
        }
        // h: (B, 512, T') → transpose to (B, T', 512)
        h.transpose(1, 2)
    }
}

struct WavLMConvLayer {
    conv: Conv1d,
    norm: Option<GroupNorm>,
}

impl WavLMConvLayer {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        has_norm: bool,
    ) -> Result<Self> {
        let weight = vb.pp("conv").get(
            (out_channels, in_channels, kernel_size),
            "weight",
        )?;
        let conv = Conv1d::new(
            weight,
            None,
            Conv1dConfig {
                stride,
                ..Default::default()
            },
        );

        let norm = if has_norm {
            Some(group_norm(out_channels, out_channels, 1e-5, vb.pp("layer_norm"))?)
        } else {
            None
        };

        Ok(Self { conv, norm })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv.forward(x)?;
        let h = if let Some(norm) = &self.norm {
            norm.forward(&h)?
        } else {
            h
        };
        h.gelu_erf()
    }
}

/// Feature projection: LayerNorm(512) + Linear(512, 768)
struct WavLMFeatureProjection {
    projection: Linear,
    layer_norm: candle_nn::LayerNorm,
}

impl WavLMFeatureProjection {
    fn load(vb: VarBuilder) -> Result<Self> {
        let projection = candle_nn::linear(512, 768, vb.pp("projection"))?;
        let layer_norm = candle_nn::layer_norm(512, 1e-5, vb.pp("layer_norm"))?;
        Ok(Self {
            projection,
            layer_norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.layer_norm.forward(x)?;
        self.projection.forward(&h)
    }
}

/// A single WavLM encoder layer with gated relative position bias.
struct WavLMEncoderLayer {
    attention: WavLMSelfAttention,
    layer_norm: candle_nn::LayerNorm,
    feed_forward: WavLMFeedForward,
    final_layer_norm: candle_nn::LayerNorm,
}

impl WavLMEncoderLayer {
    fn load(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        rel_attn_embed: Option<&Tensor>,
    ) -> Result<Self> {
        let attention = WavLMSelfAttention::load(
            vb.pp("attention"),
            dim,
            num_heads,
            rel_attn_embed,
        )?;
        let layer_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("layer_norm"))?;
        let feed_forward = WavLMFeedForward::load(vb.pp("feed_forward"), dim)?;
        let final_layer_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("final_layer_norm"))?;

        Ok(Self {
            attention,
            layer_norm,
            feed_forward,
            final_layer_norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // POST-NORM architecture (layer_norm_first=False in torchaudio WavLM)
        let residual = x;
        let h = self.attention.forward(x)?; // attention on raw input (no pre-norm)
        let h = (residual + h)?;
        let h = self.layer_norm.forward(&h)?; // post-norm after attention + residual

        let ffn_out = self.feed_forward.forward(&h)?;
        let h = (h + ffn_out)?;
        self.final_layer_norm.forward(&h) // post-norm after FFN + residual
    }
}

/// Multi-head self-attention with WavLM's gated relative position bias.
struct WavLMSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    // Gated relative position bias
    gru_rel_pos_const: Tensor,
    gru_rel_pos_linear: Linear,
    /// Shared relative attention embedding: (num_buckets, num_heads)
    rel_attn_embed: Option<Tensor>,
    num_buckets: usize,
    max_distance: usize,
}

impl WavLMSelfAttention {
    fn load(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        shared_rel_attn_embed: Option<&Tensor>,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let q_proj = candle_nn::linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(dim, dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(dim, dim, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(dim, dim, vb.pp("out_proj"))?;

        // Gated relative position bias
        let gru_rel_pos_const = vb.get((1, num_heads, 1, 1), "gru_rel_pos_const")?;
        // grep_linear / gru_rel_pos_linear: Linear(head_dim, 8)
        let gru_rel_pos_linear = candle_nn::linear(
            head_dim,
            8,
            vb.pp("gru_rel_pos_linear"),
        )?;

        // Use shared relative attention embedding (from layer 0) or this layer's own
        let rel_attn_embed = if let Ok(w) = vb.pp("rel_attn_embed").get((320, num_heads), "weight") {
            Some(w)
        } else {
            shared_rel_attn_embed.cloned()
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            gru_rel_pos_const,
            gru_rel_pos_linear,
            rel_attn_embed,
            num_buckets: 320,
            max_distance: 800,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape: (B, T, D) → (B, H, T, Dh)
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

        let scale = (self.head_dim as f64).sqrt();
        let mut attn = (q.matmul(&k.t()?.contiguous()?)? / scale)?;

        // Compute and apply gated relative position bias
        if let Some(ref rel_embed) = self.rel_attn_embed {
            // 1. Compute raw position bias: (1, H, T, T)
            let pos_bias = self.compute_position_bias(t, rel_embed)?;

            // 2. Gate the position bias using query-dependent gating
            let gated_bias = self.apply_gru_rel_pos(&pos_bias, &q)?;

            // 3. Add gated bias to attention scores
            attn = (attn + gated_bias)?;
        }

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        // (B, H, T, D/H) → (B, T, D)
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }

    /// Compute relative position bias from the embedding table.
    ///
    /// Uses T5-style bidirectional logarithmic bucketing:
    /// - First half of buckets (0..half) for negative offsets (key before query)
    /// - Second half (half..num_buckets) for positive offsets (key after query)
    /// - Within each half: first half exact, second half logarithmic
    fn compute_position_bias(
        &self,
        seq_len: usize,
        rel_embed: &Tensor,
    ) -> Result<Tensor> {
        let device = rel_embed.device();
        let num_buckets = self.num_buckets; // 320
        let max_distance = self.max_distance; // 800
        let half_buckets = num_buckets / 2; // 160
        let max_exact = half_buckets / 2; // 80

        let log_ratio_denom = (max_distance as f64 / max_exact as f64).ln();
        let large_range = (half_buckets - max_exact) as f64;

        let mut indices = Vec::with_capacity(seq_len * seq_len);
        for i in 0..seq_len {
            for j in 0..seq_len {
                let rel_pos = j as i64 - i as i64; // positive = key after query

                // Bidirectional: positive side gets offset by half_buckets
                let bucket_offset = if rel_pos > 0 { half_buckets } else { 0 };
                let abs_pos = rel_pos.unsigned_abs() as usize;

                let bucket = if abs_pos < max_exact {
                    // Small distances: identity mapping
                    abs_pos
                } else {
                    // Large distances: logarithmic mapping
                    let log_bucket = max_exact as f64
                        + (abs_pos as f64 / max_exact as f64).ln() / log_ratio_denom
                            * large_range;
                    log_bucket.min((half_buckets - 1) as f64) as usize
                };

                indices.push((bucket_offset + bucket).min(num_buckets - 1) as u32);
            }
        }

        let idx = Tensor::from_vec(indices, (seq_len * seq_len,), device)?;
        // rel_embed: (num_buckets, num_heads) → gather → (T*T, H)
        let bias = rel_embed.index_select(&idx, 0)?;
        // Reshape to (T, T, H) → permute to (H, T, T) → unsqueeze to (1, H, T, T)
        let bias = bias
            .reshape((seq_len, seq_len, self.num_heads))?
            .permute((2, 0, 1))?
            .unsqueeze(0)?;
        Ok(bias.to_dtype(DType::F32)?)
    }

    /// Apply WavLM's gated relative position modulation to position bias.
    ///
    /// Gate is query-dependent: linear(query) → reshape(2,4) → sum → sigmoid → formula
    /// Result: position_bias * gate_value, where gate_value ≈ 2.0 at initialization.
    fn apply_gru_rel_pos(
        &self,
        position_bias: &Tensor,
        query: &Tensor,
    ) -> Result<Tensor> {
        // query: (B, H, T, head_dim)
        // position_bias: (1, H, T, T)
        let (b, h, t, _dh) = query.dims4()?;

        // Linear projection: (B, H, T, head_dim) → (B, H, T, 8)
        let gate = self.gru_rel_pos_linear.forward(query)?;

        // Reshape to (B, H, T, 2, 4) → sum over last dim → (B, H, T, 2)
        let gate = gate.reshape((b, h, t, 2, 4))?;
        let gate = gate.sum(candle_core::D::Minus1)?; // (B, H, T, 2)

        // Sigmoid
        let gate = candle_nn::ops::sigmoid(&gate)?;

        // Split into gate_a and gate_b: each (B, H, T, 1)
        let gate_a = gate.narrow(3, 0, 1)?; // (B, H, T, 1)
        let gate_b = gate.narrow(3, 1, 1)?; // (B, H, T, 1)

        // gate_value = gate_a * (gate_b * gru_rel_pos_const - 1.0) + 2.0
        // gru_rel_pos_const: (1, H, 1, 1)
        let gate_value = (gate_b.broadcast_mul(&self.gru_rel_pos_const)? - 1.0)?;
        let gate_value = (gate_a.broadcast_mul(&gate_value)? + 2.0)?;
        // gate_value: (B, H, T, 1) — broadcast over key dimension

        // Modulate position bias: (1, H, T, T) * (B, H, T, 1) → (B, H, T, T)
        position_bias.broadcast_mul(&gate_value)
    }
}

/// WavLM feed-forward: Linear(768, 3072) → GELU → Linear(3072, 768)
struct WavLMFeedForward {
    intermediate_dense: Linear,
    output_dense: Linear,
}

impl WavLMFeedForward {
    fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        let ff_dim = dim * 4;
        let intermediate_dense = candle_nn::linear(dim, ff_dim, vb.pp("intermediate_dense"))?;
        let output_dense = candle_nn::linear(ff_dim, dim, vb.pp("output_dense"))?;
        Ok(Self {
            intermediate_dense,
            output_dense,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.intermediate_dense.forward(x)?;
        let h = h.gelu_erf()?;
        self.output_dense.forward(&h)
    }
}
