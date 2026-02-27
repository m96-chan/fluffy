use candle_core::{Result, Tensor};
use candle_nn::{conv1d, linear, Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

/// Global speaker encoder: SSL features → 128-dim speaker embedding.
///
/// Architecture:
/// 1. ConvNextBackbone(768→384, 4 layers, intermediate=1152)
///    - embed: Conv1d(768, 384, k=7) → norm: LayerNorm → 4× ConvNextBlock → final_layer_norm
/// 2. AttentiveStatsPool(384→128)
///    - attn: Conv1d(384,128,1)→Tanh→Conv1d(128,384,1) → softmax
///    - weighted mean + std → proj: Linear(768, 128) → norm: LayerNorm
pub struct GlobalEncoder {
    backbone: ConvNextBackbone,
    pool: AttentiveStatsPool,
}

impl GlobalEncoder {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let backbone = ConvNextBackbone::load(vb.pp("backbone"), 768, 384, 4, 1152)?;
        let pool = AttentiveStatsPool::load(vb.pp("pooling"), 384, 128)?;
        Ok(Self { backbone, pool })
    }

    /// Input: (1, T, 768) SSL features
    /// Output: (128,) speaker embedding
    pub fn forward(&self, ssl_features: &Tensor) -> Result<Tensor> {
        // (1, T, 768) → (1, 768, T) for conv
        let x = ssl_features.transpose(1, 2)?;
        let x = self.backbone.forward(&x)?; // (1, 384, T)
        let emb = self.pool.forward(&x)?; // (1, 128)
        emb.squeeze(0) // (128,)
    }
}

/// ConvNext backbone with channel projection + N ConvNext blocks.
///
/// Weight naming:
/// - `embed.{weight,bias}` — Conv1d(768, 384, 7)
/// - `norm.{weight,bias}` — LayerNorm after embed
/// - `convnext.N.*` — ConvNext blocks
/// - `final_layer_norm.{weight,bias}` — final LayerNorm
struct ConvNextBackbone {
    embed: Conv1d,
    norm: candle_nn::LayerNorm,
    blocks: Vec<ConvNextBlock>,
    final_layer_norm: candle_nn::LayerNorm,
}

impl ConvNextBackbone {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        dim: usize,
        num_layers: usize,
        intermediate_dim: usize,
    ) -> Result<Self> {
        let embed = conv1d(
            in_channels,
            dim,
            7, // kernel_size
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("embed"),
        )?;
        let norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm"))?;

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let block = ConvNextBlock::load(
                vb.pp(&format!("convnext.{i}")),
                dim,
                intermediate_dim,
            )?;
            blocks.push(block);
        }

        let final_layer_norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("final_layer_norm"))?;

        Ok(Self {
            embed,
            norm,
            blocks,
            final_layer_norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.embed.forward(x)?; // (B, dim, T)

        // LayerNorm expects (B, T, C) — transpose, norm, transpose back
        let h = h.transpose(1, 2)?;
        let h = self.norm.forward(&h)?;
        let mut h = h.transpose(1, 2)?;

        for block in &self.blocks {
            h = block.forward(&h)?;
        }

        // Final LayerNorm
        let h = h.transpose(1, 2)?;
        let h = self.final_layer_norm.forward(&h)?;
        Ok(h.transpose(1, 2)?)
    }
}

/// ConvNext block: depthwise conv → LayerNorm → pointwise up → GELU → pointwise down + residual.
/// Has layer scale parameter `gamma`.
///
/// Weight naming:
/// - `dwconv.{weight,bias}`
/// - `norm.{weight,bias}`
/// - `pwconv1.{weight,bias}`
/// - `pwconv2.{weight,bias}`
/// - `gamma` — layer scale [dim]
struct ConvNextBlock {
    dwconv: Conv1d,
    norm: candle_nn::LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNextBlock {
    fn load(vb: VarBuilder, dim: usize, intermediate_dim: usize) -> Result<Self> {
        let dwconv = conv1d(
            dim,
            dim,
            7,
            Conv1dConfig {
                padding: 3,
                groups: dim,
                ..Default::default()
            },
            vb.pp("dwconv"),
        )?;
        let norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm"))?;
        let pwconv1 = linear(dim, intermediate_dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(intermediate_dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(dim, "gamma")?;

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.dwconv.forward(x)?; // (B, C, T)

        // LayerNorm on channel dim: transpose to (B, T, C)
        let h = h.transpose(1, 2)?;
        let h = self.norm.forward(&h)?;

        // Pointwise (as linear on last dim)
        let h = self.pwconv1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.pwconv2.forward(&h)?;

        // Layer scale: multiply by gamma
        let h = h.broadcast_mul(&self.gamma)?;

        // Transpose back to (B, C, T)
        let h = h.transpose(1, 2)?;
        residual + h
    }
}

/// Attentive Statistics Pooling.
///
/// Architecture:
/// - attn = Sequential(Conv1d(in,128,1), Tanh, Conv1d(128,in,1))
/// - Attention-weighted mean and std → concat → Linear(2*in, out) → LayerNorm
///
/// Weight naming:
/// - `attn.0.{weight,bias}` — Conv1d(384, 128, 1)
/// - `attn.2.{weight,bias}` — Conv1d(128, 384, 1)
/// - `proj.{weight,bias}` — Linear(768, 128)
/// - `norm.{weight,bias}` — LayerNorm(128)
struct AttentiveStatsPool {
    attn_conv1: Conv1d, // Conv1d(in_dim, 128, 1)
    attn_conv2: Conv1d, // Conv1d(128, in_dim, 1)
    proj: Linear,
    norm: candle_nn::LayerNorm,
}

impl AttentiveStatsPool {
    fn load(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        let hidden = 128;
        let attn_conv1 = conv1d(
            in_dim,
            hidden,
            1,
            Default::default(),
            vb.pp("attn.0"),
        )?;
        let attn_conv2 = conv1d(
            hidden,
            in_dim,
            1,
            Default::default(),
            vb.pp("attn.2"),
        )?;

        // Concat of mean and std: 2*in_dim → out_dim
        let proj = linear(2 * in_dim, out_dim, vb.pp("proj"))?;
        let norm = candle_nn::layer_norm(out_dim, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            attn_conv1,
            attn_conv2,
            proj,
            norm,
        })
    }

    /// Input: (B, C, T) → Output: (B, out_dim)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute attention weights using Conv1d sequential
        // attn = Conv1d(C→128) → Tanh → Conv1d(128→C)
        let attn = self.attn_conv1.forward(x)?; // (B, 128, T)
        let attn = attn.tanh()?;
        let attn = self.attn_conv2.forward(&attn)?; // (B, C, T)

        // Softmax over T dimension (last dim)
        let attn = candle_nn::ops::softmax_last_dim(&attn)?; // (B, C, T)

        // Weighted mean: (B, C, T) * (B, C, T) → sum over T → (B, C)
        let weighted = (x * &attn)?;
        let mean = weighted.sum(2)?; // (B, C)

        // Weighted variance: E[x²] - E[x]²
        let x_sq = x.sqr()?;
        let weighted_sq = (x_sq * &attn)?;
        let mean_sq = weighted_sq.sum(2)?; // (B, C)
        let variance = (&mean_sq - mean.sqr()?)?;
        // Clamp variance to avoid sqrt of negative
        let std = variance.clamp(1e-8, f64::MAX)?.sqrt()?;

        // Concatenate mean and std: (B, 2*C)
        let stats = Tensor::cat(&[&mean, &std], 1)?;

        // Project + normalize
        let out = self.proj.forward(&stats)?;
        self.norm.forward(&out)
    }
}
