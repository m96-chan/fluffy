use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// SnakeBeta activation: x + (1/β) * sin²(α * x)
///
/// Parameters are stored in **log scale** (matching PyTorch BigVGAN convention).
/// `alpha_actual = exp(alpha_stored)`, `beta_actual = exp(beta_stored)`.
pub struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
}

impl SnakeBeta {
    pub fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        let alpha = vb.get(channels, "alpha")?;
        let beta = vb.get(channels, "beta")?;
        Ok(Self { alpha, beta })
    }

    /// Forward: input shape (B, C, T)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Reshape for broadcasting: (C,) → (1, C, 1)
        // exp() because parameters are stored in log scale
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?.exp()?;
        let beta = self.beta.unsqueeze(0)?.unsqueeze(2)?.exp()?;

        // sin²(α * x) = sin(α * x)²
        let sin_val = (x.broadcast_mul(&alpha))?.sin()?;
        let sin_sq = sin_val.sqr()?;

        // x + 1/(β + ε) * sin²(α * x)
        let inv_beta = (beta + 1e-9f64)?.recip()?;
        x + sin_sq.broadcast_mul(&inv_beta)?
    }
}
