use candle_core::{Result, Tensor};
use candle_nn::{conv1d, group_norm, Conv1d, Conv1dConfig, GroupNorm, Module, VarBuilder};

/// ResNet stack: N residual blocks with GroupNorm → SiLU → Conv1d.
pub struct ResNetStack {
    blocks: Vec<ResNetBlock>,
}

impl ResNetStack {
    pub fn load(
        vb: VarBuilder,
        channels: usize,
        num_blocks: usize,
        kernel_size: usize,
        num_groups: usize,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let block = ResNetBlock::load(
                vb.pp(&format!("blocks.{i}")),
                channels,
                kernel_size,
                num_groups,
            )?;
            blocks.push(block);
        }
        Ok(Self { blocks })
    }

    /// Forward: (B, C, T) → (B, C, T)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        Ok(h)
    }
}

/// Single residual block: GroupNorm → SiLU → Conv1d (×2) + skip connection.
pub struct ResNetBlock {
    norm1: GroupNorm,
    conv1: Conv1d,
    norm2: GroupNorm,
    conv2: Conv1d,
}

impl ResNetBlock {
    pub fn load(
        vb: VarBuilder,
        channels: usize,
        kernel_size: usize,
        num_groups: usize,
    ) -> Result<Self> {
        let padding = kernel_size / 2;
        let norm1 = group_norm(num_groups, channels, 1e-5, vb.pp("norm1"))?;
        let conv1 = conv1d(
            channels,
            channels,
            kernel_size,
            Conv1dConfig {
                padding,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let norm2 = group_norm(num_groups, channels, 1e-5, vb.pp("norm2"))?;
        let conv2 = conv1d(
            channels,
            channels,
            kernel_size,
            Conv1dConfig {
                padding,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv2.forward(&h)?;
        x + h
    }
}
