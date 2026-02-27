use candle_core::{Result, Tensor};
use candle_nn::{linear, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module, VarBuilder};

use super::resnet::ResNetBlock;
use super::snake::SnakeBeta;

/// Upsampler block: ConvTranspose1d stages → SnakeBeta → ResNet, then out_proj + out_snake.
///
/// Weight naming (matching MioCodec safetensors):
/// - `upsample_layers.N.parametrizations.weight.original0` (g), `original1` (v), `bias`
/// - `snake_activations.N.{alpha,beta}`
/// - `resnet_blocks.N.{blocks.M.norm1/norm2/conv1/conv2}`
/// - `out_proj.{weight,bias}` (Linear 128→512)
/// - `out_snake.{alpha,beta}` (SnakeBeta on 512 channels)
///
/// Channel halving: 512 → 256 → 128, then out_proj back to 512.
pub struct UpSamplerBlock {
    upsample_layers: Vec<ConvTranspose1d>,
    snake_activations: Vec<SnakeBeta>,
    resnet_blocks: Vec<ResNetBlock>,
    out_proj: Linear,
    out_snake: SnakeBeta,
}

impl UpSamplerBlock {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        factors: &[usize],
        kernel_sizes: &[usize],
    ) -> Result<Self> {
        let mut upsample_layers = Vec::with_capacity(factors.len());
        let mut snake_activations = Vec::with_capacity(factors.len());
        let mut resnet_blocks = Vec::with_capacity(factors.len());

        let mut ch = in_channels;

        for (i, (&factor, &kernel_size)) in factors.iter().zip(kernel_sizes.iter()).enumerate() {
            let out_ch = ch / 2;

            // Load ConvTranspose1d with weight parametrization (weight_norm)
            let conv = load_parametrized_conv_transpose(
                vb.pp(&format!("upsample_layers.{i}")),
                ch,
                out_ch,
                kernel_size,
                factor,
            )?;
            upsample_layers.push(conv);

            let snake = SnakeBeta::load(vb.pp(&format!("snake_activations.{i}")), out_ch)?;
            snake_activations.push(snake);

            // One ResNet block per stage (flat naming, no blocks.N wrapper)
            let resnet = ResNetBlock::load(
                vb.pp(&format!("resnet_blocks.{i}")),
                out_ch,
                3,
                32,
            )?;
            resnet_blocks.push(resnet);

            ch = out_ch;
        }

        // Output projection: Linear(final_ch, in_channels)
        let out_proj = linear(ch, in_channels, vb.pp("out_proj"))?;
        let out_snake = SnakeBeta::load(vb.pp("out_snake"), in_channels)?;

        Ok(Self {
            upsample_layers,
            snake_activations,
            resnet_blocks,
            out_proj,
            out_snake,
        })
    }

    /// Forward: (B, 512, T) → (B, 512, T * product(factors))
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();

        for i in 0..self.upsample_layers.len() {
            h = self.upsample_layers[i].forward(&h)?;
            h = self.snake_activations[i].forward(&h)?;
            h = self.resnet_blocks[i].forward(&h)?;  // single block, not stack
        }

        // h is now (B, final_ch, T*9), e.g. (B, 128, T*9)
        // Project back to in_channels via Linear
        let (b, _c, t) = h.dims3()?;
        let h = h.transpose(1, 2)?; // (B, T, 128)
        let h = self.out_proj.forward(&h)?; // (B, T, 512)
        let h = h.transpose(1, 2)?; // (B, 512, T)

        // Apply SnakeBeta activation
        let h = self.out_snake.forward(&h)?;

        // Verify output shape
        debug_assert_eq!(h.dims3()?.0, b);
        debug_assert_eq!(h.dims3()?.2, t);

        Ok(h)
    }
}

/// Load a ConvTranspose1d with weight parametrization (weight_norm).
///
/// PyTorch stores:
/// - `parametrizations.weight.original0` (g): [in_ch, 1, 1] magnitude
/// - `parametrizations.weight.original1` (v): [in_ch, out_ch, kernel] direction
/// - `bias`: [out_ch]
///
/// Reconstruction: weight = g * v / ||v|| (norm over out_ch and kernel dims)
fn load_parametrized_conv_transpose(
    vb: VarBuilder,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
) -> Result<ConvTranspose1d> {
    let param_vb = vb.pp("parametrizations.weight");

    let weight_g = param_vb.get((in_channels, 1, 1), "original0")?;
    let weight_v =
        param_vb.get((in_channels, out_channels, kernel_size), "original1")?;

    // Compute ||v|| per in_channel: sqrt(sum(v²)) over (out_ch, kernel) dims
    let v_norm = weight_v
        .sqr()?
        .sum_keepdim(2)? // sum over kernel
        .sum_keepdim(1)? // sum over out_ch
        .sqrt()?; // (in_ch, 1, 1)

    // weight = g * v / ||v||
    let weight = weight_g.broadcast_div(&v_norm)?.broadcast_mul(&weight_v)?;

    let bias = vb.get(out_channels, "bias").ok();

    let padding = (kernel_size - stride) / 2;
    let cfg = ConvTranspose1dConfig {
        stride,
        padding,
        ..Default::default()
    };

    Ok(ConvTranspose1d::new(weight, bias, cfg))
}
