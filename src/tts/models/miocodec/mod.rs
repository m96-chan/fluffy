pub mod fsq;
pub mod istft;
pub mod resnet;
pub mod snake;
pub mod transformer;
pub mod upsampler;

use candle_core::{Result, Tensor};
use candle_nn::{conv_transpose1d, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

use super::config::MioCodecConfig;
use fsq::FsqDecode;
use istft::IstftHead;
use resnet::ResNetStack;
use transformer::Transformer;
use upsampler::UpSamplerBlock;

/// MioCodec decoder: codec tokens + speaker embedding → waveform.
pub struct MioCodecDecoder {
    /// FSQ index → 768-dim embedding
    pub fsq: FsqDecode,
    /// Transformer: 768 dim, output_proj 768→512
    pub wave_prenet: Transformer,
    /// ConvTranspose1d: 512→512, stride=2 (25Hz→50Hz)
    pub wave_conv_upsample: ConvTranspose1d,
    /// ResNetStack before wave_decoder
    pub wave_prior_net: ResNetStack,
    /// Transformer with AdaLN-Zero, conditioned on speaker embedding
    pub wave_decoder: Transformer,
    /// ResNetStack after wave_decoder
    pub wave_post_net: ResNetStack,
    /// SnakeBeta upsampler: 50Hz→450Hz
    pub upsampler: UpSamplerBlock,
    /// ISTFT head: Linear→mag/phase→iSTFT
    pub istft_head: IstftHead,
    pub config: MioCodecConfig,
}

impl MioCodecDecoder {
    pub fn load(vb: VarBuilder, cfg: &MioCodecConfig) -> Result<Self> {
        let fsq = FsqDecode::load(vb.pp("local_quantizer"), &cfg.fsq_levels)?;

        // wave_prenet: dim=768, ff_dim=2048, output_proj 768→512
        let wave_prenet = Transformer::load(
            vb.pp("wave_prenet"),
            768,               // dim
            2048,              // ff_dim (768 * 8/3)
            Some(512),         // output_dim (output_proj)
            6,                 // n_layers
            12,                // n_heads
            65,                // window_size
            10000.0,           // rope_theta
            None,              // no AdaLN
        )?;

        let wave_conv_upsample = conv_transpose1d(
            512,
            512,
            2, // kernel_size
            ConvTranspose1dConfig {
                stride: 2,
                ..Default::default()
            },
            vb.pp("wave_conv_upsample"),
        )?;

        let wave_prior_net = ResNetStack::load(
            vb.pp("wave_prior_net"),
            512,
            cfg.wave_resnet_num_blocks,
            cfg.wave_resnet_kernel_size,
            cfg.wave_resnet_num_groups,
        )?;

        // wave_decoder: dim=512, ff_dim=1536, AdaLN-Zero with cond_dim=128
        let wave_decoder = Transformer::load(
            vb.pp("wave_decoder"),
            512,                // dim
            1536,               // ff_dim (512 * 3)
            None,               // no output projection
            8,                  // n_layers
            8,                  // n_heads
            65,                 // window_size
            10000.0,            // rope_theta
            Some(128),          // AdaLN-Zero condition dim (speaker embedding)
        )?;

        let wave_post_net = ResNetStack::load(
            vb.pp("wave_post_net"),
            512,
            cfg.wave_resnet_num_blocks,
            cfg.wave_resnet_kernel_size,
            cfg.wave_resnet_num_groups,
        )?;

        let upsampler = UpSamplerBlock::load(
            vb.pp("wave_upsampler"),
            512,
            &cfg.wave_upsampler_factors,
            &cfg.wave_upsampler_kernel_sizes,
        )?;

        let istft_head = IstftHead::load(
            vb.pp("istft_head"),
            512,
            cfg.n_fft,
            cfg.hop_length,
        )?;

        Ok(Self {
            fsq,
            wave_prenet,
            wave_conv_upsample,
            wave_prior_net,
            wave_decoder,
            wave_post_net,
            upsampler,
            istft_head,
            config: cfg.clone(),
        })
    }

    /// Decode codec token indices + speaker embedding → waveform f32 samples.
    ///
    /// - `token_indices`: shape (seq_len,) u32 indices 0..12799
    /// - `speaker_embedding`: shape (128,) f32 speaker embedding from GlobalEncoder
    pub fn forward_wave(
        &self,
        token_indices: &Tensor,
        speaker_embedding: &Tensor,
    ) -> Result<Tensor> {
        // FSQ decode: index → 5-dim → Linear(5, 768) → (1, seq, 768)
        let x = self.fsq.forward(token_indices)?;
        let x = x.unsqueeze(0)?; // (1, seq, 768)

        // wave_prenet: Transformer 768→512 → (1, seq, 512)
        let x = self.wave_prenet.forward(&x, None)?;

        // Conv transpose: (1, 512, seq) → (1, 512, seq*2)
        let x = x.transpose(1, 2)?;
        let x = self.wave_conv_upsample.forward(&x)?;

        // Interpolate (linear) to STFT length
        let stft_length = self.compute_stft_length(token_indices)?;
        let x = linear_interpolate(&x, stft_length)?;

        // wave_prior_net: ResNetStack
        let x = self.wave_prior_net.forward(&x)?;

        // wave_decoder: Transformer with AdaLN-Zero, conditioned on speaker embedding
        let x = x.transpose(1, 2)?;
        let speaker_emb = speaker_embedding.unsqueeze(0)?;
        let x = self.wave_decoder.forward(&x, Some(&speaker_emb))?;
        let x = x.transpose(1, 2)?;

        // wave_post_net: ResNetStack
        let x = self.wave_post_net.forward(&x)?;

        // UpSampler: (1, 512, stft_len) → (1, 512, stft_len*9)
        let x = self.upsampler.forward(&x)?;

        // ISTFTHead: → waveform
        let wav = self.istft_head.forward(&x)?;

        // Squeeze batch dim and return 1D waveform
        wav.squeeze(0)
    }

    fn compute_stft_length(&self, token_indices: &Tensor) -> Result<usize> {
        let seq_len = token_indices.dim(0)?;
        Ok(seq_len * self.config.wave_upsample_factor)
    }
}

/// Linear interpolation along the last dimension.
pub fn linear_interpolate(x: &Tensor, target_len: usize) -> Result<Tensor> {
    let (_b, _c, src_len) = x.dims3()?;
    if src_len == target_len {
        return Ok(x.clone());
    }

    let device = x.device();
    let dtype = x.dtype();

    // Create interpolation indices
    let scale = (src_len - 1) as f64 / (target_len - 1).max(1) as f64;
    let mut indices_lo = Vec::with_capacity(target_len);
    let mut indices_hi = Vec::with_capacity(target_len);
    let mut weights = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let src_pos = i as f64 * scale;
        let lo = src_pos.floor() as usize;
        let hi = (lo + 1).min(src_len - 1);
        let w = (src_pos - lo as f64) as f32;
        indices_lo.push(lo as u32);
        indices_hi.push(hi as u32);
        weights.push(w);
    }

    let idx_lo = Tensor::from_vec(indices_lo, target_len, device)?;
    let idx_hi = Tensor::from_vec(indices_hi, target_len, device)?;
    let w = Tensor::from_vec(weights, target_len, device)?.to_dtype(dtype)?;

    // x shape: (B, C, src_len)
    // Gather along dim 2
    let x_lo = x.index_select(&idx_lo, 2)?;
    let x_hi = x.index_select(&idx_hi, 2)?;

    // Broadcast weight: (1, 1, target_len)
    let w = w.unsqueeze(0)?.unsqueeze(0)?;
    let one_minus_w = (1.0f64 - &w)?;

    // candle's * doesn't auto-broadcast; use broadcast_mul
    x_lo.broadcast_mul(&one_minus_w)? + x_hi.broadcast_mul(&w)?
}
