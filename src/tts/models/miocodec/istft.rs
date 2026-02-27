use candle_core::{DType, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use std::f64::consts::PI;

/// Inverse STFT head: projects features → magnitude + phase → iSTFT → waveform.
///
/// All operations run on the model's device (GPU when available).
/// The iDFT is computed via DFT basis matrix multiplication (O(N²), N=392 is small
/// enough that this is faster than transferring to CPU for FFT).
/// Overlap-add is done via `conv_transpose1d` with an identity kernel.
///
/// Input: (B, C, T_stft) from upsampler
/// Output: (B, T_audio) waveform at target sample rate
pub struct IstftHead {
    proj: Linear,
    n_fft: usize,
    hop_length: usize,
    freq_bins: usize,
    // Precomputed tensors for GPU iDFT + overlap-add
    cos_basis: Tensor,   // (N, N) — iDFT cosine basis
    sin_basis: Tensor,   // (N, N) — iDFT sine basis
    mirror_idx: Tensor,  // (N,) u32 — conjugate mirror indices
    mirror_sign: Tensor, // (1, 1, N) — sign flip for imaginary part
    hann_window: Tensor, // (1, 1, N) — Hann window
    fold_kernel: Tensor, // (N, 1, N) — identity matrix for conv_transpose1d fold
}

impl IstftHead {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        n_fft: usize,
        hop_length: usize,
    ) -> Result<Self> {
        let freq_bins = n_fft / 2 + 1;
        let proj = linear(in_channels, 2 * freq_bins, vb.pp("out"))?;
        let device = vb.device();
        let n = n_fft;

        // 1. iDFT basis matrices: cos_basis[k, n] = cos(2πkn/N) / N
        //    For matmul: result[b, t, n] = Σ_k input[b, t, k] * basis[k, n]
        let mut cos_data = vec![0.0f32; n * n];
        let mut sin_data = vec![0.0f32; n * n];
        let inv_n = 1.0 / n as f64;
        for k in 0..n {
            for out_n in 0..n {
                let angle = 2.0 * PI * (k as f64) * (out_n as f64) * inv_n;
                cos_data[k * n + out_n] = (angle.cos() * inv_n) as f32;
                sin_data[k * n + out_n] = (angle.sin() * inv_n) as f32;
            }
        }
        let cos_basis = Tensor::from_vec(cos_data, (n, n), device)?;
        let sin_basis = Tensor::from_vec(sin_data, (n, n), device)?;

        // 2. Mirror indices for conjugate symmetry: [0,1,...,196,195,...,1]
        let mirror_indices: Vec<u32> = (0..n)
            .map(|i| {
                if i < freq_bins {
                    i as u32
                } else {
                    (n - i) as u32
                }
            })
            .collect();
        let mirror_idx = Tensor::from_vec(mirror_indices, (n,), device)?;

        // 3. Sign tensor for imaginary part: +1 for original bins, -1 for mirrored (conjugate)
        let signs: Vec<f32> = (0..n)
            .map(|i| if i < freq_bins { 1.0f32 } else { -1.0f32 })
            .collect();
        let mirror_sign = Tensor::from_vec(signs, (1, 1, n), device)?;

        // 4. Hann window: w[i] = 0.5 * (1 - cos(2πi/N))
        let window: Vec<f32> = (0..n)
            .map(|i| {
                let angle = 2.0 * PI * (i as f64) * inv_n;
                (0.5 * (1.0 - angle.cos())) as f32
            })
            .collect();
        let hann_window = Tensor::from_vec(window, (1, 1, n), device)?;

        // 5. Fold kernel: identity matrix (N, 1, N) for overlap-add via conv_transpose1d
        //    kernel[i, 0, j] = δ(i, j) → places input channel i at offset i in the output
        let fold_kernel = Tensor::eye(n, DType::F32, device)?.reshape((n, 1, n))?;

        Ok(Self {
            proj,
            n_fft,
            hop_length,
            freq_bins,
            cos_basis,
            sin_basis,
            mirror_idx,
            mirror_sign,
            hann_window,
            fold_kernel,
        })
    }

    /// Forward: (B, C, T_stft) → (B, T_audio)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, t_stft) = x.dims3()?;
        let device = x.device();

        // 1. Linear projection (on device)
        let x = x.transpose(1, 2)?;
        let projected = self.proj.forward(&x)?; // (B, T_stft, 2*freq_bins)
        let projected = projected.to_dtype(DType::F32)?;

        // 2. Split into magnitude and phase, compute complex spectrum
        let mag = projected.narrow(2, 0, self.freq_bins)?;
        let phase = projected.narrow(2, self.freq_bins, self.freq_bins)?;
        let mag_exp = mag.exp()?.clamp(0.0f32, 1e2f32)?;
        let real = (&mag_exp * phase.cos()?)?; // (B, T, freq_bins)
        let imag = (&mag_exp * phase.sin()?)?;

        // 3. Mirror to full spectrum (conjugate symmetry)
        let real_full = real.index_select(&self.mirror_idx, 2)?; // (B, T, N)
        let imag_full = imag
            .index_select(&self.mirror_idx, 2)?
            .broadcast_mul(&self.mirror_sign)?; // negate mirrored bins

        // 4. iDFT via matrix multiplication (broadcast_matmul handles 3D×2D)
        //    x[n] = (1/N) Σ_k (Re[X[k]]·cos(2πkn/N) - Im[X[k]]·sin(2πkn/N))
        let time_frames = (real_full.broadcast_matmul(&self.cos_basis)?
            - imag_full.broadcast_matmul(&self.sin_basis)?)?; // (B, T, N)

        // 5. Apply Hann window
        let windowed = time_frames.broadcast_mul(&self.hann_window)?; // (B, T, N)

        // 6. Overlap-add via conv_transpose1d with identity kernel
        //    Input (B, N, T) → Output (B, 1, audio_len)
        let ola_input = windowed.transpose(1, 2)?.contiguous()?; // (B, N, T)
        let audio = ola_input.conv_transpose1d(
            &self.fold_kernel,
            /*padding=*/ 0,
            /*output_padding=*/ 0,
            /*stride=*/ self.hop_length,
            /*dilation=*/ 1,
            /*groups=*/ 1,
        )?; // (B, 1, audio_len)

        // 7. Window normalization: divide by overlap-added squared window
        //    Build (1, N, T_stft) where each column is w[n]², then fold
        let win_sq = self.hann_window.sqr()?; // (1, 1, N)
        let win_sq_col = win_sq.reshape((1, self.n_fft, 1))?; // (1, N, 1)
        let ones = Tensor::ones((1, 1, t_stft), DType::F32, device)?;
        let win_sq_expanded = win_sq_col.broadcast_mul(&ones)?.contiguous()?; // (1, N, T_stft)
        let win_sum = win_sq_expanded.conv_transpose1d(
            &self.fold_kernel,
            0,
            0,
            self.hop_length,
            1,
            1,
        )?; // (1, 1, audio_len)
        let audio = (audio / win_sum.clamp(1e-8f32, f32::MAX)?)?;

        // 8. Trim "same" padding (removes boundary artifacts)
        let trim = (self.n_fft - self.hop_length) / 2;
        let audio_len = audio.dim(2)?;
        let trimmed_len = audio_len - 2 * trim;
        let audio = audio.narrow(2, trim, trimmed_len)?;

        audio.squeeze(1) // (B, trimmed_len)
    }
}
