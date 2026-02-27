use candle_core::{DType, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Inverse STFT head: projects features → magnitude + phase → iSTFT → waveform.
///
/// Input: (B, C, T_stft) from upsampler
/// Output: (B, T_audio) waveform at target sample rate
pub struct IstftHead {
    proj: Linear,
    n_fft: usize,
    hop_length: usize,
    /// n_fft/2 + 1
    freq_bins: usize,
}

impl IstftHead {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        n_fft: usize,
        hop_length: usize,
    ) -> Result<Self> {
        let freq_bins = n_fft / 2 + 1;
        // Project to magnitude + phase: 2 * freq_bins
        let proj = linear(in_channels, 2 * freq_bins, vb.pp("out"))?;

        Ok(Self {
            proj,
            n_fft,
            hop_length,
            freq_bins,
        })
    }

    /// Forward: (B, C, T_stft) → (B, T_audio)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c, t_stft) = x.dims3()?;
        let device = x.device();

        // Transpose to (B, T_stft, C) for linear projection
        let x = x.transpose(1, 2)?;
        let projected = self.proj.forward(&x)?; // (B, T_stft, 2*freq_bins)

        // Split into magnitude and phase
        let mag = projected.narrow(2, 0, self.freq_bins)?;
        let phase = projected.narrow(2, self.freq_bins, self.freq_bins)?;

        // Convert to complex: exp(mag) * (cos(phase) + i*sin(phase))
        // Clamp magnitude to prevent excessively large values (matches Python)
        let mag_exp = mag.exp()?.clamp(0.0f32, 1e2f32)?;
        let real = (&mag_exp * phase.cos()?)?;
        let imag = (&mag_exp * phase.sin()?)?;

        // iSTFT on CPU using rustfft
        let real_data: Vec<Vec<Vec<f32>>> = (0..b)
            .map(|bi| {
                let real_b = real.get(bi).unwrap();
                (0..t_stft)
                    .map(|ti| {
                        real_b
                            .get(ti)
                            .unwrap()
                            .to_dtype(DType::F32)
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap()
                    })
                    .collect()
            })
            .collect();

        let imag_data: Vec<Vec<Vec<f32>>> = (0..b)
            .map(|bi| {
                let imag_b = imag.get(bi).unwrap();
                (0..t_stft)
                    .map(|ti| {
                        imag_b
                            .get(ti)
                            .unwrap()
                            .to_dtype(DType::F32)
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap()
                    })
                    .collect()
            })
            .collect();

        // Perform iSTFT per batch
        // "same" padding: trim (win_length - hop_length) / 2 from each end
        // Matches Python ISTFT with padding="same"
        let raw_audio_len = (t_stft - 1) * self.hop_length + self.n_fft;
        let trim = (self.n_fft - self.hop_length) / 2;
        let trimmed_len = raw_audio_len.saturating_sub(2 * trim);
        let mut all_audio = Vec::with_capacity(b * trimmed_len);

        for bi in 0..b {
            let audio = istft(
                &real_data[bi],
                &imag_data[bi],
                self.n_fft,
                self.hop_length,
            );
            // Trim center padding (removes boundary artifacts)
            let start = trim.min(audio.len());
            let end = audio.len().saturating_sub(trim).max(start);
            all_audio.extend_from_slice(&audio[start..end]);
        }

        let per_batch = all_audio.len() / b;
        Tensor::from_vec(all_audio, (b, per_batch), device)
    }
}

/// Perform inverse STFT using rustfft.
///
/// - `real_frames` / `imag_frames`: Vec of frames, each with freq_bins values
/// - Returns the reconstructed waveform
fn istft(
    real_frames: &[Vec<f32>],
    imag_frames: &[Vec<f32>],
    n_fft: usize,
    hop_length: usize,
) -> Vec<f32> {
    let num_frames = real_frames.len();
    let freq_bins = n_fft / 2 + 1;
    let audio_len = (num_frames - 1) * hop_length + n_fft;

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    let mut output = vec![0.0f32; audio_len];
    let mut window_sum = vec![0.0f32; audio_len];

    // Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
        .collect();

    for (frame_idx, (real, imag)) in real_frames.iter().zip(imag_frames.iter()).enumerate() {
        // Build full complex spectrum (mirror conjugate for negative frequencies)
        let mut spectrum: Vec<Complex<f32>> = Vec::with_capacity(n_fft);
        for i in 0..freq_bins {
            spectrum.push(Complex::new(real[i], imag[i]));
        }
        // Mirror: bin n_fft/2+1 .. n_fft-1 = conjugate of bin n_fft/2-1 .. 1
        for i in (1..freq_bins - 1).rev() {
            spectrum.push(Complex::new(real[i], -imag[i]));
        }

        // In-place iFFT
        ifft.process(&mut spectrum);

        // Scale by 1/n_fft (rustfft doesn't normalize)
        let scale = 1.0 / n_fft as f32;

        let offset = frame_idx * hop_length;
        for i in 0..n_fft {
            if offset + i < audio_len {
                output[offset + i] += spectrum[i].re * scale * window[i];
                window_sum[offset + i] += window[i] * window[i];
            }
        }
    }

    // Normalize by window sum (overlap-add normalization)
    for i in 0..audio_len {
        if window_sum[i] > 1e-8 {
            output[i] /= window_sum[i];
        }
    }

    output
}
