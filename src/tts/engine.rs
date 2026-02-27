use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tracing::info;

use crate::error::AppError;

use super::models::config::{Lfm2Config, MioCodecConfig};
use super::models::download::{ensure_models_downloaded, ModelPaths};
use super::models::lfm2::{generate, GenerateParams, Lfm2ForCausalLM};
use super::models::miocodec::MioCodecDecoder;
use super::models::wavlm::{GlobalEncoder, WavLM};

/// TTS engine that owns all 3 models and performs end-to-end synthesis.
pub struct TtsEngine {
    lfm2: Lfm2ForCausalLM,
    miocodec: MioCodecDecoder,
    tokenizer: tokenizers::Tokenizer,
    speaker_embedding: Tensor,
    device: Device,
    lfm2_config: Lfm2Config,
}

impl TtsEngine {
    /// Initialize the TTS engine:
    /// 1. Download models from HuggingFace (if needed)
    /// 2. Load LFM2, MioCodec, WavLM onto CUDA
    /// 3. Compute speaker embedding from reference wav
    /// 4. Release WavLM to free VRAM
    pub async fn initialize(reference_wav: &Path) -> Result<Self, AppError> {
        info!("Initializing TTS engine...");

        // Step 1: Ensure models are downloaded
        let paths = ensure_models_downloaded().await?;

        // Step 2-4: Load models on a blocking thread (heavy GPU work)
        let reference_wav = reference_wav.to_path_buf();
        let engine = tokio::task::spawn_blocking(move || {
            Self::load_models(paths, &reference_wav)
        })
        .await
        .map_err(|e| AppError::Tts(format!("TTS init join error: {e}")))?
        .map_err(|e| AppError::Tts(format!("TTS init error: {e}")))?;

        info!("TTS engine initialized successfully");
        Ok(engine)
    }

    fn load_models(paths: ModelPaths, reference_wav: &Path) -> Result<Self, AppError> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| AppError::Tts(format!("CUDA device: {e}")))?;
        info!("TTS using device: {:?}", device);

        // Load LFM2 config
        let config_str = std::fs::read_to_string(&paths.lfm2_config)
            .map_err(|e| AppError::Tts(format!("Read LFM2 config: {e}")))?;
        let lfm2_config: Lfm2Config = serde_json::from_str(&config_str)
            .map_err(|e| AppError::Tts(format!("Parse LFM2 config: {e}")))?;

        info!(
            "LFM2 config: {} layers, {} hidden, {} vocab",
            lfm2_config.num_hidden_layers, lfm2_config.hidden_size, lfm2_config.vocab_size
        );

        // Load LFM2 model — BF16 on CUDA for VRAM savings, F32 on CPU
        let lfm2_dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        info!("Loading LFM2 model ({} files, {:?})...", paths.lfm2_safetensors.len(), lfm2_dtype);
        let lfm2_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&paths.lfm2_safetensors, lfm2_dtype, &device)
                .map_err(|e| AppError::Tts(format!("LFM2 safetensors: {e}")))?
        };
        let lfm2 = Lfm2ForCausalLM::load(lfm2_vb, &lfm2_config)
            .map_err(|e| AppError::Tts(format!("LFM2 load: {e}")))?;
        info!("LFM2 loaded");

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(&paths.lfm2_tokenizer)
            .map_err(|e| AppError::Tts(format!("Tokenizer: {e}")))?;
        info!("Tokenizer loaded ({} tokens)", tokenizer.get_vocab_size(true));

        // Load MioCodec config
        let miocodec_config_str = std::fs::read_to_string(&paths.miocodec_config)
            .map_err(|e| AppError::Tts(format!("Read MioCodec config: {e}")))?;
        let miocodec_config = MioCodecConfig::from_yaml(&miocodec_config_str);
        info!("MioCodec config: {}Hz, n_fft={}", miocodec_config.sample_rate, miocodec_config.n_fft);

        // Load MioCodec
        info!("Loading MioCodec...");
        let miocodec_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[paths.miocodec_safetensors.clone()],
                DType::F32,
                &device,
            )
            .map_err(|e| AppError::Tts(format!("MioCodec safetensors: {e}")))?
        };
        let miocodec = MioCodecDecoder::load(miocodec_vb, &miocodec_config)
            .map_err(|e| AppError::Tts(format!("MioCodec load: {e}")))?;
        info!("MioCodec loaded");

        // Load WavLM (temporary — will be freed after computing speaker embedding)
        // WavLM only has pytorch_model.bin, not safetensors
        info!("Loading WavLM...");
        let wavlm_vb = VarBuilder::from_pth(&paths.wavlm_pth, DType::F32, &device)
            .map_err(|e| AppError::Tts(format!("WavLM pth: {e}")))?;
        let wavlm = WavLM::load(wavlm_vb, &miocodec_config.global_ssl_layers)
            .map_err(|e| AppError::Tts(format!("WavLM load: {e}")))?;

        // Load GlobalEncoder
        let global_encoder_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[paths.miocodec_safetensors],
                DType::F32,
                &device,
            )
            .map_err(|e| AppError::Tts(format!("GlobalEncoder safetensors: {e}")))?
        };
        let global_encoder = GlobalEncoder::load(global_encoder_vb.pp("global_encoder"))
            .map_err(|e| AppError::Tts(format!("GlobalEncoder load: {e}")))?;

        // Compute speaker embedding from reference wav
        info!("Computing speaker embedding from {}", reference_wav.display());
        let speaker_embedding = compute_speaker_embedding(
            reference_wav,
            &wavlm,
            &global_encoder,
            &device,
        )?;
        info!("Speaker embedding computed: {:?}", speaker_embedding.shape());
        // Debug: print first 10 values for comparison with Python reference
        if let Ok(vals) = speaker_embedding.to_vec1::<f32>() {
            let first10: Vec<String> = vals.iter().take(10).map(|v| format!("{v:.6}")).collect();
            info!("Speaker emb first10: [{}]", first10.join(", "));
            let rms = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
            info!("Speaker emb rms={:.6}", rms);
        }

        // WavLM and GlobalEncoder are dropped here, freeing VRAM
        drop(wavlm);
        drop(global_encoder);
        info!("WavLM released (VRAM freed)");

        Ok(Self {
            lfm2,
            miocodec,
            tokenizer,
            speaker_embedding,
            device,
            lfm2_config: lfm2_config,
        })
    }

    /// Synthesize speech from text. Returns PCM f32 samples at the codec's sample rate.
    pub fn synthesize_blocking(&self, text: &str) -> Result<Vec<f32>, AppError> {
        // Step 1: Build prompt and tokenize
        let prompt = super::models::lfm2::generate::build_prompt(text);
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| AppError::Tts(format!("Tokenize: {e}")))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        info!("Tokenized '{}' → {} tokens", text, input_ids.len());

        // Step 2: Generate codec tokens with LFM2
        let params = GenerateParams {
            eos_token_id: self.lfm2_config.eos_token_id,
            ..GenerateParams::default()
        };
        let generated = generate(&self.lfm2, &input_ids, &params, &self.device)
            .map_err(|e| AppError::Tts(format!("LFM2 generate: {e}")))?;

        // Debug: log first/last generated tokens for diagnosis
        let first_n: Vec<String> = generated.iter().take(10)
            .map(|&id| {
                let tok = self.tokenizer.id_to_token(id).unwrap_or_default();
                format!("{id}={tok}")
            }).collect();
        let last_n: Vec<String> = generated.iter().rev().take(5).rev()
            .map(|&id| {
                let tok = self.tokenizer.id_to_token(id).unwrap_or_default();
                format!("{id}={tok}")
            }).collect();
        info!("First tokens: [{}]", first_n.join(", "));
        info!("Last tokens: [{}]", last_n.join(", "));

        // Step 3: Extract codec token indices from generated tokens
        let codec_tokens =
            super::models::lfm2::generate::extract_codec_tokens(&generated, &self.tokenizer);
        info!("Generated {} codec tokens from {} raw tokens", codec_tokens.len(), generated.len());

        if codec_tokens.is_empty() {
            return Err(AppError::Tts("No codec tokens generated".into()));
        }

        // Step 4: Decode with MioCodec
        let num_tokens = codec_tokens.len();
        let token_tensor = Tensor::from_vec(codec_tokens, (num_tokens,), &self.device)
            .map_err(|e| AppError::Tts(format!("Token tensor: {e}")))?;

        let waveform = self
            .miocodec
            .forward_wave(&token_tensor, &self.speaker_embedding)
            .map_err(|e| AppError::Tts(format!("MioCodec decode: {e}")))?;

        // Convert to f32 Vec
        let pcm: Vec<f32> = waveform
            .to_dtype(DType::F32)
            .map_err(|e| AppError::Tts(format!("To f32: {e}")))?
            .to_vec1()
            .map_err(|e| AppError::Tts(format!("To vec: {e}")))?;

        // Peak-normalize to -1dBFS (0.89) for audible playback
        let peak = pcm.iter().copied().fold(0.0f32, |a, x| a.max(x.abs()));
        let pcm = if peak > 1e-6 {
            let gain = 0.89 / peak;
            pcm.into_iter().map(|x| x * gain).collect::<Vec<f32>>()
        } else {
            pcm
        };

        let rms = (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len() as f32).sqrt();
        info!("Synthesized {} samples ({:.2}s): peak_raw={peak:.4}, rms={rms:.4}",
            pcm.len(), pcm.len() as f32 / 44100.0);
        Ok(pcm)
    }
}

/// Load a WAV file and resample to 16kHz mono for WavLM input.
fn load_and_resample_wav(path: &Path, device: &Device) -> Result<Tensor, AppError> {
    use rodio::Source;
    use std::io::{BufReader, Cursor};

    let wav_bytes = std::fs::read(path)
        .map_err(|e| AppError::Tts(format!("Read reference wav: {e}")))?;

    let cursor = Cursor::new(wav_bytes);
    let decoder = rodio::Decoder::new(BufReader::new(cursor))
        .map_err(|e| AppError::Tts(format!("Decode reference wav: {e}")))?;

    let src_rate = decoder.sample_rate();
    let src_channels = decoder.channels() as usize;
    let samples: Vec<f32> = decoder.map(|s| s as f32 / i16::MAX as f32).collect();

    // Downmix to mono
    let mono = if src_channels <= 1 {
        samples
    } else {
        let mut out = Vec::with_capacity(samples.len() / src_channels);
        for frame in samples.chunks(src_channels) {
            let sum: f32 = frame.iter().sum();
            out.push(sum / src_channels as f32);
        }
        out
    };

    // Resample to 16kHz
    let target_rate = 16000u32;
    let resampled = if src_rate == target_rate {
        mono
    } else {
        resample_sinc(&mono, src_rate, target_rate)
    };

    // Create tensor: (1, T)
    let len = resampled.len();
    Tensor::from_vec(resampled, (1, len), device)
        .map_err(|e| AppError::Tts(format!("Wav tensor: {e}")))
}

/// Windowed sinc resampling (matches torchaudio.transforms.Resample defaults).
///
/// Uses sinc interpolation with Hann window, lowpass_filter_width=6, rolloff=0.99.
fn resample_sinc(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    use std::f64::consts::PI;

    if input.is_empty() || src_rate == 0 || src_rate == dst_rate {
        return input.to_vec();
    }

    let lowpass_filter_width: usize = 6;
    let rolloff: f64 = 0.99;

    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((input.len() as f64) / ratio).ceil() as usize;

    // Cutoff frequency (normalized): anti-aliasing for downsampling
    let cutoff = rolloff * (dst_rate.min(src_rate) as f64) / (src_rate.max(dst_rate) as f64);

    // For downsampling, widen the kernel to cover more source samples
    let width = if src_rate > dst_rate {
        lowpass_filter_width as f64 * ratio
    } else {
        lowpass_filter_width as f64
    };

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let center = i as f64 * ratio;
        let lo = (center - width).ceil().max(0.0) as usize;
        let hi = (center + width).floor().min((input.len() - 1) as f64) as usize;

        let mut sum = 0.0f64;
        let mut weight_sum = 0.0f64;

        for j in lo..=hi {
            let x = j as f64 - center;

            // sinc(x * cutoff) * cutoff
            let sinc = if x.abs() < 1e-12 {
                cutoff
            } else {
                (PI * x * cutoff).sin() / (PI * x)
            };

            // Hann window
            let t = x / width;
            let window = if t.abs() <= 1.0 {
                0.5 * (1.0 + (PI * t).cos())
            } else {
                0.0
            };

            let w = sinc * window;
            sum += input[j] as f64 * w;
            weight_sum += w;
        }

        // Normalize to preserve amplitude
        let sample = if weight_sum.abs() > 1e-12 {
            sum / weight_sum
        } else {
            0.0
        };
        out.push(sample as f32);
    }
    out
}

/// Compute speaker embedding from a reference WAV using WavLM + GlobalEncoder.
fn compute_speaker_embedding(
    wav_path: &Path,
    wavlm: &WavLM,
    global_encoder: &GlobalEncoder,
    device: &Device,
) -> Result<Tensor, AppError> {
    let waveform = load_and_resample_wav(wav_path, device)?;

    let ssl_features = wavlm
        .extract_global_features(&waveform)
        .map_err(|e| AppError::Tts(format!("WavLM forward: {e}")))?;

    let embedding = global_encoder
        .forward(&ssl_features)
        .map_err(|e| AppError::Tts(format!("GlobalEncoder forward: {e}")))?;

    Ok(embedding)
}
