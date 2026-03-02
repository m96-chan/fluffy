use std::sync::{Arc, Mutex};
use tokio::task;
use tracing::info;

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, model::Whisper, Config};
use tokenizers::Tokenizer;

use crate::error::AppError;
use crate::stt::download::WhisperModelFiles;

/// Candle-based Whisper inference engine.
pub struct WhisperEngine {
    /// Mutex-wrapped model — forward() needs &mut self for KV cache.
    model: Mutex<Whisper>,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
    mel_filters: Vec<f32>,
    /// Precomputed suppress_tokens bias tensor (vocab_size,).
    suppress_tokens: Tensor,
    // Special token IDs
    sot_token: u32,
    eot_token: u32,
    transcribe_token: u32,
    no_timestamps_token: u32,
}

impl WhisperEngine {
    pub fn load(files: &WhisperModelFiles, device: &Device) -> Result<Self, AppError> {
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&files.config)
                .map_err(|e| AppError::Stt(format!("Failed to read config.json: {e}")))?,
        )
        .map_err(|e| AppError::Stt(format!("Failed to parse config.json: {e}")))?;

        info!(
            "Loading Whisper model: d_model={}, encoder_layers={}, decoder_layers={}, vocab_size={}",
            config.d_model, config.encoder_layers, config.decoder_layers, config.vocab_size
        );

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&files.model], candle_core::DType::F32, device)
                .map_err(|e| AppError::Stt(format!("Failed to load safetensors: {e}")))?
        };

        let model = Whisper::load(&vb, config.clone())
            .map_err(|e| AppError::Stt(format!("Failed to build Whisper model: {e}")))?;

        let tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| AppError::Stt(format!("Failed to load tokenizer: {e}")))?;

        let mel_filters = crate::stt::download::load_mel_filters(config.num_mel_bins)?;

        // Build suppress_tokens bias tensor
        let suppress_tokens_bias: Vec<f32> = (0..config.vocab_size as u32)
            .map(|i| {
                if config.suppress_tokens.contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens =
            Tensor::new(suppress_tokens_bias.as_slice(), device)
                .map_err(|e| AppError::Stt(format!("Failed to create suppress tensor: {e}")))?;

        // Resolve special token IDs
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;

        info!(
            "Whisper engine loaded on {:?} (sot={}, eot={}, transcribe={}, no_ts={})",
            device, sot_token, eot_token, transcribe_token, no_timestamps_token
        );

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            config,
            device: device.clone(),
            mel_filters,
            suppress_tokens,
            sot_token,
            eot_token,
            transcribe_token,
            no_timestamps_token,
        })
    }
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32, AppError> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| AppError::Stt(format!("Token not found in vocabulary: {token}")))
}

/// Run Whisper STT inference on the given PCM audio (f32, 16kHz, mono).
/// Returns transcribed text.
///
/// `language`: ISO 639-1 code (e.g. "ja", "en") or "auto" for auto-detect.
pub async fn transcribe(
    engine: Arc<WhisperEngine>,
    pcm: Vec<f32>,
    language: String,
) -> Result<String, AppError> {
    task::spawn_blocking(move || run_inference(&engine, &pcm, &language))
        .await
        .map_err(|e| AppError::Stt(format!("Task join error: {e}")))?
}

fn run_inference(engine: &WhisperEngine, pcm: &[f32], language: &str) -> Result<String, AppError> {
    let map_err = |e: candle_core::Error| AppError::Stt(format!("Whisper inference: {e}"));

    info!("Whisper: processing {} samples ({:.1}s)", pcm.len(), pcm.len() as f32 / 16000.0);

    // 1. PCM → mel spectrogram
    let mel = m::audio::pcm_to_mel(&engine.config, pcm, &engine.mel_filters);
    let mel_len = mel.len();
    let n_mels = engine.config.num_mel_bins;
    let n_frames = mel_len / n_mels;
    info!("Whisper: mel spectrogram {} frames", n_frames);
    let mel = Tensor::from_vec(mel, (1, n_mels, n_frames), &engine.device)
        .map_err(map_err)?;

    // 2. Build initial token sequence: [SOT, language?, transcribe, no_timestamps]
    let mut tokens: Vec<u32> = vec![engine.sot_token];

    // Language token (for multilingual models)
    if language != "auto" {
        let lang_token_str = format!("<|{language}|>");
        match token_id(&engine.tokenizer, &lang_token_str) {
            Ok(id) => tokens.push(id),
            Err(_) => {
                tracing::warn!("Language '{language}' not found in tokenizer, using auto-detect");
            }
        }
    }

    tokens.push(engine.transcribe_token);
    tokens.push(engine.no_timestamps_token);

    // 3. Lock model for mutable access (encoder/decoder need &mut self for KV cache)
    let mut model = engine
        .model
        .lock()
        .map_err(|e| AppError::Stt(format!("Whisper model lock poisoned: {e}")))?;

    // 4. Process mel segments
    let mut seek: usize = 0;
    let mut all_text = String::new();

    while seek < n_frames {
        let segment_size = usize::min(n_frames - seek, m::N_FRAMES);
        let mel_segment = mel.narrow(2, seek, segment_size).map_err(map_err)?;

        // Encode
        let audio_features = model.encoder.forward(&mel_segment, true).map_err(map_err)?;

        // Decode
        let sample_len = engine.config.max_target_positions / 2;
        let mut segment_tokens = tokens.clone();

        for i in 0..sample_len {
            let tokens_t =
                Tensor::new(segment_tokens.as_slice(), &engine.device).map_err(map_err)?;
            let tokens_t = tokens_t.unsqueeze(0).map_err(map_err)?;

            let ys = model
                .decoder
                .forward(&tokens_t, &audio_features, i == 0)
                .map_err(map_err)?;

            let (_, seq_len, _) = ys.dims3().map_err(map_err)?;
            let ys_slice = ys.i((..1, seq_len - 1..)).map_err(map_err)?;
            let logits = model
                .decoder
                .final_linear(&ys_slice)
                .map_err(map_err)?
                .i(0)
                .map_err(map_err)?
                .i(0)
                .map_err(map_err)?;

            // Apply suppress_tokens bias
            let logits = logits
                .broadcast_add(&engine.suppress_tokens)
                .map_err(map_err)?;

            // Greedy argmax
            let logits_v: Vec<f32> = logits.to_vec1().map_err(map_err)?;
            let next_token = logits_v
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(i, _)| i as u32)
                .unwrap_or(engine.eot_token);

            if next_token == engine.eot_token
                || segment_tokens.len() > engine.config.max_target_positions
            {
                break;
            }

            segment_tokens.push(next_token);
        }

        // Decode tokens to text (skip the initial prompt tokens)
        let text_tokens: Vec<u32> = segment_tokens[tokens.len()..].to_vec();
        match engine.tokenizer.decode(&text_tokens, true) {
            Ok(text) => {
                info!("Whisper segment: {} tokens → \"{}\"", text_tokens.len(), text);
                all_text.push_str(&text);
            }
            Err(e) => {
                tracing::warn!("Whisper tokenizer decode error: {e}");
            }
        }

        // Reset KV cache for next segment
        model.reset_kv_cache();

        seek += segment_size;
    }

    let result = all_text.trim().to_string();
    info!("Whisper result: \"{}\"", result);
    Ok(result)
}
