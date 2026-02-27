use std::path::Path;
use std::sync::Arc;
use tokio::task;
use tracing::info;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::error::AppError;

/// Run Whisper STT inference on the given PCM audio (f32, 16kHz, mono).
/// Returns transcribed text.
pub async fn transcribe(
    ctx: Arc<WhisperContext>,
    pcm: Vec<f32>,
) -> Result<String, AppError> {
    // whisper-rs is synchronous/CPU-bound, run in blocking thread pool
    // We move the Arc<WhisperContext> into spawn_blocking so we don't hold
    // the outer Mutex across an await point
    let result = task::spawn_blocking(move || {
        run_inference_with_pcm(&ctx, &pcm)
    })
    .await
    .map_err(|e| AppError::Stt(format!("Task join error: {}", e)))?;

    result
}

fn run_inference_with_pcm(ctx: &WhisperContext, pcm: &[f32]) -> Result<String, AppError> {
    let mut state = ctx
        .create_state()
        .map_err(|e| AppError::Stt(format!("Failed to create whisper state: {}", e)))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_translate(false);
    params.set_no_context(true);

    state
        .full(params, pcm)
        .map_err(|e| AppError::Stt(format!("Whisper inference failed: {}", e)))?;

    let num_segments = state
        .full_n_segments()
        .map_err(|e| AppError::Stt(format!("Failed to get segments: {}", e)))?;

    let mut text = String::new();
    for i in 0..num_segments {
        let segment = state
            .full_get_segment_text(i)
            .map_err(|e| AppError::Stt(format!("Failed to get segment {}: {}", i, e)))?;
        text.push_str(&segment);
    }

    Ok(text.trim().to_string())
}

pub fn load_whisper_context(model_path: &Path) -> Result<Arc<WhisperContext>, AppError> {
    info!("Loading whisper model from {:?}", model_path);
    let ctx = WhisperContext::new_with_params(
        model_path
            .to_str()
            .ok_or_else(|| AppError::Stt("Invalid model path".to_string()))?,
        WhisperContextParameters::default(),
    )
    .map_err(|e| AppError::Stt(format!("Failed to load whisper model: {}", e)))?;

    info!("Whisper model loaded successfully");
    Ok(Arc::new(ctx))
}
