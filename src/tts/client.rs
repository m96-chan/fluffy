use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::error;

use crate::error::AppError;
use super::engine::TtsEngine;

/// Synthesize speech using the local TTS engine.
///
/// Runs the GPU inference on a blocking thread to avoid blocking the async runtime.
/// Returns a channel of PCM f32 chunks (44100Hz mono).
pub async fn synthesize(
    engine: &Arc<Mutex<TtsEngine>>,
    text: &str,
) -> Result<mpsc::Receiver<Vec<f32>>, AppError> {
    let engine = engine.clone();
    let text = text.to_string();

    let (tx, rx) = mpsc::channel::<Vec<f32>>(1);

    tokio::task::spawn_blocking(move || {
        let engine = engine.blocking_lock();
        match engine.synthesize_blocking(&text) {
            Ok(pcm) => {
                let _ = tx.blocking_send(pcm);
            }
            Err(e) => {
                error!("TTS synthesis error: {}", e);
            }
        }
    });

    Ok(rx)
}
