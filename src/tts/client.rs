use reqwest::Client;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::error::AppError;

const PCM_SAMPLE_RATE: u32 = 44_100;
const PCM_CHANNELS: u16 = 1;

/// Send text to Fish Speech TTS server and receive PCM audio chunks.
/// Returns a channel that yields f32 PCM samples (44100Hz mono).
pub async fn synthesize(
    tts_url: &str,
    text: &str,
) -> Result<mpsc::Receiver<Vec<f32>>, AppError> {
    let (tx, rx) = mpsc::channel::<Vec<f32>>(8);
    let url = format!("{}/v1/tts", tts_url);
    let text = text.to_string();
    let client = Client::new();

    tokio::spawn(async move {
        if let Err(e) = stream_tts(&client, &url, &text, tx).await {
            error!("TTS error: {}", e);
        }
    });

    Ok(rx)
}

async fn stream_tts(
    client: &Client,
    url: &str,
    text: &str,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<(), AppError> {
    let body = serde_json::json!({
        "text": text,
        "format": "pcm",
        "streaming": true,
        "sample_rate": PCM_SAMPLE_RATE,
    });

    let response = client
        .post(url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| AppError::Tts(format!("TTS request failed: {}", e)))?;

    if !response.status().is_success() {
        // Try non-streaming fallback
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            warn!("Streaming TTS endpoint not found, trying non-streaming fallback");
            return synthesize_non_streaming(client, url, text, tx).await;
        }

        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(AppError::Tts(format!("TTS API error {}: {}", status, body)));
    }

    // Read chunked raw PCM (16-bit LE)
    use tokio_stream::StreamExt;
    let mut stream = response.bytes_stream();
    let mut leftover: Vec<u8> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| AppError::Tts(format!("TTS stream error: {}", e)))?;

        leftover.extend_from_slice(&chunk);

        // Convert pairs of bytes to f32 samples (16-bit LE)
        let complete_samples = (leftover.len() / 2) * 2;
        let samples: Vec<f32> = leftover[..complete_samples]
            .chunks_exact(2)
            .map(|b| {
                let sample = i16::from_le_bytes([b[0], b[1]]);
                sample as f32 / 32768.0
            })
            .collect();

        leftover = leftover[complete_samples..].to_vec();

        if !samples.is_empty() {
            if tx.send(samples).await.is_err() {
                break; // Receiver dropped
            }
        }
    }

    Ok(())
}

async fn synthesize_non_streaming(
    client: &Client,
    url: &str,
    text: &str,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<(), AppError> {
    // Non-streaming fallback: POST without streaming flag, get full audio
    let base_url = url.trim_end_matches("/v1/tts");
    let fallback_url = format!("{}/v1/tts", base_url);

    let body = serde_json::json!({
        "text": text,
        "format": "pcm",
    });

    let response = client
        .post(&fallback_url)
        .json(&body)
        .send()
        .await
        .map_err(|e| AppError::Tts(format!("TTS fallback request failed: {}", e)))?;

    let bytes = response
        .bytes()
        .await
        .map_err(|e| AppError::Tts(format!("TTS fallback read failed: {}", e)))?;

    let samples: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|b| {
            let sample = i16::from_le_bytes([b[0], b[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let _ = tx.send(samples).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i16_to_f32_conversion_correct() {
        // 0 → 0.0
        let zero: Vec<u8> = vec![0x00, 0x00];
        let sample = i16::from_le_bytes([zero[0], zero[1]]) as f32 / 32768.0;
        assert_eq!(sample, 0.0);

        // i16::MAX → ~1.0
        let max_bytes = i16::MAX.to_le_bytes();
        let sample = i16::from_le_bytes(max_bytes) as f32 / 32768.0;
        assert!((sample - 1.0).abs() < 0.001);

        // i16::MIN → -1.0
        let min_bytes = i16::MIN.to_le_bytes();
        let sample = i16::from_le_bytes(min_bytes) as f32 / 32768.0;
        assert_eq!(sample, -1.0);
    }
}
