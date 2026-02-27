use rodio::Source;
use std::io::BufReader;
use std::process::Command;
use tokio::sync::mpsc;

use crate::error::AppError;
use crate::state::AppConfig;

const PCM_SAMPLE_RATE: u32 = 44_100;

/// Synthesize speech with local clone TTS and return f32 PCM chunks (44100Hz mono).
pub async fn synthesize(
    config: &AppConfig,
    text: &str,
) -> Result<mpsc::Receiver<Vec<f32>>, AppError> {
    let (tx, rx) = mpsc::channel::<Vec<f32>>(1);
    let text = text.to_string();
    let tts_bin = config.tts_clone_bin.clone();
    let voice = config.tts_clone_voice_wav.clone();
    let model = config.tts_clone_model.clone();

    if !voice.exists() {
        return Err(AppError::Tts(format!(
            "Clone voice wav not found: {}",
            voice.display()
        )));
    }

    tokio::task::spawn_blocking(move || -> Result<(), AppError> {
        let out_wav = std::env::temp_dir().join(format!(
            "fluffy_clone_tts_{}_{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));

        let output = Command::new(&tts_bin)
            .env("COQUI_TOS_AGREED", "1")
            .arg("--text")
            .arg(&text)
            .arg("--model_name")
            .arg(&model)
            .arg("--speaker_wav")
            .arg(&voice)
            .arg("--out_path")
            .arg(&out_wav)
            .output()
            .map_err(|e| AppError::Tts(format!("Failed to execute local TTS binary '{}': {}", tts_bin, e)))?;

        if !output.status.success() {
            return Err(AppError::Tts(format!(
                "Local `tts` CLI failed (status={}): {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let file = std::fs::File::open(&out_wav)
            .map_err(|e| AppError::Tts(format!("Failed to open generated wav: {}", e)))?;
        let decoder = rodio::Decoder::new(BufReader::new(file))
            .map_err(|e| AppError::Tts(format!("Failed to decode generated wav: {}", e)))?;
        let src_rate = decoder.sample_rate();
        let src_channels = decoder.channels() as usize;
        let src_samples: Vec<f32> = decoder
            .map(|s| s as f32 / i16::MAX as f32)
            .collect();
        let mono = downmix_to_mono(&src_samples, src_channels);
        let resampled = resample_linear(&mono, src_rate, PCM_SAMPLE_RATE);

        let _ = std::fs::remove_file(&out_wav);

        if tx.blocking_send(resampled).is_err() {
            return Ok(());
        }
        Ok(())
    })
    .await
    .map_err(|e| AppError::Tts(format!("Local clone TTS task join error: {}", e)))??;

    Ok(rx)
}

fn downmix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let mut out = Vec::with_capacity(samples.len() / channels);
    for frame in samples.chunks(channels) {
        let sum: f32 = frame.iter().copied().sum();
        out.push(sum / channels as f32);
    }
    out
}

fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == 0 || src_rate == dst_rate {
        return input.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((input.len() as f64) / ratio).max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f64) * ratio;
        let i0 = src_pos.floor() as usize;
        let i1 = (i0 + 1).min(input.len() - 1);
        let t = (src_pos - i0 as f64) as f32;
        let s = input[i0] * (1.0 - t) + input[i1] * t;
        out.push(s);
    }
    out
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
