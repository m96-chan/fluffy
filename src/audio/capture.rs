use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, SupportedStreamConfig};

/// Newtype wrapper to make cpal::Stream sendable across threads.
///
/// # Safety
/// cpal::Stream on Linux (ALSA/PipeWire) is safe to send between threads in
/// practice; the `!Send` bound is a conservative cross-platform marker.
/// The stream object is only ever dropped on the dedicated keeper thread.
struct SendableStream(Stream);
unsafe impl Send for SendableStream {}
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::error::AppError;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub const FRAME_SIZE: usize = 512; // ~32ms at 16kHz

/// Start capturing audio from the specified device (or default).
/// Returns a receiver that yields chunks of f32 mono PCM at 16kHz.
pub async fn start_capture(
    device_name: Option<String>,
) -> Result<mpsc::Receiver<Vec<f32>>, AppError> {
    let (tx, rx) = mpsc::channel::<Vec<f32>>(256);

    let host = cpal::default_host();
    let device = if let Some(name) = device_name {
        host.input_devices()
            .map_err(|e| AppError::Audio(e.to_string()))?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| AppError::Audio(format!("Audio device '{}' not found", name)))?
    } else {
        host.default_input_device()
            .ok_or_else(|| AppError::Audio("No default input device".to_string()))?
    };

    info!("Using audio device: {}", device.name().unwrap_or_default());

    // Try to get 16kHz config, fall back to device default
    let config = get_input_config(&device)?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    info!("Audio config: {}Hz, {} channels", sample_rate, channels);

    let needs_resample = sample_rate != TARGET_SAMPLE_RATE;
    let resample_ratio = TARGET_SAMPLE_RATE as f64 / sample_rate as f64;

    let tx_clone = tx.clone();
    let mut resample_buffer: Vec<f32> = Vec::new();
    let mut input_buf = Vec::new();

    let stream = match config.sample_format() {
        SampleFormat::F32 => {
            let tx = tx_clone;
            device
                .build_input_stream(
                    &config.into(),
                    move |data: &[f32], _| {
                        handle_input(
                            data,
                            channels,
                            needs_resample,
                            resample_ratio,
                            &mut input_buf,
                            &mut resample_buffer,
                            &tx,
                        );
                    },
                    |e| error!("Audio stream error: {}", e),
                    None,
                )
                .map_err(|e| AppError::Audio(e.to_string()))?
        }
        SampleFormat::I16 => {
            let tx = tx_clone;
            device
                .build_input_stream(
                    &config.into(),
                    move |data: &[i16], _| {
                        let f32_data: Vec<f32> =
                            data.iter().map(|&s| s as f32 / 32768.0).collect();
                        handle_input(
                            &f32_data,
                            channels,
                            needs_resample,
                            resample_ratio,
                            &mut input_buf,
                            &mut resample_buffer,
                            &tx,
                        );
                    },
                    |e| error!("Audio stream error: {}", e),
                    None,
                )
                .map_err(|e| AppError::Audio(e.to_string()))?
        }
        SampleFormat::U8 => {
            let tx = tx_clone;
            device
                .build_input_stream(
                    &config.into(),
                    move |data: &[u8], _| {
                        let f32_data: Vec<f32> =
                            data.iter().map(|&s| (s as f32 - 128.0) / 128.0).collect();
                        handle_input(
                            &f32_data,
                            channels,
                            needs_resample,
                            resample_ratio,
                            &mut input_buf,
                            &mut resample_buffer,
                            &tx,
                        );
                    },
                    |e| error!("Audio stream error: {}", e),
                    None,
                )
                .map_err(|e| AppError::Audio(e.to_string()))?
        }
        fmt => {
            return Err(AppError::Audio(format!(
                "Unsupported sample format: {:?}",
                fmt
            )))
        }
    };

    stream.play().map_err(|e| AppError::Audio(e.to_string()))?;

    // Keep stream alive on a dedicated thread.
    // Wrap in SendableStream to cross the thread boundary safely (see above).
    let sendable = SendableStream(stream);
    std::thread::spawn(move || {
        let _stream = sendable;
        loop {
            std::thread::sleep(std::time::Duration::from_millis(100));
            if tx.is_closed() {
                break;
            }
        }
    });

    Ok(rx)
}

fn get_input_config(device: &cpal::Device) -> Result<SupportedStreamConfig, AppError> {
    // First try exact 16kHz
    let desired = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(TARGET_SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Fixed(FRAME_SIZE as u32),
    };

    // Check if device supports it
    if let Ok(configs) = device.supported_input_configs() {
        for c in configs {
            if c.min_sample_rate().0 <= TARGET_SAMPLE_RATE
                && c.max_sample_rate().0 >= TARGET_SAMPLE_RATE
            {
                return Ok(c.with_sample_rate(cpal::SampleRate(TARGET_SAMPLE_RATE)));
            }
        }
    }

    // Fall back to default config (will resample)
    warn!("Device does not natively support 16kHz; will resample");
    device
        .default_input_config()
        .map_err(|e| AppError::Audio(e.to_string()))
}

fn handle_input(
    data: &[f32],
    channels: usize,
    needs_resample: bool,
    resample_ratio: f64,
    input_buf: &mut Vec<f32>,
    resample_buffer: &mut Vec<f32>,
    tx: &mpsc::Sender<Vec<f32>>,
) {
    // Convert to mono by averaging channels
    let mono: Vec<f32> = data
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();

    if needs_resample {
        // Simple linear interpolation resampler
        let resampled = linear_resample(&mono, resample_ratio);
        resample_buffer.extend_from_slice(&resampled);

        while resample_buffer.len() >= FRAME_SIZE {
            let frame: Vec<f32> = resample_buffer.drain(..FRAME_SIZE).collect();
            let _ = tx.try_send(frame);
        }
    } else {
        input_buf.extend_from_slice(&mono);
        while input_buf.len() >= FRAME_SIZE {
            let frame: Vec<f32> = input_buf.drain(..FRAME_SIZE).collect();
            let _ = tx.try_send(frame);
        }
    }
}

/// Simple linear interpolation resampler
fn linear_resample(input: &[f32], ratio: f64) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let out_len = (input.len() as f64 * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;

        let s0 = input[src_idx.min(input.len() - 1)];
        let s1 = if src_idx + 1 < input.len() {
            input[src_idx + 1]
        } else {
            s0
        };

        out.push(s0 + (s1 - s0) * frac as f32);
    }
    out
}

pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices()
        .map(|devs| {
            devs.filter_map(|d| d.name().ok()).collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_resample_doubles_length() {
        let input: Vec<f32> = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let output = linear_resample(&input, 2.0);
        assert_eq!(output.len(), 10);
    }

    #[test]
    fn linear_resample_identity() {
        let input: Vec<f32> = vec![0.1, 0.5, 0.9];
        let output = linear_resample(&input, 1.0);
        assert_eq!(output.len(), 3);
        assert!((output[0] - 0.1).abs() < 1e-5);
    }
}
