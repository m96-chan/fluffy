//! Standalone TTS test binary.
//!
//! Usage:
//!   cargo run --bin tts_test -- "こんにちは" [--wav assets/voice/ref.wav] [--out output.pcm]
//!
//! Without --out, plays audio through the default output device.

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let (text, wav_path, out_path) = parse_args(&args);

    tracing::info!("Text: {text}");
    tracing::info!("Reference WAV: {}", wav_path.display());

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        run(text, wav_path, out_path).await;
    });
}

async fn run(text: String, wav_path: PathBuf, out_path: Option<PathBuf>) {
    use fluffy::tts::engine::TtsEngine;

    let t0 = Instant::now();
    let engine = TtsEngine::initialize(&wav_path)
        .await
        .expect("TTS engine init failed");
    tracing::info!("Engine initialized in {:.1}s", t0.elapsed().as_secs_f32());

    tracing::info!("Warmup...");
    let t_warm = Instant::now();
    engine.warmup().ok();
    tracing::info!("Warmup done in {:.1}ms", t_warm.elapsed().as_secs_f64() * 1000.0);

    let t1 = Instant::now();
    let pcm = engine
        .synthesize_blocking(&text)
        .expect("Synthesis failed");
    let dur = t1.elapsed();
    let audio_secs = pcm.len() as f32 / 44100.0;
    tracing::info!(
        "Synthesized {:.2}s audio in {:.2}s (RTF={:.3})",
        audio_secs,
        dur.as_secs_f32(),
        dur.as_secs_f32() / audio_secs,
    );

    if let Some(path) = out_path {
        // Write as 16-bit PCM WAV (44.1kHz mono)
        let peak = pcm.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let gain = if peak > 1e-6 { 0.9 / peak } else { 1.0 };
        let samples_i16: Vec<i16> = pcm
            .iter()
            .map(|&s| (s * gain * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();
        let data_bytes: Vec<u8> = samples_i16
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let data_len = data_bytes.len() as u32;
        let file_len = 36 + data_len;
        let sample_rate: u32 = 44100;
        let byte_rate: u32 = sample_rate * 2; // 16-bit mono
        let mut wav_bytes = Vec::with_capacity(44 + data_bytes.len());
        wav_bytes.extend_from_slice(b"RIFF");
        wav_bytes.extend_from_slice(&file_len.to_le_bytes());
        wav_bytes.extend_from_slice(b"WAVE");
        wav_bytes.extend_from_slice(b"fmt ");
        wav_bytes.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav_bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        wav_bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav_bytes.extend_from_slice(&sample_rate.to_le_bytes());
        wav_bytes.extend_from_slice(&byte_rate.to_le_bytes());
        wav_bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav_bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav_bytes.extend_from_slice(b"data");
        wav_bytes.extend_from_slice(&data_len.to_le_bytes());
        wav_bytes.extend_from_slice(&data_bytes);
        std::fs::write(&path, &wav_bytes).expect("Failed to write WAV");
        tracing::info!("Wrote {} bytes to {}", wav_bytes.len(), path.display());
    } else {
        // Play through default audio device
        tracing::info!("Playing audio...");
        play_pcm(&pcm);
    }
}

fn play_pcm(pcm: &[f32]) {
    use rodio::{OutputStream, Sink, Source};

    let (_stream, handle) = OutputStream::try_default().expect("No audio output device");
    let sink = Sink::try_new(&handle).expect("Sink creation failed");

    let source = rodio::buffer::SamplesBuffer::new(1, 44100, pcm.to_vec());
    sink.append(source);
    sink.sleep_until_end();
}

fn parse_args(args: &[String]) -> (String, PathBuf, Option<PathBuf>) {
    let mut text = None;
    let mut wav = PathBuf::from("assets/voice/このボイス.wav");
    let mut out = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--wav" => {
                i += 1;
                wav = PathBuf::from(&args[i]);
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(&args[i]));
            }
            "--help" | "-h" => {
                eprintln!("Usage: tts_test <TEXT> [--wav <ref.wav>] [--out <output.pcm>]");
                std::process::exit(0);
            }
            _ => {
                if text.is_none() {
                    text = Some(args[i].clone());
                }
            }
        }
        i += 1;
    }

    let text = text.unwrap_or_else(|| {
        eprintln!("Usage: tts_test <TEXT> [--wav <ref.wav>] [--out <output.pcm>]");
        std::process::exit(1);
    });

    (text, wav, out)
}
