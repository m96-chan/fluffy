use rodio::{OutputStream, OutputStreamHandle, Sink, Source};
use tokio::sync::mpsc;

use crate::error::AppError;
use crate::events::PipelineMessage;
use crate::pipeline::lip_sync;

const PLAYBACK_SAMPLE_RATE: u32 = 44_100;
const PLAYBACK_CHANNELS: u16 = 1;
/// Analyse every ~50ms of audio (2205 samples at 44.1kHz).
const LIP_SYNC_INTERVAL: usize = 2205;

/// Receives PCM chunks (f32, 44100Hz, mono) and plays them
pub struct PlaybackEngine {
    _stream: OutputStream,
    _handle: OutputStreamHandle,
    sink: Sink,
}

impl PlaybackEngine {
    pub fn new() -> Result<Self, AppError> {
        let (_stream, _handle) =
            OutputStream::try_default().map_err(|e| AppError::Audio(e.to_string()))?;
        let sink = Sink::try_new(&_handle).map_err(|e| AppError::Audio(e.to_string()))?;

        Ok(Self {
            _stream,
            _handle,
            sink,
        })
    }

    /// Queue a PCM chunk for playback with lip sync analysis during consumption.
    pub fn queue_chunk(&self, pcm: Vec<f32>, lip_tx: &mpsc::Sender<PipelineMessage>) {
        let source = LipSyncSource::new(pcm, lip_tx.clone());
        self.sink.append(source);
    }

    /// Block until all queued audio has finished playing.
    pub fn wait_until_end(&self) {
        self.sink.sleep_until_end();
    }

    /// Immediately stop all queued and playing audio.
    pub fn stop(&self) {
        self.sink.stop();
    }
}

/// A rodio Source that analyses audio for lip sync as it's consumed by the
/// sound card, giving frame-accurate mouth movement timing.
struct LipSyncSource {
    data: Vec<f32>,
    pos: usize,
    analysis_buf: Vec<f32>,
    lip_tx: mpsc::Sender<PipelineMessage>,
}

impl LipSyncSource {
    fn new(data: Vec<f32>, lip_tx: mpsc::Sender<PipelineMessage>) -> Self {
        Self {
            data,
            pos: 0,
            analysis_buf: Vec::with_capacity(LIP_SYNC_INTERVAL),
            lip_tx,
        }
    }

    fn analyze_and_send(&mut self) {
        let w = lip_sync::compute_vowel_weights(&self.analysis_buf);
        let _ = self.lip_tx.try_send(PipelineMessage::LipSync {
            aa: w.aa,
            ih: w.ih,
            ou: w.ou,
            ee: w.ee,
            oh: w.oh,
        });
        self.analysis_buf.clear();
    }
}

impl Iterator for LipSyncSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.pos >= self.data.len() {
            // Flush remaining samples
            if !self.analysis_buf.is_empty() {
                self.analyze_and_send();
            }
            return None;
        }

        let sample = self.data[self.pos];
        self.pos += 1;
        self.analysis_buf.push(sample);

        if self.analysis_buf.len() >= LIP_SYNC_INTERVAL {
            self.analyze_and_send();
        }

        Some(sample)
    }
}

impl Source for LipSyncSource {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        PLAYBACK_CHANNELS
    }

    fn sample_rate(&self) -> u32 {
        PLAYBACK_SAMPLE_RATE
    }

    fn total_duration(&self) -> Option<std::time::Duration> {
        None
    }
}
