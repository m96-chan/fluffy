use rodio::{OutputStream, OutputStreamHandle, Sink, Source};
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::error::AppError;

const PLAYBACK_SAMPLE_RATE: u32 = 44_100;
const PLAYBACK_CHANNELS: u16 = 1;

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

    /// Queue a PCM chunk for playback
    pub fn queue_chunk(&self, pcm: Vec<f32>) {
        let source = PcmSource::new(pcm, PLAYBACK_SAMPLE_RATE, PLAYBACK_CHANNELS);
        self.sink.append(source);
    }

    pub fn stop(&self) {
        self.sink.stop();
    }

    pub fn is_empty(&self) -> bool {
        self.sink.empty()
    }
}

/// A rodio Source backed by a Vec<f32>
struct PcmSource {
    data: std::vec::IntoIter<f32>,
    sample_rate: u32,
    channels: u16,
}

impl PcmSource {
    fn new(data: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            data: data.into_iter(),
            sample_rate,
            channels,
        }
    }
}

impl Iterator for PcmSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        self.data.next()
    }
}

impl Source for PcmSource {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<std::time::Duration> {
        None
    }
}
