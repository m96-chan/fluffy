/// Voice Activity Detector using RMS energy threshold
///
/// State machine:
///   SILENT → SPEECH: RMS > threshold
///   SPEECH → SILENT: consecutive silence frames >= hold_frames → emit utterance
pub struct VadState {
    threshold: f32,
    silence_hold_frames: usize,
    pre_roll_samples: usize,

    state: VadInternalState,
    silence_counter: usize,
    speech_buffer: Vec<f32>,
    pre_roll_ring: Vec<Vec<f32>>,
}

#[derive(Debug, PartialEq, Clone)]
enum VadInternalState {
    Silent,
    Speech,
}

impl VadState {
    pub fn new(threshold: f32, silence_hold_frames: usize, pre_roll_samples: usize) -> Self {
        Self {
            threshold,
            silence_hold_frames,
            pre_roll_samples,
            state: VadInternalState::Silent,
            silence_counter: 0,
            speech_buffer: Vec::new(),
            pre_roll_ring: Vec::new(),
        }
    }

    /// Process a frame of audio. Returns Some(utterance_pcm) when an utterance ends.
    pub fn process_frame(&mut self, frame: &[f32]) -> Option<Vec<f32>> {
        let rms = compute_rms(frame);

        match self.state {
            VadInternalState::Silent => {
                // Keep a rolling pre-roll buffer (limited to pre_roll_samples)
                self.pre_roll_ring.push(frame.to_vec());
                while self.pre_roll_ring.iter().map(|f| f.len()).sum::<usize>()
                    > self.pre_roll_samples + frame.len()
                {
                    if self.pre_roll_ring.is_empty() {
                        break;
                    }
                    self.pre_roll_ring.remove(0);
                }

                if rms > self.threshold {
                    // Transition to speech: prepend pre-roll
                    self.state = VadInternalState::Speech;
                    self.silence_counter = 0;

                    // Add pre-roll samples to speech buffer
                    for pre_frame in &self.pre_roll_ring {
                        self.speech_buffer.extend_from_slice(pre_frame);
                    }
                    self.pre_roll_ring.clear();
                    self.speech_buffer.extend_from_slice(frame);
                }

                None
            }
            VadInternalState::Speech => {
                self.speech_buffer.extend_from_slice(frame);

                if rms <= self.threshold {
                    self.silence_counter += 1;
                    if self.silence_counter >= self.silence_hold_frames {
                        // Utterance ended — emit
                        self.state = VadInternalState::Silent;
                        self.silence_counter = 0;
                        let utterance = std::mem::take(&mut self.speech_buffer);
                        self.pre_roll_ring.clear();
                        return Some(utterance);
                    }
                } else {
                    self.silence_counter = 0;
                }

                None
            }
        }
    }

    /// Flush any remaining speech (called on stop_listening)
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        if self.state == VadInternalState::Speech && !self.speech_buffer.is_empty() {
            self.state = VadInternalState::Silent;
            self.silence_counter = 0;
            let utterance = std::mem::take(&mut self.speech_buffer);
            Some(utterance)
        } else {
            None
        }
    }
}

pub fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_silence(n: usize) -> Vec<f32> {
        vec![0.0_f32; n]
    }

    fn make_speech(n: usize) -> Vec<f32> {
        // Simulate speech above threshold (0.02 default)
        vec![0.1_f32; n]
    }

    const FRAME: usize = 512;
    const THRESHOLD: f32 = 0.02;
    const HOLD_FRAMES: usize = 5;
    const PRE_ROLL: usize = 8000;

    #[test]
    fn silent_frames_produce_no_utterance() {
        let mut vad = VadState::new(THRESHOLD, HOLD_FRAMES, PRE_ROLL);
        for _ in 0..10 {
            let result = vad.process_frame(&make_silence(FRAME));
            assert!(result.is_none(), "Silent frames should not produce utterance");
        }
    }

    #[test]
    fn speech_then_silence_emits_utterance() {
        let mut vad = VadState::new(THRESHOLD, HOLD_FRAMES, PRE_ROLL);

        // Speak for 5 frames
        for _ in 0..5 {
            let result = vad.process_frame(&make_speech(FRAME));
            assert!(result.is_none());
        }

        // Silence for HOLD_FRAMES - triggers utterance on last silence frame
        let mut utterance = None;
        for _ in 0..HOLD_FRAMES {
            utterance = vad.process_frame(&make_silence(FRAME));
        }

        assert!(utterance.is_some(), "Should emit utterance after silence hold");
        let utt = utterance.unwrap();
        assert!(!utt.is_empty());
    }

    #[test]
    fn pre_roll_is_included_in_utterance() {
        let mut vad = VadState::new(THRESHOLD, HOLD_FRAMES, PRE_ROLL);
        let frame = make_silence(FRAME);

        // Add some silence frames (these become pre-roll)
        for _ in 0..3 {
            vad.process_frame(&frame);
        }

        // Speech starts
        vad.process_frame(&make_speech(FRAME));

        // Collect speech buffer length before silence
        // Silence to trigger end
        let mut utterance = None;
        for _ in 0..HOLD_FRAMES {
            utterance = vad.process_frame(&make_silence(FRAME));
        }

        let utt = utterance.unwrap();
        // Should include pre-roll (3 silence frames) + 1 speech frame + hold silence frames
        assert!(utt.len() > FRAME, "Utterance should include pre-roll samples");
    }

    #[test]
    fn configurable_threshold_respected() {
        let high_threshold = 0.5;
        let mut vad = VadState::new(high_threshold, HOLD_FRAMES, PRE_ROLL);

        // Low-level speech (0.1 RMS) should not trigger with threshold 0.5
        for _ in 0..5 {
            let result = vad.process_frame(&make_speech(FRAME));
            assert!(result.is_none());
        }
        // Still in SILENT state
        for _ in 0..HOLD_FRAMES {
            let result = vad.process_frame(&make_silence(FRAME));
            assert!(result.is_none());
        }
    }

    #[test]
    fn flush_returns_remaining_speech() {
        let mut vad = VadState::new(THRESHOLD, HOLD_FRAMES, PRE_ROLL);

        // Start speaking
        for _ in 0..3 {
            vad.process_frame(&make_speech(FRAME));
        }

        // Flush without enough silence to auto-trigger
        let result = vad.flush();
        assert!(result.is_some(), "Flush should return remaining speech");
    }

    #[test]
    fn rms_of_silence_is_zero() {
        let silence = make_silence(512);
        assert_eq!(compute_rms(&silence), 0.0);
    }

    #[test]
    fn rms_of_unit_signal() {
        let ones: Vec<f32> = vec![1.0; 512];
        assert!((compute_rms(&ones) - 1.0).abs() < 1e-5);
    }
}
