use crate::audio::vad::compute_rms;

/// Compute lip sync amplitude from a PCM chunk, normalized to [0, 1].
pub fn compute_lip_sync_amplitude(samples: &[f32]) -> f32 {
    let rms = compute_rms(samples);
    // Boost amplitude for visibility (speech is typically at low levels)
    (rms * 5.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_gives_zero_amplitude() {
        let silence = vec![0.0f32; 512];
        assert_eq!(compute_lip_sync_amplitude(&silence), 0.0);
    }

    #[test]
    fn loud_signal_clamped_to_one() {
        let loud = vec![1.0f32; 512];
        assert_eq!(compute_lip_sync_amplitude(&loud), 1.0);
    }
}
