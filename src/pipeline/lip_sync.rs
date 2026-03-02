use std::f32::consts::PI;

use crate::audio::vad::compute_rms;

/// Vowel weights for driving 5 morph targets (あいうえお).
#[derive(Debug, Clone, Copy)]
pub struct VowelWeights {
    pub aa: f32,
    pub ih: f32,
    pub ou: f32,
    pub ee: f32,
    pub oh: f32,
}

impl VowelWeights {
    pub fn silence() -> Self {
        Self { aa: 0.0, ih: 0.0, ou: 0.0, ee: 0.0, oh: 0.0 }
    }
}

/// Analyse a PCM chunk (44.1kHz mono) and return per-vowel weights [0,1].
///
/// Uses Goertzel algorithm to measure energy at Japanese vowel formant
/// frequencies (F1 + F2), then scores each vowel by formant proximity.
pub fn compute_vowel_weights(samples: &[f32]) -> VowelWeights {
    let rms = compute_rms(samples);
    let amplitude = (rms * 5.0).min(1.0);

    if amplitude < 0.02 {
        return VowelWeights::silence();
    }

    const SR: f32 = 44_100.0;

    // F1 formant region (300-800 Hz)
    let f1_low  = goertzel_energy(samples, SR, 300.0);  // い, う
    let f1_mid  = goertzel_energy(samples, SR, 500.0);  // え, お
    let f1_high = goertzel_energy(samples, SR, 800.0);  // あ

    // F2 formant region (900-2500 Hz)
    let f2_low   = goertzel_energy(samples, SR, 900.0);  // お
    let f2_mid   = goertzel_energy(samples, SR, 1200.0); // あ
    let f2_high  = goertzel_energy(samples, SR, 1800.0); // え
    let f2_vhigh = goertzel_energy(samples, SR, 2300.0); // い

    // Score each vowel by (F1, F2) formant match:
    //   あ: F1≈800  F2≈1200
    //   い: F1≈300  F2≈2300
    //   う: F1≈300  F2≈1500
    //   え: F1≈500  F2≈1800
    //   お: F1≈500  F2≈900
    let aa_score = f1_high * f2_mid;
    let ih_score = f1_low  * f2_vhigh;
    let ou_score = f1_low  * (f2_mid + f2_high) * 0.5;
    let ee_score = f1_mid  * f2_high;
    let oh_score = f1_mid  * f2_low;

    let total = (aa_score + ih_score + ou_score + ee_score + oh_score).max(1e-8);

    VowelWeights {
        aa: (aa_score / total * amplitude).min(1.0),
        ih: (ih_score / total * amplitude).min(1.0),
        ou: (ou_score / total * amplitude).min(1.0),
        ee: (ee_score / total * amplitude).min(1.0),
        oh: (oh_score / total * amplitude).min(1.0),
    }
}

/// Goertzel algorithm — O(N) energy at a single frequency bin.
fn goertzel_energy(samples: &[f32], sample_rate: f32, target_hz: f32) -> f32 {
    let n = samples.len() as f32;
    let k = (target_hz * n / sample_rate).round();
    let omega = 2.0 * PI * k / n;
    let coeff = 2.0 * omega.cos();

    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;

    for &sample in samples {
        let s0 = sample + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }

    let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    power.max(0.0).sqrt() / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_gives_zero_weights() {
        let silence = vec![0.0f32; 2048];
        let w = compute_vowel_weights(&silence);
        assert_eq!(w.aa, 0.0);
        assert_eq!(w.ih, 0.0);
        assert_eq!(w.ou, 0.0);
        assert_eq!(w.ee, 0.0);
        assert_eq!(w.oh, 0.0);
    }

    #[test]
    fn loud_signal_produces_nonzero_weights() {
        // 800 Hz sine (F1 of あ) at 44100 Hz
        let n = 2048;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 800.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let w = compute_vowel_weights(&samples);
        // Should have some non-zero weight (F1=800 → あ dominant)
        assert!(w.aa > 0.0 || w.ih > 0.0 || w.ee > 0.0 || w.oh > 0.0 || w.ou > 0.0);
    }

    #[test]
    fn goertzel_detects_pure_tone() {
        let n = 1024;
        let hz = 440.0;
        let sr = 44100.0;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * hz * i as f32 / sr).sin())
            .collect();
        let energy_440 = goertzel_energy(&samples, sr, hz);
        let energy_1000 = goertzel_energy(&samples, sr, 1000.0);
        assert!(energy_440 > energy_1000 * 5.0, "440Hz tone should dominate at 440Hz bin");
    }
}
