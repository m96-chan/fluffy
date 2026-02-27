/// Procedural animation math: breathing, head sway.
/// Pure functions — no Bevy dependency — fully testable.

use std::f32::consts::TAU;

/// Parameters for the breathing animation.
#[derive(Debug, Clone)]
pub struct BreathingParams {
    /// Oscillation frequency in Hz (full inhale→exhale cycle).
    pub frequency_hz: f32,
    /// Maximum rotation angle in radians applied to the chest bone (X axis).
    pub amplitude_rad: f32,
}

impl Default for BreathingParams {
    fn default() -> Self {
        Self {
            frequency_hz: 0.25, // one breath every 4 seconds
            amplitude_rad: 0.015,
        }
    }
}

/// Parameters for the head micro-sway animation.
#[derive(Debug, Clone)]
pub struct HeadSwayParams {
    /// Primary sway frequency (Hz).
    pub freq_primary_hz: f32,
    /// Secondary drift frequency (Hz) — gives organic feel.
    pub freq_secondary_hz: f32,
    /// Maximum X-axis (nod) amplitude in radians.
    pub amplitude_x_rad: f32,
    /// Maximum Y-axis (turn) amplitude in radians.
    pub amplitude_y_rad: f32,
}

impl Default for HeadSwayParams {
    fn default() -> Self {
        Self {
            freq_primary_hz: 0.13,
            freq_secondary_hz: 0.07,
            amplitude_x_rad: 0.008,
            amplitude_y_rad: 0.012,
        }
    }
}

/// Compute chest bone X-rotation offset for breathing at time `t` (seconds).
/// Returns radians. Scales amplitude by `phase_scale` (0.0 = silent, 1.0 = full).
pub fn breathing_rotation(t: f32, params: &BreathingParams, phase_scale: f32) -> f32 {
    let angle = (t * TAU * params.frequency_hz).sin();
    angle * params.amplitude_rad * phase_scale
}

/// Compute head rotation offsets (x_rad, y_rad) at time `t` (seconds).
pub fn head_sway(t: f32, params: &HeadSwayParams) -> (f32, f32) {
    let x = (t * TAU * params.freq_primary_hz).sin() * params.amplitude_x_rad;
    let y = (t * TAU * params.freq_secondary_hz).sin() * params.amplitude_y_rad
        + (t * TAU * params.freq_primary_hz * 1.3).cos() * params.amplitude_y_rad * 0.4;
    (x, y)
}

/// Animation scale factor for each mascot phase.
/// Breathing is reduced during Speaking (mouth is moving, subtle override).
pub fn breathing_scale_for_phase(phase: &crate::events::MascotPhase) -> f32 {
    use crate::events::MascotPhase::*;
    match phase {
        Idle | Listening => 1.0,
        Processing | Thinking => 0.6,
        Speaking => 0.3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::MascotPhase;

    // ── breathing ────────────────────────────────────────────────────────────

    #[test]
    fn breathing_at_t0_is_zero() {
        let rot = breathing_rotation(0.0, &BreathingParams::default(), 1.0);
        assert!(rot.abs() < 1e-6, "sin(0) should be 0, got {}", rot);
    }

    #[test]
    fn breathing_amplitude_bounded() {
        let params = BreathingParams::default();
        for i in 0..1000 {
            let t = i as f32 * 0.01;
            let rot = breathing_rotation(t, &params, 1.0);
            assert!(
                rot.abs() <= params.amplitude_rad + 1e-6,
                "amplitude exceeded at t={}: {}",
                t,
                rot
            );
        }
    }

    #[test]
    fn breathing_scales_with_phase() {
        let t = 1.0;
        let params = BreathingParams::default();
        let full = breathing_rotation(t, &params, 1.0);
        let half = breathing_rotation(t, &params, 0.5);
        assert!((half - full * 0.5).abs() < 1e-6);
    }

    #[test]
    fn breathing_zero_at_zero_scale() {
        let rot = breathing_rotation(1.5, &BreathingParams::default(), 0.0);
        assert!(rot.abs() < 1e-6);
    }

    #[test]
    fn breathing_completes_cycle() {
        let params = BreathingParams::default();
        let period = 1.0 / params.frequency_hz;
        let start = breathing_rotation(0.0, &params, 1.0);
        let end = breathing_rotation(period, &params, 1.0);
        assert!((start - end).abs() < 1e-5, "cycle should repeat: {} vs {}", start, end);
    }

    // ── head sway ────────────────────────────────────────────────────────────

    #[test]
    fn head_sway_amplitude_bounded() {
        let params = HeadSwayParams::default();
        for i in 0..1000 {
            let t = i as f32 * 0.05;
            let (x, y) = head_sway(t, &params);
            assert!(
                x.abs() <= params.amplitude_x_rad * 1.01,
                "X exceeded at t={}: {}",
                t,
                x
            );
            // Y combines two sines, max is amplitude + 40% of amplitude
            let y_max = params.amplitude_y_rad * 1.41;
            assert!(y.abs() <= y_max, "Y exceeded at t={}: {}", t, y);
        }
    }

    #[test]
    fn head_sway_returns_different_x_y() {
        let (x, y) = head_sway(1.0, &HeadSwayParams::default());
        // Frequencies are different so x ≠ y (in general)
        // Just verify they're not both exactly 0 at t=1
        assert!(x != 0.0 || y != 0.0);
    }

    // ── phase scale ──────────────────────────────────────────────────────────

    #[test]
    fn idle_has_full_breathing() {
        assert_eq!(breathing_scale_for_phase(&MascotPhase::Idle), 1.0);
    }

    #[test]
    fn speaking_has_reduced_breathing() {
        let scale = breathing_scale_for_phase(&MascotPhase::Speaking);
        assert!(scale < 1.0, "Speaking should reduce breathing");
    }

    #[test]
    fn all_phases_have_positive_scale() {
        use MascotPhase::*;
        for phase in [Idle, Listening, Processing, Thinking, Speaking] {
            assert!(breathing_scale_for_phase(&phase) > 0.0);
        }
    }
}
