/// Pure math for perch target selection and gravity simulation.
/// No Bevy dependency — fully unit-testable.
use crate::perch::tracker::ForeignWindow;

/// Mascot bounding box in screen coordinates.
#[derive(Debug, Clone)]
pub struct MascotRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl MascotRect {
    pub fn bottom(&self) -> i32 {
        self.y + self.height as i32
    }

    pub fn center_x(&self) -> i32 {
        self.x + self.width as i32 / 2
    }
}

/// Where the mascot should perch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerchTarget {
    /// Sit on top of a foreign window.
    Window {
        x11_id: u32,
        /// Y position for the mascot's top-left (so bottom aligns with window top).
        mascot_y: i32,
    },
    /// Rest on the screen bottom edge.
    Ground {
        mascot_y: i32,
    },
    /// No perch target found (mascot is above all windows and screen).
    None,
}

/// Snap threshold in pixels — how close the mascot bottom must be to a
/// window top edge to "snap" onto it.
const SNAP_THRESHOLD: i32 = 20;

/// Find the best perch target for the mascot.
///
/// Looks for the highest window top-edge that is below the mascot's bottom,
/// where the mascot's horizontal center overlaps the window's width.
pub fn find_perch_target(
    mascot: &MascotRect,
    windows: &[ForeignWindow],
    _screen_width: u32,
    screen_height: u32,
) -> PerchTarget {
    let mascot_bottom = mascot.bottom();
    let mascot_cx = mascot.center_x();

    let mut best: Option<(u32, i32)> = None; // (x11_id, window_top_y)

    for w in windows {
        let win_top = w.y;
        let win_left = w.x;
        let win_right = w.x + w.width as i32;

        // Mascot center must overlap window horizontally
        if mascot_cx < win_left || mascot_cx > win_right {
            continue;
        }

        // Window top must be at or below mascot bottom (within snap threshold)
        // i.e. window_top >= mascot_bottom - snap_threshold
        if win_top < mascot_bottom - SNAP_THRESHOLD {
            continue;
        }

        // Pick the closest (highest) surface below the mascot
        match best {
            Some((_, best_y)) if win_top < best_y => {
                best = Some((w.x11_id, win_top));
            }
            None => {
                best = Some((w.x11_id, win_top));
            }
            _ => {}
        }
    }

    // Check ground (screen bottom)
    let ground_y = screen_height as i32;
    let ground_mascot_y = ground_y - mascot.height as i32;

    if let Some((x11_id, win_top)) = best {
        let win_mascot_y = win_top - mascot.height as i32;

        // If ground is closer than the window, prefer ground
        if ground_y >= mascot_bottom - SNAP_THRESHOLD && ground_y < win_top {
            return PerchTarget::Ground {
                mascot_y: ground_mascot_y,
            };
        }

        PerchTarget::Window {
            x11_id,
            mascot_y: win_mascot_y,
        }
    } else if ground_y >= mascot_bottom - SNAP_THRESHOLD {
        PerchTarget::Ground {
            mascot_y: ground_mascot_y,
        }
    } else {
        PerchTarget::None
    }
}

/// Simple gravity simulation state.
#[derive(Debug, Clone)]
pub struct GravityState {
    pub velocity_y: f32,
    pub is_falling: bool,
}

impl Default for GravityState {
    fn default() -> Self {
        Self {
            velocity_y: 0.0,
            is_falling: false,
        }
    }
}

impl GravityState {
    /// Start a fall from current position.
    pub fn start_fall(&mut self) {
        self.velocity_y = 0.0;
        self.is_falling = true;
    }

    /// Advance one timestep. Returns the delta-Y in pixels.
    pub fn step(&mut self, dt: f32, gravity: f32) -> f32 {
        if !self.is_falling {
            return 0.0;
        }
        self.velocity_y += gravity * dt;
        self.velocity_y * dt
    }

    /// Land on a surface.
    pub fn land(&mut self) {
        self.is_falling = false;
        self.velocity_y = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mascot(x: i32, y: i32) -> MascotRect {
        MascotRect {
            x,
            y,
            width: 400,
            height: 600,
        }
    }

    fn make_window(x11_id: u32, x: i32, y: i32, width: u32, height: u32) -> ForeignWindow {
        ForeignWindow {
            x11_id,
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn no_windows_no_ground() {
        let mascot = make_mascot(100, 0);
        let result = find_perch_target(&mascot, &[], 1920, 10);
        // Screen height 10 is above mascot bottom (600), so no target
        assert_eq!(result, PerchTarget::None);
    }

    #[test]
    fn falls_to_ground() {
        let mascot = make_mascot(100, 100);
        let result = find_perch_target(&mascot, &[], 1920, 1080);
        assert_eq!(
            result,
            PerchTarget::Ground {
                mascot_y: 1080 - 600
            }
        );
    }

    #[test]
    fn perches_on_window() {
        let mascot = make_mascot(100, 100);
        let windows = vec![make_window(42, 0, 750, 800, 400)];
        let result = find_perch_target(&mascot, &windows, 1920, 1080);
        assert_eq!(
            result,
            PerchTarget::Window {
                x11_id: 42,
                mascot_y: 750 - 600,
            }
        );
    }

    #[test]
    fn picks_closest_window() {
        let mascot = make_mascot(100, 100);
        let windows = vec![
            make_window(1, 0, 800, 800, 200),  // further
            make_window(2, 0, 720, 800, 200),  // closer
            make_window(3, 0, 900, 800, 200),  // even further
        ];
        let result = find_perch_target(&mascot, &windows, 1920, 1080);
        assert_eq!(
            result,
            PerchTarget::Window {
                x11_id: 2,
                mascot_y: 720 - 600,
            }
        );
    }

    #[test]
    fn ignores_non_overlapping_window() {
        let mascot = make_mascot(100, 100); // center_x = 300
        let windows = vec![make_window(42, 800, 750, 400, 200)]; // 800..1200, doesn't overlap cx=300
        let result = find_perch_target(&mascot, &windows, 1920, 1080);
        assert_eq!(
            result,
            PerchTarget::Ground {
                mascot_y: 1080 - 600
            }
        );
    }

    #[test]
    fn ignores_window_above_mascot() {
        let mascot = make_mascot(100, 400); // bottom = 1000
        let windows = vec![make_window(42, 0, 200, 800, 100)]; // top at 200, well above mascot bottom
        let result = find_perch_target(&mascot, &windows, 1920, 1080);
        assert_eq!(
            result,
            PerchTarget::Ground {
                mascot_y: 1080 - 600
            }
        );
    }

    #[test]
    fn gravity_not_falling() {
        let mut g = GravityState::default();
        assert_eq!(g.step(0.016, 800.0), 0.0);
    }

    #[test]
    fn gravity_accelerates() {
        let mut g = GravityState::default();
        g.start_fall();
        let dy1 = g.step(0.016, 800.0);
        let dy2 = g.step(0.016, 800.0);
        assert!(dy2 > dy1, "should accelerate: dy2={} > dy1={}", dy2, dy1);
    }

    #[test]
    fn gravity_land_stops() {
        let mut g = GravityState::default();
        g.start_fall();
        g.step(0.016, 800.0);
        g.land();
        assert!(!g.is_falling);
        assert_eq!(g.velocity_y, 0.0);
        assert_eq!(g.step(0.016, 800.0), 0.0);
    }

    #[test]
    fn mascot_rect_helpers() {
        let m = make_mascot(100, 200);
        assert_eq!(m.bottom(), 800);
        assert_eq!(m.center_x(), 300);
    }
}
