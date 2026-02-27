/// Drag delta calculation for window repositioning.
/// Pure math — no Bevy dependency — fully testable.

use bevy::prelude::{Resource, Vec2};

#[derive(Debug, Default, Clone, Resource)]
pub struct DragState {
    pub active: bool,
    /// Cursor position when drag started (physical pixels).
    pub start_cursor: Vec2,
    /// Window position when drag started.
    pub start_window: Vec2,
}

impl DragState {
    pub fn begin(&mut self, cursor: Vec2, window_pos: Vec2) {
        self.active = true;
        self.start_cursor = cursor;
        self.start_window = window_pos;
    }

    pub fn end(&mut self) {
        self.active = false;
    }

    /// Compute new window position given current cursor position.
    pub fn compute_new_position(&self, current_cursor: Vec2) -> Vec2 {
        if !self.active {
            return self.start_window;
        }
        let delta = current_cursor - self.start_cursor;
        self.start_window + delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_not_active() {
        let drag = DragState::default();
        assert!(!drag.active);
    }

    #[test]
    fn begin_sets_active() {
        let mut drag = DragState::default();
        drag.begin(Vec2::ZERO, Vec2::new(100.0, 200.0));
        assert!(drag.active);
    }

    #[test]
    fn end_clears_active() {
        let mut drag = DragState::default();
        drag.begin(Vec2::ZERO, Vec2::ZERO);
        drag.end();
        assert!(!drag.active);
    }

    #[test]
    fn no_movement_returns_start_position() {
        let mut drag = DragState::default();
        drag.begin(Vec2::new(50.0, 50.0), Vec2::new(100.0, 200.0));
        let new_pos = drag.compute_new_position(Vec2::new(50.0, 50.0));
        assert_eq!(new_pos, Vec2::new(100.0, 200.0));
    }

    #[test]
    fn move_right_50px() {
        let mut drag = DragState::default();
        drag.begin(Vec2::new(0.0, 0.0), Vec2::new(100.0, 100.0));
        let new_pos = drag.compute_new_position(Vec2::new(50.0, 0.0));
        assert_eq!(new_pos, Vec2::new(150.0, 100.0));
    }

    #[test]
    fn move_up_and_left() {
        let mut drag = DragState::default();
        drag.begin(Vec2::new(200.0, 200.0), Vec2::new(500.0, 300.0));
        let new_pos = drag.compute_new_position(Vec2::new(150.0, 170.0));
        assert_eq!(new_pos, Vec2::new(450.0, 270.0));
    }

    #[test]
    fn inactive_returns_start_window() {
        let drag = DragState {
            active: false,
            start_window: Vec2::new(42.0, 99.0),
            ..Default::default()
        };
        let pos = drag.compute_new_position(Vec2::new(999.0, 999.0));
        assert_eq!(pos, Vec2::new(42.0, 99.0));
    }

    #[test]
    fn multiple_moves_from_same_origin() {
        let mut drag = DragState::default();
        drag.begin(Vec2::new(100.0, 100.0), Vec2::new(200.0, 200.0));

        let p1 = drag.compute_new_position(Vec2::new(110.0, 120.0));
        assert_eq!(p1, Vec2::new(210.0, 220.0));

        let p2 = drag.compute_new_position(Vec2::new(80.0, 90.0));
        assert_eq!(p2, Vec2::new(180.0, 190.0));
    }
}
