/// Window perching plugin — lets the mascot sit on top of other windows.
///
/// Architecture:
///   X11 background thread (tracker.rs) → std::sync::mpsc → Bevy systems (this file)
///   Physics (physics.rs) is pure math with no Bevy dependency.
pub mod physics;
pub mod tracker;

use std::sync::{mpsc, Mutex};
use std::time::Duration;

use bevy::prelude::*;
use bevy::window::{PrimaryWindow, WindowPosition};
use tracing::info;

use physics::{GravityState, MascotRect, PerchTarget};
use tracker::WindowSnapshot;

use crate::state::AppConfig;
use crate::window::config::WindowConfig;

pub struct PerchPlugin;

impl Plugin for PerchPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PerchState::default())
            .add_systems(Startup, start_tracker)
            .add_systems(
                Update,
                (
                    drain_window_snapshots,
                    evaluate_perch,
                    apply_gravity,
                    update_window_position,
                )
                    .chain(),
            );
    }
}

/// Current perching mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerchMode {
    /// Normal standing/idle — no perch active.
    Standing,
    /// Sitting on top of a foreign window.
    Perched { x11_id: u32 },
    /// Falling through air.
    Falling,
}

/// Bevy resource holding perch state.
pub struct PerchState {
    receiver: Option<Mutex<mpsc::Receiver<WindowSnapshot>>>,
    latest_snapshot: Option<WindowSnapshot>,
    pub mode: PerchMode,
    pub gravity: GravityState,
    /// Cooldown after drag ends before re-evaluating perch (seconds remaining).
    drag_cooldown: f32,
    /// Whether drag was active last frame.
    was_dragging: bool,
}

impl Resource for PerchState {}

impl Default for PerchState {
    fn default() -> Self {
        Self {
            receiver: None,
            latest_snapshot: None,
            mode: PerchMode::Standing,
            gravity: GravityState::default(),
            drag_cooldown: 0.0,
            was_dragging: false,
        }
    }
}

/// Startup system: spawn the X11 tracker thread.
fn start_tracker(config: Res<AppConfig>, mut perch: ResMut<PerchState>) {
    if !config.perch_enabled {
        info!("Perch: disabled in config");
        return;
    }

    let receiver = tracker::spawn_tracker("Fluffy".to_string(), Duration::from_millis(250));

    match receiver {
        Some(rx) => {
            perch.receiver = Some(Mutex::new(rx));
            info!("Perch: tracker started");
        }
        None => {
            info!("Perch: X11 not available, perching disabled");
        }
    }
}

/// Drain all pending snapshots from the tracker thread, keeping only the latest.
fn drain_window_snapshots(mut perch: ResMut<PerchState>) {
    let Some(ref rx_mutex) = perch.receiver else {
        return;
    };
    let Ok(rx) = rx_mutex.lock() else {
        return;
    };

    let mut latest = None;
    loop {
        match rx.try_recv() {
            Ok(snap) => latest = Some(snap),
            Err(_) => break,
        }
    }
    drop(rx);

    if latest.is_some() {
        perch.latest_snapshot = latest;
    }
}

/// Evaluate whether the mascot should perch, fall, or stand.
fn evaluate_perch(
    time: Res<Time>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut perch: ResMut<PerchState>,
    window_q: Query<&Window, With<PrimaryWindow>>,
) {
    if perch.receiver.is_none() {
        return;
    }

    // Detect drag start/end for cooldown
    let dragging = mouse.pressed(MouseButton::Left);
    if dragging {
        perch.was_dragging = true;
        return; // Skip evaluation during drag
    }
    if perch.was_dragging && !dragging {
        // Drag just ended — set cooldown
        perch.was_dragging = false;
        perch.drag_cooldown = 0.3;
        perch.mode = PerchMode::Standing;
        perch.gravity.land();
        return;
    }

    // Tick cooldown
    if perch.drag_cooldown > 0.0 {
        perch.drag_cooldown -= time.delta_secs();
        return;
    }

    let Some(ref snapshot) = perch.latest_snapshot else {
        return;
    };
    let Ok(win) = window_q.single() else {
        return;
    };
    let WindowPosition::At(pos) = win.position else {
        return;
    };

    let mascot = MascotRect {
        x: pos.x,
        y: pos.y,
        width: win.physical_width(),
        height: win.physical_height(),
    };

    let target = physics::find_perch_target(
        &mascot,
        &snapshot.windows,
        snapshot.screen_width,
        snapshot.screen_height,
    );

    match (&perch.mode, &target) {
        // Already perched — check if target window still exists
        (PerchMode::Perched { x11_id }, _) => {
            let still_valid = snapshot
                .windows
                .iter()
                .any(|w| w.x11_id == *x11_id);
            if !still_valid {
                // Target window gone — start falling
                info!("Perch: target window lost, falling");
                perch.mode = PerchMode::Falling;
                perch.gravity.start_fall();
            }
        }

        // Standing — look for a perch target
        (PerchMode::Standing, PerchTarget::Window { x11_id, .. }) => {
            info!("Perch: perching on window {:#x}", x11_id);
            perch.mode = PerchMode::Perched { x11_id: *x11_id };
        }
        (PerchMode::Standing, PerchTarget::Ground { .. }) => {
            // Already on ground, nothing to do
        }
        (PerchMode::Standing, PerchTarget::None) => {
            // No surface below — start falling
            perch.mode = PerchMode::Falling;
            perch.gravity.start_fall();
        }

        // Falling — handled by apply_gravity
        (PerchMode::Falling, _) => {}
    }
}

/// Apply gravity when falling, landing when a surface is reached.
fn apply_gravity(
    time: Res<Time>,
    config: Res<AppConfig>,
    mut perch: ResMut<PerchState>,
    mut window_q: Query<&mut Window, With<PrimaryWindow>>,
    mut win_cfg: ResMut<WindowConfig>,
) {
    if perch.mode != PerchMode::Falling {
        return;
    }

    let snapshot = match perch.latest_snapshot.clone() {
        Some(s) => s,
        None => return,
    };
    let Ok(mut win) = window_q.single_mut() else {
        return;
    };
    let WindowPosition::At(mut pos) = win.position else {
        return;
    };

    let dy = perch.gravity.step(time.delta_secs(), config.perch_gravity);
    pos.y += dy as i32;

    // Check for landing
    let mascot = MascotRect {
        x: pos.x,
        y: pos.y,
        width: win.physical_width(),
        height: win.physical_height(),
    };

    let target = physics::find_perch_target(
        &mascot,
        &snapshot.windows,
        snapshot.screen_width,
        snapshot.screen_height,
    );

    match target {
        PerchTarget::Window { x11_id, mascot_y } => {
            if pos.y >= mascot_y {
                pos.y = mascot_y;
                perch.gravity.land();
                perch.mode = PerchMode::Perched { x11_id };
                info!("Perch: landed on window {:#x} at y={}", x11_id, mascot_y);
            }
        }
        PerchTarget::Ground { mascot_y } => {
            if pos.y >= mascot_y {
                pos.y = mascot_y;
                perch.gravity.land();
                perch.mode = PerchMode::Standing;
                info!("Perch: landed on ground at y={}", mascot_y);
            }
        }
        PerchTarget::None => {
            let max_y = snapshot.screen_height as i32 - mascot.height as i32;
            if pos.y > max_y {
                pos.y = max_y;
                perch.gravity.land();
                perch.mode = PerchMode::Standing;
            }
        }
    }

    win.position = WindowPosition::At(pos);
    win_cfg.x = pos.x;
    win_cfg.y = pos.y;
}

/// When perched, track the target window's movement.
fn update_window_position(
    perch: Res<PerchState>,
    mut window_q: Query<&mut Window, With<PrimaryWindow>>,
    mut win_cfg: ResMut<WindowConfig>,
) {
    let PerchMode::Perched { x11_id } = perch.mode else {
        return;
    };

    let Some(ref snapshot) = perch.latest_snapshot else {
        return;
    };
    let Ok(mut win) = window_q.single_mut() else {
        return;
    };
    let WindowPosition::At(mut pos) = win.position else {
        return;
    };

    // Find the target window in the snapshot
    let Some(target) = snapshot.windows.iter().find(|w| w.x11_id == x11_id) else {
        return; // Window not found — evaluate_perch will handle this
    };

    // Align mascot bottom with target window top
    let mascot_y = target.y - win.physical_height() as i32;
    if pos.y != mascot_y {
        pos.y = mascot_y;
        win.position = WindowPosition::At(pos);
        win_cfg.x = pos.x;
        win_cfg.y = pos.y;
    }
}
