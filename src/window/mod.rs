pub mod config;
pub mod drag;

use bevy::prelude::*;
use bevy::window::{CursorOptions, PrimaryWindow, WindowPosition};

use config::WindowConfig;

pub struct WindowManagerPlugin;

impl Plugin for WindowManagerPlugin {
    fn build(&self, app: &mut App) {
        let cfg = WindowConfig::load();

        app.insert_resource(cfg)
            .add_systems(Startup, apply_initial_config)
            .add_systems(
                Update,
                (handle_drag, handle_click_through_toggle, save_on_exit),
            );
    }
}

// ── Startup ──────────────────────────────────────────────────────────────────

fn apply_initial_config(
    cfg: Res<WindowConfig>,
    mut window_q: Query<(&mut Window, &mut CursorOptions), With<PrimaryWindow>>,
) {
    let Ok((mut win, mut cursor)) = window_q.single_mut() else { return };
    win.position = WindowPosition::At(IVec2::new(cfg.x, cfg.y));
    cursor.hit_test = !cfg.click_through;
}

// ── Drag — left-click triggers OS-native window move via start_drag_move() ──

fn handle_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    mut window_q: Query<&mut Window, With<PrimaryWindow>>,
    mut cfg: ResMut<WindowConfig>,
) {
    let Ok(mut win) = window_q.single_mut() else { return };

    // Left-click starts OS-native drag (winit handles all the math)
    if mouse.just_pressed(MouseButton::Left) {
        win.start_drag_move();
    }

    // Persist position whenever mouse is released
    if mouse.just_released(MouseButton::Left) {
        if let WindowPosition::At(p) = win.position {
            cfg.x = p.x;
            cfg.y = p.y;
            let _ = cfg.save();
        }
    }
}

// ── Click-through toggle (C key) ─────────────────────────────────────────────

fn handle_click_through_toggle(
    keys: Res<ButtonInput<KeyCode>>,
    mut cfg: ResMut<WindowConfig>,
    mut window_q: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if !keys.just_pressed(KeyCode::KeyC) {
        return;
    }

    cfg.click_through = !cfg.click_through;
    let _ = cfg.save();

    for mut cursor in window_q.iter_mut() {
        cursor.hit_test = !cfg.click_through;
        info!(
            "Click-through: {} (C to toggle)",
            if cfg.click_through { "ON" } else { "OFF" }
        );
    }
}

// ── Save position on close ────────────────────────────────────────────────────

fn save_on_exit(
    mut exit: MessageReader<AppExit>,
    mut cfg: ResMut<WindowConfig>,
    window_q: Query<&Window, With<PrimaryWindow>>,
) {
    for _ in exit.read() {
        if let Ok(win) = window_q.single() {
            if let WindowPosition::At(p) = win.position {
                cfg.x = p.x;
                cfg.y = p.y;
                let _ = cfg.save();
            }
        }
    }
}
