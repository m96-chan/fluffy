use bevy::prelude::*;
use bevy::window::{CompositeAlphaMode, ExitCondition, WindowLevel, WindowResolution};
use std::fs;
use std::path::Path;

mod audio;
mod chat;
mod error;
mod events;
mod llm;
mod mascot;
mod perch;
mod pipeline;
mod state;
mod stt;
mod tts;
mod window;

fn main() {
    load_dotenv_if_exists(".env");
    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                exit_condition: ExitCondition::DontExit,
                close_when_requested: false,
                primary_window: Some(Window {
                    title: "Fluffy".to_string(),
                    transparent: true,
                    decorations: false,
                    window_level: WindowLevel::AlwaysOnTop,
                    composite_alpha_mode: CompositeAlphaMode::PreMultiplied,
                    resizable: false,
                    // 左400px=キャラクター、右350px=チャットパネル（固定）
                    resolution: WindowResolution::new(
                        chat::state::WINDOW_WIDTH,
                        chat::state::WINDOW_HEIGHT,
                    ),
                    ..default()
                }),
                ..default()
            }),
        )
        .insert_resource(ClearColor(Color::NONE))
        .insert_resource(state::AppConfig::load())
        .insert_resource(state::PipelineState::default())
        .add_message::<events::PipelineMessage>()
        .add_plugins(mascot::MascotPlugin)
        .add_plugins(pipeline::relay::PipelineRelayPlugin)
        .add_plugins(pipeline::plugin::PipelinePlugin)
        .add_plugins(stt::loader::WhisperLoaderPlugin)
        .add_plugins(chat::overlay::ChatOverlayPlugin)
        .add_plugins(perch::PerchPlugin)
        .add_plugins(window::WindowManagerPlugin)
        .run();
}

fn load_dotenv_if_exists(path: impl AsRef<Path>) {
    let path = path.as_ref();
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((raw_key, raw_val)) = trimmed.split_once('=') else {
            continue;
        };
        let key = raw_key.trim();
        if key.is_empty() || std::env::var_os(key).is_some() {
            continue;
        }
        let mut val = raw_val.trim().to_string();
        if (val.starts_with('"') && val.ends_with('"')) || (val.starts_with('\'') && val.ends_with('\'')) {
            val = val[1..val.len() - 1].to_string();
        }
        std::env::set_var(key, val);
    }
}
