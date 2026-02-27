use bevy::prelude::*;
use bevy::window::{CompositeAlphaMode, ExitCondition, WindowLevel, WindowResolution};

mod audio;
mod chat;
mod error;
mod events;
mod llm;
mod mascot;
mod pipeline;
mod state;
mod stt;
mod tts;
mod window;

fn main() {
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
        .insert_resource(state::AppConfig::default())
        .insert_resource(state::PipelineState::default())
        .add_message::<events::PipelineMessage>()
        .add_plugins(mascot::MascotPlugin)
        .add_plugins(pipeline::relay::PipelineRelayPlugin)
        .add_plugins(pipeline::plugin::PipelinePlugin)
        .add_plugins(stt::loader::WhisperLoaderPlugin)
        .add_plugins(chat::overlay::ChatOverlayPlugin)
        .add_plugins(window::WindowManagerPlugin)
        .run();
}
