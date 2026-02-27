use bevy::prelude::*;
use bevy::window::{CompositeAlphaMode, WindowLevel, WindowResolution};

mod animation;
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
mod vrm;
mod window;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
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
        .add_plugins(vrm::VrmPlugin)
        .insert_resource(ClearColor(Color::NONE))
        .insert_resource(state::AppConfig::default())
        .insert_resource(state::PipelineState::default())
        .insert_resource(mascot::LipSyncState::default())
        .insert_resource(mascot::ExpressionState::default())
        .add_message::<events::PipelineMessage>()
        .add_plugins(mascot::MascotPlugin)
        .add_plugins(pipeline::relay::PipelineRelayPlugin)
        .add_plugins(pipeline::plugin::PipelinePlugin)
        .add_plugins(stt::loader::WhisperLoaderPlugin)
        .add_plugins(chat::overlay::ChatOverlayPlugin)
        .add_plugins(animation::ProceduralAnimationPlugin)
        .add_plugins(window::WindowManagerPlugin)
        .run();
}
