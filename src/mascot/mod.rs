use bevy::prelude::*;

use crate::events::PipelineMessage;
use crate::vrm::{VrmExpressionMap, VrmGltfHandle};

pub struct MascotPlugin;

impl Plugin for MascotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_scene)
            .add_systems(Update, (handle_pipeline_messages, update_expressions));
    }
}

/// Marker for the VRM mascot entity.
#[derive(Component)]
pub struct MascotTag;

/// Current lip sync amplitude (0.0 - 1.0).
#[derive(Resource, Default)]
pub struct LipSyncState {
    pub amplitude: f32,
}

/// Desired facial expression name.
#[derive(Resource, Default, Clone)]
pub struct ExpressionState {
    pub emotion: String,
}

fn spawn_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    // 3D カメラ — viewport を左400pxに固定、UI は描画しない
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 0,
            viewport: Some(bevy::camera::Viewport {
                physical_position: UVec2::ZERO,
                physical_size: UVec2::new(
                    crate::chat::state::MASCOT_WIDTH,
                    crate::chat::state::WINDOW_HEIGHT,
                ),
                depth: 0.0..1.0,
            }),
            ..default()
        },
        Transform::from_xyz(0.0, 0.8, 3.5).looking_at(Vec3::new(0.0, 0.8, 0.0), Vec3::Y),
        AmbientLight {
            color: Color::WHITE,
            brightness: 300.0,
            affects_lightmapped_meshes: false,
        },
        // このカメラはUIを描画しない — UIは専用の2Dカメラが担当
        bevy::ui::IsDefaultUiCamera::default(),
    ));

    // UI専用カメラ — フルウィンドウ(750×600)でチャットパネルを描画
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            clear_color: bevy::camera::ClearColorConfig::None,
            ..default()
        },
        bevy::ui::IsDefaultUiCamera,
    ));

    // Directional light
    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            ..default()
        },
        Transform::from_xyz(1.0, 3.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

// VRM mascot — include_source=true でVRM拡張JSONにアクセスできるようにする
    let vrm_handle: Handle<bevy::gltf::Gltf> = asset_server.load_with_settings(
        "models/mascot.vrm",
        |s: &mut bevy::gltf::GltfLoaderSettings| {
            s.include_source = true;
        },
    );
    commands.spawn((
        VrmGltfHandle(vrm_handle),
        // VRM 0.x は +Z が正面、Bevy は -Z がカメラ方向 → Y軸180°回転
        Transform::from_xyz(0.0, 0.0, 0.0)
            .with_rotation(Quat::from_rotation_y(std::f32::consts::PI)),
        Visibility::default(),
        MascotTag,
    ));
}

fn handle_pipeline_messages(
    mut reader: MessageReader<PipelineMessage>,
    mut lip_sync: ResMut<LipSyncState>,
    mut expression_state: ResMut<ExpressionState>,
) {
    for msg in reader.read() {
        match msg {
            PipelineMessage::LipSyncAmplitude { amplitude } => {
                lip_sync.amplitude = *amplitude;
            }
            PipelineMessage::EmotionChange { emotion } => {
                expression_state.emotion = emotion.clone();
            }
            PipelineMessage::PhaseChanged(phase) => {
                info!("Phase: {:?}", phase);
            }
            _ => {}
        }
    }
}

/// Drive morph targets for lip sync and expressions.
fn update_expressions(
    lip_sync: Res<LipSyncState>,
    expression_state: Res<ExpressionState>,
    mascot_query: Query<&VrmExpressionMap, With<MascotTag>>,
    mut morph_query: Query<(&Name, &mut MorphWeights)>,
) {
    let Ok(expr_map) = mascot_query.single() else {
        return;
    };

    // Collect current target weights: expression_name → weight
    let mut targets: Vec<(&str, f32)> = vec![
        ("aa", lip_sync.amplitude.clamp(0.0, 1.0)),
    ];

    let emotion = expression_state.emotion.as_str();
    for name in ["happy", "sad", "surprised", "angry", "relaxed", "joy", "sorrow", "fun"] {
        let normalized = normalize_preset_name(name, emotion);
        targets.push((name, normalized));
    }

    // Apply each target
    for (preset_name, weight) in &targets {
        let Some(binds) = expr_map.binds.get(*preset_name) else {
            continue;
        };
        for bind in binds {
            for (entity_name, mut morph_weights) in morph_query.iter_mut() {
                if entity_name.as_str() == bind.node_name {
                    let weights = morph_weights.weights_mut();
                    if bind.morph_index < weights.len() {
                        weights[bind.morph_index] = *weight;
                    }
                }
            }
        }
    }
}

fn normalize_preset_name(preset: &str, current_emotion: &str) -> f32 {
    if preset == current_emotion {
        1.0
    } else {
        // VRM 0.x aliases
        let aliases: &[(&str, &str)] = &[
            ("joy", "happy"),
            ("sorrow", "sad"),
            ("fun", "relaxed"),
        ];
        for (vrm0, vrm1) in aliases {
            if preset == *vrm0 && current_emotion == *vrm1 {
                return 1.0;
            }
            if preset == *vrm1 && current_emotion == *vrm0 {
                return 1.0;
            }
        }
        0.0
    }
}
