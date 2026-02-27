use bevy::app::AppExit;
use bevy::gltf::Gltf;
use bevy::prelude::*;
use bevy::window::WindowCloseRequested;
use std::time::Duration;

use crate::events::PipelineMessage;

pub struct MascotPlugin;

impl Plugin for MascotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_scene)
            .insert_resource(MascotClipSet::default())
            .insert_resource(ExitGreetingRequest::default())
            .insert_resource(BlinkState::default())
            .add_systems(
                Update,
                (
                    spawn_mascot_scene,
                    fit_mascot_to_window_once,
                    resolve_blink_target_once,
                    drive_blink,
                    prepare_mascot_clips,
                    setup_player_graph,
                    handle_window_close_request,
                    drive_mascot_animation,
                    handle_pipeline_messages,
                ),
            );
    }
}

#[derive(Component)]
pub struct MascotTag;

#[derive(Component)]
struct MascotGltfHandle(pub Handle<Gltf>);

#[derive(Component)]
struct MascotSpawned;

#[derive(Component)]
struct MascotFitted;

#[derive(Component)]
struct MascotAnimState {
    greeting_node: Option<AnimationNodeIndex>,
    phase: MascotPhase,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MascotPhase {
    Idle,
    ExitGreeting,
    Exiting,
}

#[derive(Resource, Default)]
struct MascotClipSet {
    idle: Option<Handle<AnimationClip>>,
    greeting: Option<Handle<AnimationClip>>,
}

#[derive(Resource, Default)]
struct ExitGreetingRequest {
    requested: bool,
}

#[derive(Resource)]
struct BlinkState {
    target: Option<(Entity, usize)>,
    elapsed: f32,
}

impl Default for BlinkState {
    fn default() -> Self {
        Self {
            target: None,
            elapsed: 0.0,
        }
    }
}

fn spawn_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        Camera3d::default(),
        Projection::Orthographic(OrthographicProjection::default_3d()),
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
        Transform::from_xyz(0.0, 0.0, 300.0).looking_at(Vec3::ZERO, Vec3::Y),
        AmbientLight {
            color: Color::WHITE,
            brightness: 300.0,
            affects_lightmapped_meshes: false,
        },
    ));

    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            clear_color: bevy::camera::ClearColorConfig::None,
            ..default()
        },
        bevy::ui::IsDefaultUiCamera,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            ..default()
        },
        Transform::from_xyz(1.0, 3.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let gltf_handle: Handle<Gltf> = asset_server.load_with_settings(
        "models/mascot.glb",
        |s: &mut bevy::gltf::GltfLoaderSettings| {
            s.include_source = true;
        },
    );
    commands.spawn((
        MascotGltfHandle(gltf_handle),
        Transform::from_xyz(0.0, 0.0, 0.0),
        Visibility::default(),
        MascotTag,
    ));
}

fn spawn_mascot_scene(
    mut commands: Commands,
    gltf_assets: Res<Assets<Gltf>>,
    mascot_q: Query<(Entity, &MascotGltfHandle), (With<MascotTag>, Without<MascotSpawned>)>,
) {
    let Ok((mascot_entity, handle)) = mascot_q.single() else {
        return;
    };
    let Some(gltf) = gltf_assets.get(&handle.0) else {
        return;
    };

    if let Some(scene) = gltf.scenes.first() {
        commands.entity(mascot_entity).insert(SceneRoot(scene.clone()));
    } else {
        warn!("Mascot GLB has no scene");
    }

    commands.entity(mascot_entity).insert(MascotSpawned);
}

fn fit_mascot_to_window_once(
    mut commands: Commands,
    mut mascot_q: Query<(Entity, &mut Transform), (With<MascotTag>, With<MascotSpawned>, Without<MascotFitted>)>,
    children_q: Query<&Children>,
    global_q: Query<&GlobalTransform>,
) {
    let Ok((mascot_entity, mut root_transform)) = mascot_q.single_mut() else {
        return;
    };

    let mut stack = vec![mascot_entity];
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut count = 0usize;

    while let Some(entity) = stack.pop() {
        if let Ok(gt) = global_q.get(entity) {
            let y = gt.translation().y;
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            count += 1;
        }
        if let Ok(children) = children_q.get(entity) {
            for child in children.iter() {
                stack.push(child);
            }
        }
    }

    if count == 0 {
        return;
    }

    let current_height = max_y - min_y;
    if current_height <= 1e-4 {
        return;
    }

    let target_height = crate::chat::state::WINDOW_HEIGHT as f32 * 0.92;
    let new_uniform_scale = target_height / current_height;
    root_transform.scale = Vec3::splat(new_uniform_scale);
    root_transform.translation = Vec3::new(0.0, -target_height * 0.5, 0.0);

    commands.entity(mascot_entity).insert(MascotFitted);
    info!(
        "Mascot: fitted to viewport (nodes={}, src_height={:.3}, dst_height={:.1}, scale={:.5}, y={:.1})",
        count, current_height, target_height, new_uniform_scale, root_transform.translation.y
    );
}

fn resolve_blink_target_once(
    mut blink: ResMut<BlinkState>,
    morph_q: Query<(Entity, &MorphWeights, Option<&Name>)>,
    meshes: Res<Assets<Mesh>>,
) {
    if blink.target.is_some() {
        return;
    }

    const CANDIDATES: &[&str] = &[
        "vrc.blink",
        "eye_close",
        "eye_blink_1",
        "eye_blink_1_l",
        "eye_blink_1_r",
    ];

    for (entity, morph, name) in &morph_q {
        let Some(mesh_handle) = morph.first_mesh() else {
            continue;
        };
        let Some(mesh) = meshes.get(mesh_handle) else {
            continue;
        };
        let Some(names) = mesh.morph_target_names() else {
            continue;
        };

        let mut found_idx = None;
        for cand in CANDIDATES {
            if let Some(i) = names.iter().position(|n| n.eq_ignore_ascii_case(cand)) {
                found_idx = Some(i);
                break;
            }
        }

        if let Some(idx) = found_idx {
            blink.target = Some((entity, idx));
            info!(
                "Mascot: blink target resolved entity={:?} name={:?} index={}",
                entity,
                name.map(|n| n.as_str()),
                idx
            );
            return;
        }
    }
}

fn drive_blink(
    time: Res<Time>,
    mut blink: ResMut<BlinkState>,
    mut morph_q: Query<&mut MorphWeights>,
) {
    let Some((entity, blink_idx)) = blink.target else {
        return;
    };
    let Ok(mut morph) = morph_q.get_mut(entity) else {
        return;
    };
    if blink_idx >= morph.weights().len() {
        return;
    }

    // 3.2s周期。末尾 0.16s で 0->1->0 の瞬き。
    blink.elapsed += time.delta_secs();
    let phase = blink.elapsed.rem_euclid(3.2);
    let weight = if phase < 0.08 {
        phase / 0.08
    } else if phase < 0.16 {
        1.0 - ((phase - 0.08) / 0.08)
    } else {
        0.0
    };
    morph.weights_mut()[blink_idx] = weight.clamp(0.0, 1.0);
}

fn prepare_mascot_clips(
    mut clips: ResMut<MascotClipSet>,
    mascot_q: Query<&MascotGltfHandle, With<MascotTag>>,
    gltf_assets: Res<Assets<Gltf>>,
) {
    if clips.idle.is_some() {
        return;
    }

    let Ok(handle) = mascot_q.single() else {
        return;
    };
    let Some(gltf) = gltf_assets.get(&handle.0) else {
        return;
    };

    let mut idle = None;
    let mut greeting = None;

    for (name, clip) in &gltf.named_animations {
        let n = name.to_ascii_lowercase();
        if idle.is_none() && n.contains("idle") {
            idle = Some(clip.clone());
        }
        if greeting.is_none() && (n.contains("greet") || n.contains("standing")) {
            greeting = Some(clip.clone());
        }
    }

    if idle.is_none() {
        idle = gltf.animations.first().cloned();
    }

    clips.idle = idle;
    clips.greeting = greeting;
    info!(
        "Mascot clips: idle={}, greeting={}",
        clips.idle.is_some(),
        clips.greeting.is_some()
    );
}

fn setup_player_graph(
    clips: Res<MascotClipSet>,
    mut commands: Commands,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut players: Query<(Entity, &mut AnimationPlayer), (Added<AnimationPlayer>, Without<MascotAnimState>)>,
) {
    let Some(idle_clip) = clips.idle.clone() else {
        return;
    };

    for (entity, mut player) in &mut players {
        let mut transitions = AnimationTransitions::new();
        let mut graph = AnimationGraph::new();
        let idle_node = graph.add_clip(idle_clip.clone(), 1.0, graph.root);
        let greeting_node = clips
            .greeting
            .clone()
            .map(|h| graph.add_clip(h, 1.0, graph.root));
        let graph_handle = graphs.add(graph);

        transitions
            .play(&mut player, idle_node, Duration::from_millis(300))
            .set_repeat(bevy::animation::RepeatAnimation::Forever);
        commands.entity(entity).insert(MascotAnimState {
            greeting_node,
            phase: MascotPhase::Idle,
        });
        info!("Mascot: idle started");

        commands.entity(entity).insert(AnimationGraphHandle(graph_handle));
        commands.entity(entity).insert(transitions);
    }
}

fn handle_window_close_request(
    mut close_reader: MessageReader<WindowCloseRequested>,
    mut request: ResMut<ExitGreetingRequest>,
) {
    for _ in close_reader.read() {
        request.requested = true;
        info!("Mascot: exit requested; will play greeting before shutdown");
    }
}

fn drive_mascot_animation(
    mut request: ResMut<ExitGreetingRequest>,
    mut app_exit: MessageWriter<AppExit>,
    mut players: Query<(&mut AnimationPlayer, &mut AnimationTransitions, &mut MascotAnimState)>,
) {
    for (mut player, mut transitions, mut state) in &mut players {
        if request.requested && state.phase == MascotPhase::Idle {
            if let Some(greet) = state.greeting_node {
                transitions
                    .play(&mut player, greet, Duration::from_millis(300))
                    .set_repeat(bevy::animation::RepeatAnimation::Never);
                state.phase = MascotPhase::ExitGreeting;
                request.requested = false;
                info!("Mascot: exit greeting started");
            } else {
                app_exit.write(AppExit::Success);
                state.phase = MascotPhase::Exiting;
                request.requested = false;
            }
        }

        match state.phase {
            MascotPhase::ExitGreeting => {
                let finished = state
                    .greeting_node
                    .and_then(|n| player.animation(n))
                    .map(|a| a.is_finished())
                    .unwrap_or(true);
                if finished {
                    app_exit.write(AppExit::Success);
                    state.phase = MascotPhase::Exiting;
                    info!("Mascot: exit greeting finished; app exit sent");
                }
            }
            MascotPhase::Idle | MascotPhase::Exiting => {}
        }
    }
}

fn handle_pipeline_messages(mut reader: MessageReader<PipelineMessage>) {
    for msg in reader.read() {
        if let PipelineMessage::PhaseChanged(phase) = msg {
            info!("Phase: {:?}", phase);
        }
    }
}
