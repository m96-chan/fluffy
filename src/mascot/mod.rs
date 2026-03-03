use bevy::app::AppExit;
use bevy::gltf::Gltf;
use bevy::prelude::*;
use bevy::window::WindowCloseRequested;
use std::time::Duration;

use crate::events::PipelineMessage;
use crate::perch::{PerchMode, PerchState};

pub struct MascotPlugin;

impl Plugin for MascotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_scene)
            .insert_resource(MascotClipSet::default())
            .insert_resource(ExitGreetingRequest::default())
            .insert_resource(BlinkState::default())
            .insert_resource(LipSyncState::default())
            .insert_resource(MascotBehaviorState::default())
            .insert_resource(YawnSchedule::default())
            .insert_resource(ExpressionState::default())
            .add_systems(
                Update,
                (
                    spawn_mascot_scene,
                    fit_mascot_to_window_once,
                    resolve_blink_target_once,
                    resolve_lip_sync_target_once,
                    resolve_expression_targets_once,
                    drive_blink,
                    drive_lip_sync,
                    drive_expression,
                    prepare_mascot_clips,
                    setup_player_graph,
                    handle_window_close_request,
                    handle_pipeline_messages,
                    drive_mascot_animation,
                    sync_perch_to_animation,
                ),
            )
            .add_systems(
                PostUpdate,
                (pin_hips_bone, pin_finger_bones, correct_bone_twist)
                    .after(bevy::app::AnimationSystems),
            );
    }
}

#[derive(Component)]
pub struct MascotTag;

#[derive(Component)]
struct ShadowTag;

#[derive(Component)]
struct MascotGltfHandle(pub Handle<Gltf>);

#[derive(Component)]
struct MascotSpawned;

#[derive(Component)]
struct MascotFitted;

/// Marker for the skeleton root bone (Hips) whose translation must be
/// pinned to prevent animation root motion from pushing the model off-screen
/// (the 462x parent scale amplifies any bone translation enormously).
#[derive(Component)]
struct PinnedHipsBone {
    rest_translation: Vec3,
}

/// Marker for finger bones whose rotation is pinned to their rest pose.
/// Mixamo animations often include finger keyframes that don't match
/// the model's hand shape, causing unnatural finger poses.
#[derive(Component)]
struct PinnedFingerBone {
    rest_rotation: Quat,
}

/// Post-animation corrective twist for bones whose twist axis is flipped
/// due to Mixamo→VRM skeleton retargeting mismatch.
#[derive(Component)]
struct TwistCorrectedBone {
    /// Correction quaternion applied after animation: final = animated * correction
    correction: Quat,
}

#[derive(Component)]
struct MascotAnimState {
    idle_node: AnimationNodeIndex,
    thinking_node: AnimationNodeIndex,
    yawn_node: Option<AnimationNodeIndex>,
    greeting_node: Option<AnimationNodeIndex>,
    sitting_node: Option<AnimationNodeIndex>,
    falling_down_node: Option<AnimationNodeIndex>,
    phase: MascotPhase,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MascotPhase {
    StartupGreeting,
    Idle,
    Thinking,
    Yawn,
    ExitGreeting,
    Exiting,
    Sitting,
    FallingDown,
}

#[derive(Resource, Default)]
struct MascotClipSet {
    idle: Option<Handle<AnimationClip>>,
    thinking: Option<Handle<AnimationClip>>,
    yawn: Option<Handle<AnimationClip>>,
    greeting: Option<Handle<AnimationClip>>,
    sitting: Option<Handle<AnimationClip>>,
    falling_down: Option<Handle<AnimationClip>>,
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

#[derive(Resource, Default)]
struct LipSyncState {
    targets: Option<VowelTargets>,
    /// Current smoothed weights (EMA).
    current: [f32; 5],
    /// Target weights from latest message.
    target: [f32; 5],
}

struct VowelTargets {
    entity: Entity,
    /// Morph indices: [aa, ih, ou, ee, oh]
    indices: [usize; 5],
}

#[derive(Resource, Default)]
struct MascotBehaviorState {
    thinking: bool,
}

/// Expression morph target indices resolved from the model.
struct ExpressionTargets {
    entity: Entity,
    /// Maps morph target name → index.
    indices: Vec<(String, usize)>,
}

/// Defines which morph targets to activate for an emotion, with their target weights.
struct EmotionDef {
    targets: &'static [(&'static str, f32)],
}

/// All known emotion definitions. Each emotion drives a set of morph targets.
fn emotion_defs() -> &'static [(&'static str, EmotionDef)] {
    use std::sync::LazyLock;
    static DEFS: LazyLock<Vec<(&'static str, EmotionDef)>> = LazyLock::new(|| vec![
        ("happy", EmotionDef { targets: &[("eye_happy", 1.0), ("mouth_smile_1", 1.0)] }),
        ("sad", EmotionDef { targets: &[("eye_sad", 1.0), ("mouth_n", 0.6)] }),
        ("surprised", EmotionDef { targets: &[("eye_surprise", 1.0), ("mouth_pokan", 1.0)] }),
        ("angry", EmotionDef { targets: &[("eye_angry", 1.0), ("mouth_angry_1", 0.8)] }),
        ("thinking", EmotionDef { targets: &[("eye_nagomi", 0.7), ("mouth_n", 0.4)] }),
        ("excited", EmotionDef { targets: &[("eye_happy", 1.0), ("mouth_smile_2", 1.0)] }),
        ("confused", EmotionDef { targets: &[("eye_maru", 0.8), ("mouth_hawa", 0.7)] }),
        ("shy", EmotionDef { targets: &[("eye_smile_1", 0.8), ("mouth_ω", 0.7)] }),
        ("embarrassed", EmotionDef { targets: &[("eye_smile_2", 0.8), ("mouth_hawa", 0.6)] }),
        ("neutral", EmotionDef { targets: &[] }),
    ]);
    &DEFS
}

#[derive(Resource)]
struct ExpressionState {
    targets: Option<ExpressionTargets>,
    /// Current emotion name (empty = neutral).
    emotion: String,
    /// Per-morph-target current smoothed weight.
    current_weights: Vec<f32>,
    /// Per-morph-target desired weight for current emotion.
    target_weights: Vec<f32>,
}

impl Default for ExpressionState {
    fn default() -> Self {
        Self {
            targets: None,
            emotion: String::new(),
            current_weights: Vec::new(),
            target_weights: Vec::new(),
        }
    }
}

#[derive(Resource)]
struct YawnSchedule {
    next_at_secs: f64,
    rng_state: u64,
}

impl Default for YawnSchedule {
    fn default() -> Self {
        Self {
            next_at_secs: 0.0,
            rng_state: 0x1234_5678_9abc_def0,
        }
    }
}

impl YawnSchedule {
    fn next_interval_secs(&mut self) -> f64 {
        // Deterministic LCG random in [150, 210] sec around 3 minutes.
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let unit = ((self.rng_state >> 32) as f64) / (u32::MAX as f64);
        150.0 + unit * 60.0
    }

    fn bump_from(&mut self, now_secs: f64) {
        self.next_at_secs = now_secs + self.next_interval_secs();
    }
}

fn spawn_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        Camera3d::default(),
        Projection::Orthographic(OrthographicProjection::default_3d()),
        Camera {
            order: 0,
            viewport: Some(bevy::camera::Viewport {
                physical_position: UVec2::new(crate::chat::state::CHAT_PANEL_WIDTH, 0),
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
    config: Res<crate::state::AppConfig>,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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

    // Spawn drop shadow under mascot feet
    if config.shadow_enabled {
        let viewport_half_w = crate::chat::state::MASCOT_WIDTH as f32 * 0.5;
        let shadow_width = (current_height * new_uniform_scale * 0.5).min(viewport_half_w * 0.9);
        let shadow_depth = shadow_width * 0.25;
        let foot_y = -target_height * 0.5;

        let shadow_texture: Handle<Image> = asset_server.load("textures/shadow.png");
        commands.spawn((
            Mesh3d(meshes.add(Plane3d::new(Vec3::Z, Vec2::new(shadow_width, shadow_depth)))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color_texture: Some(shadow_texture),
                base_color: Color::srgba(0.0, 0.0, 0.0, config.shadow_opacity),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                ..default()
            })),
            Transform::from_xyz(0.0, foot_y, -1.0),
            ShadowTag,
        ));
    }
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

    // 3.2s cycle. Last 0.16s: 0->1->0 blink.
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

fn resolve_lip_sync_target_once(
    mut lip: ResMut<LipSyncState>,
    morph_q: Query<(Entity, &MorphWeights, Option<&Name>)>,
    meshes: Res<Assets<Mesh>>,
) {
    if lip.targets.is_some() {
        return;
    }

    const VOWEL_CANDIDATES: &[&[&str]] = &[
        &["vrc.v_aa", "vrc.aa", "aa", "mouth_open", "jawOpen", "mouth_a"],
        &["vrc.v_ih", "vrc.ih", "vrc.v_i", "vrc.i", "ih", "mouth_i"],
        &["vrc.v_ou", "vrc.ou", "vrc.v_u", "vrc.u", "ou", "mouth_u"],
        &["vrc.v_e", "vrc.ee", "vrc.v_ee", "vrc.e", "ee", "mouth_e"],
        &["vrc.v_oh", "vrc.oh", "vrc.v_o", "vrc.o", "oh", "mouth_o"],
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

        let mut indices = [usize::MAX; 5];
        let mut found_count = 0;

        for (v, candidates) in VOWEL_CANDIDATES.iter().enumerate() {
            for cand in *candidates {
                if let Some(i) = names.iter().position(|n| n.eq_ignore_ascii_case(cand)) {
                    indices[v] = i;
                    found_count += 1;
                    break;
                }
            }
        }

        if indices[0] != usize::MAX {
            info!(
                "Mascot: lip sync targets resolved entity={:?} name={:?} aa={} ih={} ou={} ee={} oh={} ({}/5 found)",
                entity,
                name.map(|n| n.as_str()),
                indices[0], indices[1], indices[2], indices[3], indices[4],
                found_count
            );
            lip.targets = Some(VowelTargets { entity, indices });
            return;
        }
    }
}

fn drive_lip_sync(
    time: Res<Time>,
    mut lip: ResMut<LipSyncState>,
    mut morph_q: Query<&mut MorphWeights>,
) {
    let Some(ref targets) = lip.targets else {
        return;
    };
    let entity = targets.entity;
    let indices = targets.indices;
    let Ok(mut morph) = morph_q.get_mut(entity) else {
        return;
    };
    let weights_len = morph.weights().len();

    let alpha = 1.0 - (-time.delta_secs() * 12.0).exp();
    let decay = (1.0 - time.delta_secs() * 8.0).max(0.0);

    for i in 0..5 {
        lip.current[i] += (lip.target[i] - lip.current[i]) * alpha;
        lip.target[i] *= decay;

        if indices[i] < weights_len {
            morph.weights_mut()[indices[i]] = lip.current[i].clamp(0.0, 1.0);
        }
    }
}

fn resolve_expression_targets_once(
    mut expr: ResMut<ExpressionState>,
    morph_q: Query<(Entity, &MorphWeights, Option<&Name>)>,
    meshes: Res<Assets<Mesh>>,
) {
    if expr.targets.is_some() {
        return;
    }

    let defs = emotion_defs();
    let mut wanted: Vec<&str> = Vec::new();
    for (_, def) in defs {
        for (name, _) in def.targets {
            if !wanted.contains(name) {
                wanted.push(name);
            }
        }
    }

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

        let mut found: Vec<(String, usize)> = Vec::new();
        for want in &wanted {
            if let Some(i) = names.iter().position(|n| n.eq_ignore_ascii_case(want)) {
                found.push((want.to_string(), i));
            }
        }

        if !found.is_empty() {
            let n = found.len();
            info!(
                "Mascot: expression targets resolved entity={:?} name={:?} ({} morph targets)",
                entity, name.map(|n| n.as_str()), n,
            );
            expr.current_weights = vec![0.0; found.len()];
            expr.target_weights = vec![0.0; found.len()];
            expr.targets = Some(ExpressionTargets {
                entity,
                indices: found,
            });
            return;
        }
    }
}

fn drive_expression(
    time: Res<Time>,
    mut expr: ResMut<ExpressionState>,
    mut morph_q: Query<&mut MorphWeights>,
) {
    let Some(ref targets) = expr.targets else {
        return;
    };
    let entity = targets.entity;
    let Ok(mut morph) = morph_q.get_mut(entity) else {
        return;
    };
    let weights_len = morph.weights().len();

    let alpha = 1.0 - (-time.delta_secs() * 4.0).exp();

    let morph_indices: Vec<usize> = targets.indices.iter().map(|(_, idx)| *idx).collect();

    for (i, &morph_idx) in morph_indices.iter().enumerate() {
        expr.current_weights[i] += (expr.target_weights[i] - expr.current_weights[i]) * alpha;

        if morph_idx < weights_len {
            morph.weights_mut()[morph_idx] = expr.current_weights[i].clamp(0.0, 1.0);
        }
    }
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
    let mut thinking = None;
    let mut yawn = None;
    let mut sitting = None;
    let mut falling_down = None;

    for (name, clip) in &gltf.named_animations {
        let n = name.to_ascii_lowercase();
        if idle.is_none() && n.contains("idle") {
            idle = Some(clip.clone());
        }
        if greeting.is_none() && (n.contains("greet") || n.contains("standing")) {
            greeting = Some(clip.clone());
        }
        if thinking.is_none() && n.contains("thinking") {
            thinking = Some(clip.clone());
        }
        if yawn.is_none() && n.contains("yawn") {
            yawn = Some(clip.clone());
        }
        if sitting.is_none() && n.contains("sitting") {
            sitting = Some(clip.clone());
        }
        if falling_down.is_none() && (n.contains("falling") || n.contains("fall")) {
            falling_down = Some(clip.clone());
        }
    }

    if idle.is_none() {
        idle = gltf.animations.first().cloned();
    }

    clips.idle = idle;
    clips.greeting = greeting;
    clips.thinking = thinking.or_else(|| clips.idle.clone());
    clips.yawn = yawn;
    clips.sitting = sitting;
    clips.falling_down = falling_down;
    info!(
        "Mascot clips: idle={}, thinking={}, yawn={}, greeting={}, sitting={}, falling={}",
        clips.idle.is_some(),
        clips.thinking.is_some(),
        clips.yawn.is_some(),
        clips.greeting.is_some(),
        clips.sitting.is_some(),
        clips.falling_down.is_some(),
    );
}

fn setup_player_graph(
    clips: Res<MascotClipSet>,
    mut commands: Commands,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut players: Query<(Entity, &mut AnimationPlayer), (Added<AnimationPlayer>, Without<MascotAnimState>)>,
    children_q: Query<&Children>,
    name_q: Query<(&Name, &Transform)>,
) {
    let Some(idle_clip) = clips.idle.clone() else {
        return;
    };

    for (entity, mut player) in &mut players {
        let mut transitions = AnimationTransitions::new();
        let mut graph = AnimationGraph::new();
        let idle_node = graph.add_clip(idle_clip.clone(), 1.0, graph.root);
        let thinking_clip = clips.thinking.clone().unwrap_or(idle_clip.clone());
        let thinking_node = graph.add_clip(thinking_clip, 1.0, graph.root);
        let yawn_node = clips
            .yawn
            .clone()
            .map(|h| graph.add_clip(h, 1.0, graph.root));
        let greeting_node = clips
            .greeting
            .clone()
            .map(|h| graph.add_clip(h, 1.0, graph.root));
        let sitting_node = clips
            .sitting
            .clone()
            .map(|h| graph.add_clip(h, 1.0, graph.root));
        let falling_down_node = clips
            .falling_down
            .clone()
            .map(|h| graph.add_clip(h, 1.0, graph.root));
        let graph_handle = graphs.add(graph);

        let initial_phase = if let Some(greet) = greeting_node {
            transitions
                .play(&mut player, greet, Duration::from_millis(180))
                .set_repeat(bevy::animation::RepeatAnimation::Never);
            MascotPhase::StartupGreeting
        } else {
            transitions
                .play(&mut player, idle_node, Duration::from_millis(180))
                .set_repeat(bevy::animation::RepeatAnimation::Forever);
            MascotPhase::Idle
        };
        commands.entity(entity).insert(MascotAnimState {
            idle_node,
            thinking_node,
            yawn_node,
            greeting_node,
            sitting_node,
            falling_down_node,
            phase: initial_phase,
        });
        info!("Mascot: animation graph initialized; startup={:?}", initial_phase);

        // Walk the skeleton tree to pin special bones.
        // - Hips: pin translation (prevent animation root motion)
        // - Finger bones: pin rotation (prevent Mixamo finger poses)
        // Walk the skeleton tree to pin special bones.
        let mut stack: Vec<Entity> = Vec::new();
        if let Ok(children) = children_q.get(entity) {
            stack.extend(children.iter());
        }
        let mut finger_count = 0u32;
        let mut twist_count = 0u32;
        while let Some(child) = stack.pop() {
            if let Ok((name, transform)) = name_q.get(child) {
                let n = name.as_str();
                if n == "Hips" {
                    commands.entity(child).insert(PinnedHipsBone {
                        rest_translation: transform.translation,
                    });
                    info!("Mascot: pinned Hips bone at rest={:?}", transform.translation);
                }
                if is_finger_bone(n) {
                    commands.entity(child).insert(PinnedFingerBone {
                        rest_rotation: transform.rotation,
                    });
                    finger_count += 1;
                }
                // Hand twist is 180° off due to Mixamo→VRM retargeting.
                // Correct by rotating around the bone's length axis (Y).
                if n == "Hand.L" || n == "Hand.R" {
                    commands.entity(child).insert(TwistCorrectedBone {
                        correction: Quat::from_rotation_y(std::f32::consts::PI),
                    });
                    twist_count += 1;
                }
            }
            if let Ok(grandchildren) = children_q.get(child) {
                stack.extend(grandchildren.iter());
            }
        }
        if finger_count > 0 {
            info!("Mascot: pinned {} finger bones to rest rotation", finger_count);
        }
        if twist_count > 0 {
            info!("Mascot: twist-corrected {} upper arm bones", twist_count);
        }

        commands.entity(entity).insert(AnimationGraphHandle(graph_handle));
        commands.entity(entity).insert(transitions);
    }
}

/// Pin the Hips bone's translation after animation evaluation.
/// This removes root motion while preserving rotation/scale animation.
/// Runs in PostUpdate, after Bevy's animation system has applied clips.
fn pin_hips_bone(mut q: Query<(&PinnedHipsBone, &mut Transform)>) {
    for (pin, mut transform) in &mut q {
        transform.translation = pin.rest_translation;
    }
}

/// Pin finger bones' rotation to their rest pose after animation evaluation.
fn pin_finger_bones(mut q: Query<(&PinnedFingerBone, &mut Transform)>) {
    for (pin, mut transform) in &mut q {
        transform.rotation = pin.rest_rotation;
    }
}

/// Apply corrective twist to bones after animation evaluation.
fn correct_bone_twist(mut q: Query<(&TwistCorrectedBone, &mut Transform)>) {
    for (twist, mut transform) in &mut q {
        transform.rotation = transform.rotation * twist.correction;
    }
}

fn is_finger_bone(name: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "Thumb", "Index", "Middle", "Ring", "Pinky", "Little",
    ];
    const SUFFIXES: &[&str] = &["Proximal", "Intermediate", "Distal"];
    KEYWORDS.iter().any(|k| name.contains(k))
        && SUFFIXES.iter().any(|s| name.contains(s))
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
    time: Res<Time>,
    mut request: ResMut<ExitGreetingRequest>,
    behavior: Res<MascotBehaviorState>,
    mut yawn: ResMut<YawnSchedule>,
    mut app_exit: MessageWriter<AppExit>,
    mut players: Query<(&mut AnimationPlayer, &mut AnimationTransitions, &mut MascotAnimState)>,
) {
    let now = time.elapsed_secs_f64();
    if yawn.next_at_secs <= 0.0 {
        yawn.bump_from(now);
    }

    let desired_loop = if behavior.thinking {
        MascotPhase::Thinking
    } else {
        MascotPhase::Idle
    };

    for (mut player, mut transitions, mut state) in &mut players {
        if request.requested && state.phase != MascotPhase::ExitGreeting && state.phase != MascotPhase::Exiting {
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
            MascotPhase::StartupGreeting => {
                let finished = state
                    .greeting_node
                    .and_then(|n| player.animation(n))
                    .map(|a| a.is_finished())
                    .unwrap_or(true);
                if finished {
                    let node = if desired_loop == MascotPhase::Thinking {
                        state.thinking_node
                    } else {
                        state.idle_node
                    };
                    transitions
                        .play(&mut player, node, Duration::from_millis(180))
                        .set_repeat(bevy::animation::RepeatAnimation::Forever);
                    state.phase = desired_loop;
                }
            }
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
            MascotPhase::Yawn => {
                let finished = state
                    .yawn_node
                    .and_then(|n| player.animation(n))
                    .map(|a| a.is_finished())
                    .unwrap_or(true);
                if finished {
                    let node = if desired_loop == MascotPhase::Thinking {
                        state.thinking_node
                    } else {
                        state.idle_node
                    };
                    transitions
                        .play(&mut player, node, Duration::from_millis(180))
                        .set_repeat(bevy::animation::RepeatAnimation::Forever);
                    state.phase = desired_loop;
                    yawn.bump_from(now);
                }
            }
            MascotPhase::Idle | MascotPhase::Thinking => {
                if !behavior.thinking && !request.requested && now >= yawn.next_at_secs {
                    if let Some(yawn_node) = state.yawn_node {
                        transitions
                            .play(&mut player, yawn_node, Duration::from_millis(120))
                            .set_repeat(bevy::animation::RepeatAnimation::Never);
                        state.phase = MascotPhase::Yawn;
                        info!("Mascot: yawn started");
                        continue;
                    } else {
                        yawn.bump_from(now);
                    }
                }

                if state.phase != desired_loop {
                    let node = if desired_loop == MascotPhase::Thinking {
                        state.thinking_node
                    } else {
                        state.idle_node
                    };
                    transitions
                        .play(&mut player, node, Duration::from_millis(180))
                        .set_repeat(bevy::animation::RepeatAnimation::Forever);
                    state.phase = desired_loop;
                }
            }
            MascotPhase::Exiting => {}
            // Perch-driven phases — handled by sync_perch_to_animation
            MascotPhase::Sitting | MascotPhase::FallingDown => {}
        }
    }
}

fn handle_pipeline_messages(
    mut reader: MessageReader<PipelineMessage>,
    time: Res<Time>,
    mut behavior: ResMut<MascotBehaviorState>,
    mut yawn: ResMut<YawnSchedule>,
    mut lip: ResMut<LipSyncState>,
    mut expr: ResMut<ExpressionState>,
) {
    let now = time.elapsed_secs_f64();
    for msg in reader.read() {
        match msg {
            PipelineMessage::PhaseChanged(phase) => {
                info!("Phase: {:?}", phase);
                if matches!(phase, crate::events::MascotPhase::Thinking) {
                    behavior.thinking = true;
                    yawn.bump_from(now);
                } else if !matches!(phase, crate::events::MascotPhase::Processing) {
                    behavior.thinking = false;
                }
                if matches!(phase, crate::events::MascotPhase::Idle | crate::events::MascotPhase::Listening) {
                    set_emotion(&mut expr, "neutral");
                }
            }
            PipelineMessage::SttResult { .. } => {
                yawn.bump_from(now);
                behavior.thinking = true;
            }
            PipelineMessage::LlmToken { .. } => {
                behavior.thinking = false;
                yawn.bump_from(now);
            }
            PipelineMessage::LlmDone => {
                behavior.thinking = false;
                yawn.bump_from(now);
            }
            PipelineMessage::LipSync { aa, ih, ou, ee, oh } => {
                lip.target = [*aa, *ih, *ou, *ee, *oh];
            }
            PipelineMessage::EmotionChange { emotion } => {
                set_emotion(&mut expr, emotion);
            }
            PipelineMessage::Interrupted => {
                set_emotion(&mut expr, "surprised");
                behavior.thinking = false;
                yawn.bump_from(now);
            }
            PipelineMessage::PipelineError { .. } => {}
        }
    }
}

/// Synchronize perch mode → mascot animation phase.
fn sync_perch_to_animation(
    perch: Res<PerchState>,
    mut players: Query<(&mut AnimationPlayer, &mut AnimationTransitions, &mut MascotAnimState)>,
) {
    for (mut player, mut transitions, mut state) in &mut players {
        if matches!(
            state.phase,
            MascotPhase::ExitGreeting | MascotPhase::Exiting | MascotPhase::StartupGreeting
        ) {
            return;
        }

        match perch.mode {
            PerchMode::Perched { .. } => {
                if state.phase != MascotPhase::Sitting {
                    if let Some(sitting) = state.sitting_node {
                        transitions
                            .play(&mut player, sitting, Duration::from_millis(300))
                            .set_repeat(bevy::animation::RepeatAnimation::Forever);
                        state.phase = MascotPhase::Sitting;
                        info!("Mascot: animation → Sitting");
                    }
                }
            }
            PerchMode::Falling => {
                if state.phase != MascotPhase::FallingDown {
                    if let Some(falling) = state.falling_down_node {
                        transitions
                            .play(&mut player, falling, Duration::from_millis(120))
                            .set_repeat(bevy::animation::RepeatAnimation::Never);
                        state.phase = MascotPhase::FallingDown;
                        info!("Mascot: animation → FallingDown");
                    }
                }
            }
            PerchMode::Standing => {
                if matches!(state.phase, MascotPhase::Sitting | MascotPhase::FallingDown) {
                    transitions
                        .play(&mut player, state.idle_node, Duration::from_millis(300))
                        .set_repeat(bevy::animation::RepeatAnimation::Forever);
                    state.phase = MascotPhase::Idle;
                    info!("Mascot: animation → Idle (from perch)");
                }
            }
        }
    }
}

/// Update expression target weights for the given emotion.
fn set_emotion(expr: &mut ExpressionState, emotion: &str) {
    if expr.emotion == emotion {
        return;
    }
    info!("Mascot: emotion → {}", emotion);
    expr.emotion = emotion.to_string();

    for w in expr.target_weights.iter_mut() {
        *w = 0.0;
    }

    let Some(ref targets) = expr.targets else {
        return;
    };

    let defs = emotion_defs();
    let def = defs.iter().find(|(name, _)| *name == emotion);
    let Some((_, def)) = def else {
        return;
    };

    for (morph_name, weight) in def.targets {
        if let Some(i) = targets.indices.iter().position(|(n, _)| n.eq_ignore_ascii_case(morph_name)) {
            expr.target_weights[i] = *weight;
        }
    }
}
