pub mod bones;
pub mod procedural;

use bevy::gltf::Gltf;
use bevy::prelude::*;
use bevy_mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use std::collections::HashMap;

/// リターゲット対象ボーン（親→子の順）。
/// GlobalTransform デルタ適用時に親から順に処理する必要があるため順序が重要。
const RETARGET_BONES: &[&str] = &[
    "Hips",
    "Spine", "Chest", "Neck", "Head",
    "Shoulder.L", "UpperArm.L", "LowerArm.L", "Hand.L",
    "Shoulder.R", "UpperArm.R", "LowerArm.R", "Hand.R",
    "UpperLeg.L", "LowerLeg.L", "Foot.L", "Toe.L",
    "UpperLeg.R", "LowerLeg.R", "Foot.R", "Toe.R",
];

use crate::events::{MascotPhase, PipelineMessage};
use crate::mascot::MascotTag;
use bones::{collect_bones, HumanoidBones};

pub struct ProceduralAnimationPlugin;

impl Plugin for ProceduralAnimationPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CurrentPhase::default())
            .add_systems(
                Update,
                (
                    sync_phase,
                    discover_bones,
                    setup_proxy_skeleton,
                    capture_tpose_from_ibm,
                    copy_proxy_to_manuka,
                )
                    .chain(),
            );
    }
}

// ── Resources / Components ────────────────────────────────────────────────────

#[derive(Resource, Default)]
pub struct CurrentPhase(pub MascotPhase);

/// プロキシスケルトン（idle.glb）のルートエンティティ。
#[derive(Component)]
struct ProxyRoot;

/// プロキシスケルトンのアームチュア（AnimationPlayerを置く場所）。
#[derive(Component)]
struct ProxyArmature;

/// リターゲット用Tポーズ参照。
///
/// プロキシ側は SkinnedMesh の InverseBindMatrices (IBM) から算出: `IBM⁻¹ = GT_tpose`。
/// IBMはBlenderのボーンオリエンテーションを含む正確なバインドポーズGTを持つ。
/// MANUKA側は起動時のGlobalTransform（Tポーズ + 180°Y ルート回転）。
///
/// 使用例（各フレーム）:
///   delta = GT_proxy_anim × GT_proxy_tpose⁻¹  （Tポーズからのワールド空間変化量）
///   GT_manuka_desired = delta × GT_manuka_tpose  （MANUKAのTポーズに適用）
#[derive(Resource)]
struct TposeReference {
    /// ボーン名 → プロキシのTポーズ GlobalTransform 回転（IBM⁻¹から算出）
    proxy: HashMap<String, Quat>,
    /// ボーン名 → MANUKAのTポーズ GlobalTransform 回転
    manuka: HashMap<String, Quat>,
}

// ── Systems ───────────────────────────────────────────────────────────────────

fn sync_phase(mut phase: ResMut<CurrentPhase>, mut reader: MessageReader<PipelineMessage>) {
    for msg in reader.read() {
        if let PipelineMessage::PhaseChanged(p) = msg {
            phase.0 = p.clone();
        }
    }
}

fn discover_bones(
    mut commands: Commands,
    mascot_q: Query<Entity, (With<MascotTag>, Without<HumanoidBones>)>,
    names: Query<(Entity, &Name)>,
    children: Query<&Children>,
) {
    for root in mascot_q.iter() {
        let bones = collect_bones(root, &names, &children);
        if bones.is_ready() {
            info!(
                "ProceduralAnim: bones discovered (chest={:?}, head={:?})",
                bones.chest(),
                bones.head()
            );
            commands.entity(root).insert(bones);
        }
    }
}

// ── Proxy skeleton setup ──────────────────────────────────────────────────────

fn setup_proxy_skeleton(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gltf_assets: Res<Assets<Gltf>>,
    mut anim_graphs: ResMut<Assets<AnimationGraph>>,
    mascot_q: Query<(), (With<MascotTag>, With<HumanoidBones>)>,
    proxy_q: Query<Entity, With<ProxyRoot>>,
    proxy_armature_q: Query<(), With<ProxyArmature>>,
    names: Query<(Entity, &Name)>,
    children: Query<&Children>,
    existing_players: Query<Entity, With<AnimationPlayer>>,
    mut cached_handle: Local<Option<Handle<Gltf>>>,
) {
    if mascot_q.single().is_err() { return; }

    let handle = cached_handle
        .get_or_insert_with(|| asset_server.load("anims/idle.glb"))
        .clone();

    let Some(gltf) = gltf_assets.get(&handle) else { return };

    // ① プロキシシーンをスポーン（まだなければ）
    if proxy_q.is_empty() {
        if let Some(scene_handle) = gltf.default_scene.clone().or_else(|| gltf.scenes.first().cloned()) {
            commands.spawn((
                ProxyRoot,
                SceneRoot(scene_handle),
                Transform::from_xyz(99999.0, 99999.0, 99999.0),
                Visibility::Hidden,
            ));
            info!("Proxy skeleton: scene spawned");
        }
        return;
    }

    // ② AnimationPlayerがまだなければ設定
    if !proxy_armature_q.is_empty() { return; }

    let Ok(proxy_root) = proxy_q.single() else { return };

    let player_entity = find_existing_animation_player(
        proxy_root, &names, &children, &existing_players,
    );

    if let Some(player_entity) = player_entity {
        let clip = gltf.named_animations.get("Action")
            .or_else(|| gltf.animations.first())
            .cloned();

        if let Some(clip) = clip {
            let mut graph = AnimationGraph::new();
            let node = graph.add_clip(clip.clone(), 1.0, graph.root);
            let graph_handle = anim_graphs.add(graph);

            let mut player = AnimationPlayer::default();
            player.play(node).repeat();

            commands.entity(player_entity).insert((
                player,
                AnimationGraphHandle(graph_handle),
                ProxyArmature,
            ));
            info!("Proxy skeleton: AnimationPlayer configured ({:?})", player_entity);
        }
    }
}

fn find_existing_animation_player(
    root: Entity,
    names: &Query<(Entity, &Name)>,
    children: &Query<&Children>,
    players: &Query<Entity, With<AnimationPlayer>>,
) -> Option<Entity> {
    if players.get(root).is_ok() {
        return Some(root);
    }
    if let Ok(ch) = children.get(root) {
        for child in ch.iter() {
            if let Some(found) = find_existing_animation_player(child, names, children, players) {
                return Some(found);
            }
        }
    }
    None
}

// ── T-pose capture from InverseBindMatrices ──────────────────────────────────

/// プロキシの SkinnedMesh InverseBindMatrices から各ボーンのTポーズGT回転を取得し、
/// MANUKA側のTポーズGT（起動時GlobalTransform）と合わせて TposeReference を構築する。
///
/// IBM⁻¹ = GT_bind_pose（ワールド空間）なので、Blenderのボーンオリエンテーションを
/// 正確に含んだTポーズ参照が得られる。
fn capture_tpose_from_ibm(
    mut commands: Commands,
    existing: Option<Res<TposeReference>>,
    proxy_q: Query<Entity, With<ProxyRoot>>,
    mascot_q: Query<Entity, With<MascotTag>>,
    proxy_armature_q: Query<(), With<ProxyArmature>>,
    skinned_meshes: Query<&SkinnedMesh>,
    ibm_assets: Res<Assets<SkinnedMeshInverseBindposes>>,
    names: Query<(Entity, &Name)>,
    children: Query<&Children>,
    global_transforms: Query<&GlobalTransform>,
) {
    if existing.is_some() { return; }
    if proxy_armature_q.is_empty() { return; }

    let Ok(proxy_root) = proxy_q.single() else { return };
    let Ok(manuka_root) = mascot_q.single() else { return };

    // プロキシ階層内のSkinnedMeshを探す
    let Some(sm_entity) = find_skinned_mesh_in_subtree(proxy_root, &skinned_meshes, &children)
        else {
            // IBMがまだロードされていない可能性があるので次フレームに再試行
            return;
        };

    let Ok(skinned_mesh) = skinned_meshes.get(sm_entity) else { return };
    let Some(ibm) = ibm_assets.get(&skinned_mesh.inverse_bindposes) else {
        return; // IBMアセットがまだロードされていない
    };

    // IBM⁻¹ = GT_tpose_world → 回転成分を抽出
    let mut proxy_tpose: HashMap<String, Quat> = HashMap::new();
    for (joint_entity, ibm_mat) in skinned_mesh.joints.iter().zip(ibm.iter()) {
        if let Ok((_, name)) = names.get(*joint_entity) {
            if RETARGET_BONES.contains(&name.as_str()) {
                let (_, rot, _) = ibm_mat.inverse().to_scale_rotation_translation();
                proxy_tpose.insert(name.to_string(), rot.normalize());
            }
        }
    }

    if proxy_tpose.is_empty() {
        return; // ボーンが見つからなかった（まだ階層が構築されていない）
    }

    // MANUKAのTポーズGT（起動時は常にTポーズ）
    let mut manuka_tpose: HashMap<String, Quat> = HashMap::new();
    for &bone_name in RETARGET_BONES {
        if let Some(entity) = find_named_child_any(manuka_root, bone_name, &names, &children) {
            if let Ok(gt) = global_transforms.get(entity) {
                manuka_tpose.insert(bone_name.to_string(), gt.rotation());
            }
        }
    }

    info!("T-pose reference captured from IBM ({} proxy bones, {} manuka bones):",
        proxy_tpose.len(), manuka_tpose.len());
    for bone in ["Hips", "UpperArm.L", "UpperLeg.L"] {
        let p = proxy_tpose.get(bone);
        let m = manuka_tpose.get(bone);
        info!("  [{bone}] proxy_IBM_tpose={p:.4?}, manuka_tpose={m:.4?}");
    }

    commands.insert_resource(TposeReference {
        proxy: proxy_tpose,
        manuka: manuka_tpose,
    });
}

fn find_skinned_mesh_in_subtree(
    root: Entity,
    skinned_meshes: &Query<&SkinnedMesh>,
    children: &Query<&Children>,
) -> Option<Entity> {
    if skinned_meshes.get(root).is_ok() {
        return Some(root);
    }
    if let Ok(ch) = children.get(root) {
        for child in ch.iter() {
            if let Some(found) = find_skinned_mesh_in_subtree(child, skinned_meshes, children) {
                return Some(found);
            }
        }
    }
    None
}

// ── Transform copy: proxy → MANUKA ───────────────────────────────────────────

/// GlobalTransform デルタ方式でプロキシ→MANUKAにリターゲット。
///
/// 各ボーンについて:
///   delta = GT_proxy_anim × GT_proxy_tpose⁻¹  (TポーズからのWT空間変化量)
///   GT_manuka_desired = delta × GT_manuka_tpose (MANUKAのTポーズに適用)
///   local_manuka = parent_GT⁻¹ × GT_manuka_desired
///
/// プロキシのTポーズはIBM⁻¹から算出するため、腕・脚の絶対ポーズも正しく転送される。
fn copy_proxy_to_manuka(
    rest: Option<Res<TposeReference>>,
    proxy_q: Query<Entity, With<ProxyRoot>>,
    mascot_q: Query<Entity, With<MascotTag>>,
    proxy_armature_q: Query<(), With<ProxyArmature>>,
    names: Query<(Entity, &Name)>,
    children: Query<&Children>,
    global_transforms: Query<&GlobalTransform>,
    parents: Query<&ChildOf>,
    mut transforms: Query<&mut Transform>,
    mut frame_count: Local<u32>,
    mut upright_correction: Local<Option<Quat>>,
) {
    let Some(rest) = rest else { return };
    if proxy_armature_q.is_empty() { return; }

    let Ok(proxy_root) = proxy_q.single() else { return };
    let Ok(manuka_root) = mascot_q.single() else { return };

    // プロキシの現在の GlobalTransform 回転を収集
    let mut proxy_gt_anim: HashMap<String, Quat> = HashMap::new();
    for &bone_name in RETARGET_BONES {
        if let Some(entity) = find_named_child_any(proxy_root, bone_name, &names, &children) {
            if let Ok(gt) = global_transforms.get(entity) {
                proxy_gt_anim.insert(bone_name.to_string(), gt.rotation());
            }
        }
    }

    if proxy_gt_anim.is_empty() { return; }

    // MANUKA のボーンエンティティを収集
    let mut manuka_bones: HashMap<String, Entity> = HashMap::new();
    for &bone_name in RETARGET_BONES {
        if let Some(entity) = find_named_child_any(manuka_root, bone_name, &names, &children) {
            manuka_bones.insert(bone_name.to_string(), entity);
        }
    }

    // 親→子の順に処理（RETARGET_BONES は既にその順序）
    // 処理済みボーンの computed GlobalTransform を追跡
    let mut computed_gt: HashMap<String, Quat> = HashMap::new();

    *frame_count += 1;
    let debug = matches!(*frame_count, 1 | 10 | 30 | 120);

    for &bone_name in RETARGET_BONES {
        let Some(&gt_proxy_anim) = proxy_gt_anim.get(bone_name) else { continue };
        let Some(&gt_proxy_rest) = rest.proxy.get(bone_name) else { continue };
        let Some(&gt_manuka_rest) = rest.manuka.get(bone_name) else { continue };
        let Some(&manuka_entity) = manuka_bones.get(bone_name) else { continue };

        // global_delta = GT_proxy_anim * GT_proxy_rest⁻¹
        let mut global_delta = gt_proxy_anim * gt_proxy_rest.inverse();

        // 初回の Hips から「立ち姿勢 + 正面向き」補正を推定。
        if bone_name == "Hips" && upright_correction.is_none() {
            let correction = compute_hips_upright_correction(global_delta, gt_manuka_rest);
            *upright_correction = Some(correction);
            info!("Retarget: upright correction initialized from Hips");
        }

        if let Some(correction) = *upright_correction {
            global_delta = correction * global_delta;
        }

        // GT_manuka_desired = global_delta * GT_manuka_rest
        let gt_desired = global_delta * gt_manuka_rest;

        // 親の GlobalTransform 回転を取得（computed があればそちらを使う）
        let parent_gt_rot = get_parent_computed_gt(
            manuka_entity, &parents, &names, &computed_gt, &global_transforms,
        );

        // local = parent_GT⁻¹ * desired_GT
        let local_rot = parent_gt_rot.inverse() * gt_desired;

        if debug && ["Hips", "Chest", "UpperLeg.L", "Shoulder.L", "UpperArm.L"].contains(&bone_name) {
            let (dx, dy, dz) = global_delta.to_euler(EulerRot::XYZ);
            let (lx, ly, lz) = local_rot.to_euler(EulerRot::XYZ);
            info!(
                "Frame {} [{bone_name}]: delta_euler≈({:.1}°, {:.1}°, {:.1}°), local≈({:.1}°, {:.1}°, {:.1}°)",
                *frame_count,
                dx.to_degrees(), dy.to_degrees(), dz.to_degrees(),
                lx.to_degrees(), ly.to_degrees(), lz.to_degrees(),
            );
        }

        // Transform に反映
        if let Ok(mut tf) = transforms.get_mut(manuka_entity) {
            tf.rotation = local_rot;
        }

        // computed GT を記録
        computed_gt.insert(bone_name.to_string(), gt_desired);
    }
}

/// Hips の初期姿勢を VRM の休止姿勢に合わせる補正回転を求める。
///
/// 1) まず「上方向」を一致させる（寝転び補正）
/// 2) 次に上方向軸まわりのねじれを合わせる（正面合わせ）
fn compute_hips_upright_correction(global_delta: Quat, gt_manuka_rest: Quat) -> Quat {
    let current_hips = global_delta * gt_manuka_rest;

    let rest_up = (gt_manuka_rest * Vec3::Y).normalize_or_zero();
    let current_up = (current_hips * Vec3::Y).normalize_or_zero();
    let align_up = if rest_up.length_squared() > 0.0 && current_up.length_squared() > 0.0 {
        Quat::from_rotation_arc(current_up, rest_up)
    } else {
        Quat::IDENTITY
    };

    let corrected = align_up * current_hips;
    let rest_forward = project_on_plane(gt_manuka_rest * Vec3::Z, rest_up).normalize_or_zero();
    let corrected_forward = project_on_plane(corrected * Vec3::Z, rest_up).normalize_or_zero();

    let align_forward = if rest_forward.length_squared() > 0.0
        && corrected_forward.length_squared() > 0.0
    {
        Quat::from_rotation_arc(corrected_forward, rest_forward)
    } else {
        Quat::IDENTITY
    };

    align_forward * align_up
}

fn project_on_plane(v: Vec3, normal: Vec3) -> Vec3 {
    v - normal * v.dot(normal)
}

/// MANUKA ボーンの親の GlobalTransform 回転を取得。
/// 既に computed_gt にある親ボーンがあればそちらを使い、
/// なければ Bevy の GlobalTransform を使う。
fn get_parent_computed_gt(
    entity: Entity,
    parents: &Query<&ChildOf>,
    names: &Query<(Entity, &Name)>,
    computed_gt: &HashMap<String, Quat>,
    global_transforms: &Query<&GlobalTransform>,
) -> Quat {
    if let Ok(child_of) = parents.get(entity) {
        let parent = child_of.parent();
        // 親がリターゲット済みボーンならそちらを使う
        if let Ok((_, name)) = names.get(parent) {
            if let Some(&rot) = computed_gt.get(name.as_str()) {
                return rot;
            }
        }
        // それ以外は Bevy の GlobalTransform
        if let Ok(gt) = global_transforms.get(parent) {
            return gt.rotation();
        }
    }
    Quat::IDENTITY
}

fn find_named_child(
    root: Entity,
    target_name: &str,
    names: &Query<(Entity, &Name)>,
    children: &Query<&Children>,
) -> Option<Entity> {
    if let Ok((_, name)) = names.get(root) {
        if name.as_str() == target_name {
            return Some(root);
        }
    }
    if let Ok(ch) = children.get(root) {
        for child in ch.iter() {
            if let Some(found) = find_named_child(child, target_name, names, children) {
                return Some(found);
            }
        }
    }
    None
}

fn bone_name_aliases(name: &str) -> [&str; 3] {
    match name {
        "Chest" => ["Chest", "UpperChest", "Chest"],
        "UpperChest" => ["UpperChest", "Chest", "UpperChest"],
        "Toe.L" => ["Toe.L", "Toes_L", "Toe_L"],
        "Toe.R" => ["Toe.R", "Toes_R", "Toe_R"],
        "Shoulder.L" => ["Shoulder.L", "Shoulder_L", "Shoulder.L"],
        "UpperArm.L" => ["UpperArm.L", "UpperArm_L", "UpperArm.L"],
        "LowerArm.L" => ["LowerArm.L", "LowerArm_L", "LowerArm.L"],
        "Hand.L" => ["Hand.L", "Hand_L", "Hand.L"],
        "UpperLeg.L" => ["UpperLeg.L", "UpperLeg_L", "UpperLeg.L"],
        "LowerLeg.L" => ["LowerLeg.L", "LowerLeg_L", "LowerLeg.L"],
        "Foot.L" => ["Foot.L", "Foot_L", "Foot.L"],
        "Shoulder.R" => ["Shoulder.R", "Shoulder_R", "Shoulder.R"],
        "UpperArm.R" => ["UpperArm.R", "UpperArm_R", "UpperArm.R"],
        "LowerArm.R" => ["LowerArm.R", "LowerArm_R", "LowerArm.R"],
        "Hand.R" => ["Hand.R", "Hand_R", "Hand.R"],
        "UpperLeg.R" => ["UpperLeg.R", "UpperLeg_R", "UpperLeg.R"],
        "LowerLeg.R" => ["LowerLeg.R", "LowerLeg_R", "LowerLeg.R"],
        "Foot.R" => ["Foot.R", "Foot_R", "Foot.R"],
        _ => [name, name, name],
    }
}

fn find_named_child_any(
    root: Entity,
    canonical_name: &str,
    names: &Query<(Entity, &Name)>,
    children: &Query<&Children>,
) -> Option<Entity> {
    for alias in bone_name_aliases(canonical_name) {
        if let Some(found) = find_named_child(root, alias, names, children) {
            return Some(found);
        }
    }
    None
}
