pub mod bones;
pub mod procedural;

use bevy::gltf::Gltf;
use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

/// リターゲット対象ボーン。
///
/// Mixamo と VRM 0.x ではボーンローカル軸の向きが異なるため、
/// 腕・脚の直接リターゲットは正確な軸補正が必要（TODO: #5 proper retargeting）。
/// 現時点では体幹・頭のみリターゲットし、腕・脚は VRM バインドポーズを維持する。
/// これで足のクロスや腕の開きを回避しつつ呼吸・頭揺れのモーションを得る。
const RETARGET_BONES: &[&str] = &[
    "Hips",
    "Spine",
    "Chest",
    "UpperChest",
    "Neck",
    "Head",
    // 肩は肩甲骨の動きのみ（腕より軸ズレが小さい）
    "Shoulder_L",
    "Shoulder_R",
];

use crate::events::{MascotPhase, PipelineMessage};
use crate::mascot::MascotTag;
use bones::{collect_bones, HumanoidBones};
use procedural::{
    breathing_rotation, breathing_scale_for_phase, head_sway, BreathingParams, HeadSwayParams,
};

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
                    copy_proxy_to_manuka,
                    debug_dump_bone_names,
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

/// プロキシのセットアップ完了フラグ。
#[derive(Component)]
struct ProxyReady;

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
    mut cached_handle: Local<Option<Handle<Gltf>>>,
) {
    // MANUKAのボーンが揃うまで待機
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
    if !proxy_armature_q.is_empty() { return; } // already done

    let Ok(proxy_root) = proxy_q.single() else { return };

    // ProxyRootの子孫から "Armature" を探す
    if let Some(armature_entity) = find_named_child(proxy_root, "Armature", &names, &children) {
        if let Some(clip) = gltf.animations.first() {
            let mut graph = AnimationGraph::new();
            let node = graph.add_clip(clip.clone(), 1.0, graph.root);
            let graph_handle = anim_graphs.add(graph);

            let mut player = AnimationPlayer::default();
            player.play(node).repeat();

            commands.entity(armature_entity).insert((
                player,
                AnimationGraphHandle(graph_handle),
                ProxyArmature,
            ));
            info!("Proxy skeleton: AnimationPlayer started on Armature ({:?})", armature_entity);
        }
    }
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

// ── Transform copy: proxy → MANUKA ───────────────────────────────────────────

/// レストポーズ（初フレームでキャプチャ）。
struct RestPoses {
    proxy: HashMap<String, Quat>,
    manuka: HashMap<String, Quat>,
}

/// プロキシ→MANUKAのリターゲティングコピー。
///
/// 正しい式: delta = proxy_rest⁻¹ × proxy_current
///           manuka_out = manuka_rest × delta
///
/// 初フレームで両骨格のレストポーズを記録し、以降はデルタだけを転送する。
fn copy_proxy_to_manuka(
    proxy_q: Query<Entity, With<ProxyRoot>>,
    mascot_q: Query<Entity, With<MascotTag>>,
    proxy_armature_q: Query<(), With<ProxyArmature>>,
    names: Query<(Entity, &Name)>,
    children: Query<&Children>,
    mut tf_set: ParamSet<(
        Query<&Transform>,      // p0: 読み取り
        Query<&mut Transform>,  // p1: 書き込み
    )>,
    mut rest_poses: Local<Option<RestPoses>>,
) {
    // アニメーションが開始してから実行
    if proxy_armature_q.is_empty() { return; }

    let Ok(proxy_root) = proxy_q.single() else { return };
    let Ok(manuka_root) = mascot_q.single() else { return };

    let whitelist: HashSet<&str> = RETARGET_BONES.iter().copied().collect();

    // 現フレームのプロキシとMANUKAの回転を収集
    let mut proxy_cur: HashMap<String, Quat> = HashMap::new();
    let mut manuka_cur: HashMap<String, Quat> = HashMap::new();
    {
        let tf_ro = tf_set.p0();
        collect_bone_rotations(proxy_root, &names, &children, &tf_ro, &whitelist, &mut proxy_cur);
        collect_bone_rotations(manuka_root, &names, &children, &tf_ro, &whitelist, &mut manuka_cur);
    }

    if proxy_cur.is_empty() { return; }

    // 初フレームでレストポーズをキャプチャ
    let rest = rest_poses.get_or_insert_with(|| {
        info!("Retarget: rest poses captured ({} bones)", proxy_cur.len());
        RestPoses {
            proxy: proxy_cur.clone(),
            manuka: manuka_cur.clone(),
        }
    });

    // リターゲット計算: delta = proxy_rest⁻¹ × proxy_now
    // Blender側でA-pose→T-poseに変換済みなのでシンプルに適用
    let mut retargeted: HashMap<String, Quat> = HashMap::new();
    for name in RETARGET_BONES {
        let name = *name;
        if let (Some(&proxy_rest), Some(&proxy_now), Some(&manuka_rest)) = (
            rest.proxy.get(name),
            proxy_cur.get(name),
            rest.manuka.get(name),
        ) {
            let delta = proxy_rest.inverse() * proxy_now;
            retargeted.insert(name.to_string(), manuka_rest * delta);
        }
    }

    // MANUKAに適用
    {
        let mut tf_rw = tf_set.p1();
        apply_bone_rotations(manuka_root, &names, &children, &retargeted, &mut tf_rw);
    }
}

fn collect_bone_rotations(
    entity: Entity,
    names: &Query<(Entity, &Name)>,
    children: &Query<&Children>,
    transforms: &Query<&Transform>,
    whitelist: &HashSet<&str>,
    out: &mut HashMap<String, Quat>,
) {
    if let Ok((_, name)) = names.get(entity) {
        if whitelist.contains(name.as_str()) {
            if let Ok(tf) = transforms.get(entity) {
                out.insert(name.as_str().to_string(), tf.rotation);
            }
        }
    }
    if let Ok(ch) = children.get(entity) {
        for child in ch.iter() {
            collect_bone_rotations(child, names, children, transforms, whitelist, out);
        }
    }
}

fn apply_bone_rotations(
    entity: Entity,
    names: &Query<(Entity, &Name)>,
    children: &Query<&Children>,
    proxy_map: &HashMap<String, Quat>,
    transforms: &mut Query<&mut Transform>,
) {
    if let Ok((_, name)) = names.get(entity) {
        if let Some(&rot) = proxy_map.get(name.as_str()) {
            if let Ok(mut tf) = transforms.get_mut(entity) {
                tf.rotation = rot;
            }
        }
    }
    if let Ok(ch) = children.get(entity) {
        for child in ch.iter() {
            apply_bone_rotations(child, names, children, proxy_map, transforms);
        }
    }
}

// ── Debug ─────────────────────────────────────────────────────────────────────

fn debug_dump_bone_names(
    mascot_q: Query<Entity, With<MascotTag>>,
    names: Query<(Entity, &Name)>,
    children: Query<&Children>,
    mut done: Local<bool>,
) {
    if *done { return; }
    let Ok(root) = mascot_q.single() else { return };

    let mut all_names: Vec<String> = Vec::new();
    collect_names(root, &names, &children, &mut all_names);
    if all_names.is_empty() { return; }

    info!("=== VRM bone names ({} entities) ===", all_names.len());
    for name in &all_names {
        info!("  {}", name);
    }
    info!("=== end ===");
    *done = true;
}

fn collect_names(
    entity: Entity,
    names: &Query<(Entity, &Name)>,
    children: &Query<&Children>,
    out: &mut Vec<String>,
) {
    if let Ok((_, name)) = names.get(entity) {
        out.push(name.as_str().to_string());
    }
    if let Ok(ch) = children.get(entity) {
        for child in ch.iter() {
            collect_names(child, names, children, out);
        }
    }
}
