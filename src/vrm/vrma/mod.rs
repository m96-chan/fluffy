pub mod loader;
pub mod player;
pub mod retarget;

use bevy::prelude::*;
use bevy::gltf::Gltf;
use std::collections::{HashMap, HashSet};

use crate::mascot::MascotTag;
use crate::vrm::VrmGltfHandle;
use crate::vrm::vrma::player::{VrmaPlayback, VrmaRequest, VrmaState};

pub struct VrmaPlugin;

impl Plugin for VrmaPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(loader::VrmaLoaderPlugin)
            .insert_resource(VrmaPlayback::default())
            .add_systems(
                Update,
                (
                    attach_animation_targets_to_mascot,
                    queue_idle_vrma_request,
                    process_vrma_requests,
                )
                    .chain(),
            );
    }
}

#[derive(Component)]
struct IdleVrmaQueued;
#[derive(Component)]
struct AnimationTargetsReady;

fn attach_animation_targets_to_mascot(
    mut commands: Commands,
    mascot_q: Query<Entity, (With<MascotTag>, Without<AnimationTargetsReady>)>,
    children_q: Query<&Children>,
    names_q: Query<&Name>,
) {
    let Ok(mascot_entity) = mascot_q.single() else {
        return;
    };
    let Ok(root_children) = children_q.get(mascot_entity) else {
        // scene not spawned yet
        return;
    };

    let mut assigned = 0usize;
    for child in root_children.iter() {
        let mut path = Vec::<String>::new();
        assign_targets_recursive(
            child,
            mascot_entity,
            &mut path,
            &children_q,
            &names_q,
            &mut commands,
            &mut assigned,
        );
    }

    if assigned > 0 {
        commands.entity(mascot_entity).insert(AnimationTargetsReady);
        info!("VRMA: attached {assigned} animation targets on mascot hierarchy");
    }
}

/// Automatically queue `assets/anims/idle.vrma` once for the mascot.
fn queue_idle_vrma_request(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut playback: ResMut<VrmaPlayback>,
    mascot_q: Query<(Entity, Option<&IdleVrmaQueued>), With<MascotTag>>,
) {
    let Ok((mascot_entity, queued)) = mascot_q.single() else {
        return;
    };
    if queued.is_some() {
        return;
    }
    if !std::path::Path::new("assets/anims/idle.vrma").exists() {
        warn!("VRMA: assets/anims/idle.vrma not found; skip auto-play request");
        commands.entity(mascot_entity).insert(IdleVrmaQueued);
        return;
    }

    let handle: Handle<Gltf> = asset_server.load_with_settings(
        "anims/idle.vrma",
        |s: &mut bevy::gltf::GltfLoaderSettings| {
            s.include_source = true;
        },
    );
    playback.requests.push(VrmaRequest {
        handle,
        looping: true,
    });
    commands.entity(mascot_entity).insert(IdleVrmaQueued);
    info!("VRMA: queued auto-play for anims/idle.vrma");
}

/// Process pending VRMA playback requests.
/// When the Gltf asset is ready, parse the humanoid map and play animations.
fn process_vrma_requests(
    mut playback: ResMut<VrmaPlayback>,
    gltf_assets: Res<Assets<Gltf>>,
    mut animation_clips: ResMut<Assets<AnimationClip>>,
    mascot_q: Query<
        (
            Entity,
            Option<&VrmaState>,
            &VrmGltfHandle,
            Option<&AnimationTargetsReady>,
        ),
        With<MascotTag>,
    >,
    names: Query<(Entity, &Name)>,
    children: Query<&Children>,
    mut commands: Commands,
    mut anim_graphs: ResMut<Assets<AnimationGraph>>,
    mut animation_players: Query<&mut AnimationPlayer>,
    mut animation_target_bindings: Query<(
        Entity,
        &bevy::animation::AnimationTargetId,
        Option<&mut bevy::animation::AnimatedBy>,
    )>,
) {
    playback.requests.retain(|req| {
        let Some(gltf) = gltf_assets.get(&req.handle) else {
            return true; // still loading, keep request
        };

        let Ok((mascot_entity, _, mascot_gltf_handle, targets_ready)) = mascot_q.single() else {
            return false;
        };
        if targets_ready.is_none() {
            info!("VRMA: animation targets not ready yet; retrying next frame");
            return true;
        }
        let Some(mascot_gltf) = gltf_assets.get(&mascot_gltf_handle.0) else {
            return true; // mascot still loading
        };

        // Build bone_map: vrm0_bone_name → entity
        let mut bone_map = HashMap::new();
        collect_entities_recursive(mascot_entity, &names, &children, &mut bone_map);

        let vrma_target_maps = build_target_maps(gltf);
        let model_target_maps = build_target_maps(mascot_gltf);
        let vrma_humanoid = parse_vrma_humanoid_bone_node_indices(gltf);
        let model_humanoid = parse_vrm1_humanoid_bone_node_indices(mascot_gltf);

        // Play the first animation clip via AnimationGraph
        if let Some(clip_handle) = gltf.animations.first() {
            let Some(source_clip) = animation_clips.get(clip_handle).cloned() else {
                return true; // clip still loading
            };

            // Remap VRMA clip target IDs to this mascot's humanoid targets.
            let clip_to_play = if let (
                Some((_, vrma_id_to_node)),
                Some((model_node_to_id, _)),
                Some(vrma_bone_to_node),
                Some(model_bone_to_node),
            ) = (&vrma_target_maps, &model_target_maps, &vrma_humanoid, &model_humanoid)
            {
                let (remapped, matched, total) = remap_clip_targets(
                    &source_clip,
                    vrma_id_to_node,
                    vrma_bone_to_node,
                    model_bone_to_node,
                    model_node_to_id,
                );
                info!("VRMA: remapped animation targets {matched}/{total}");
                let rebound = rebind_targets_for_clip(
                    mascot_entity,
                    &remapped,
                    &mut animation_target_bindings,
                    &mut commands,
                );
                info!("VRMA: rebound {rebound} animation targets to mascot player");
                animation_clips.add(remapped)
            } else {
                warn!("VRMA: humanoid/node maps unavailable; falling back to original clip");
                clip_handle.clone()
            };

            let mut graph = AnimationGraph::new();
            let node = graph.add_clip(
                clip_to_play,
                1.0,
                graph.root,
            );
            let graph_handle = anim_graphs.add(graph);

            if let Ok(mut player) = animation_players.get_mut(mascot_entity) {
                player.play(node).set_repeat(
                    if req.looping {
                        bevy::animation::RepeatAnimation::Forever
                    } else {
                        bevy::animation::RepeatAnimation::Never
                    },
                );
                commands
                    .entity(mascot_entity)
                    .insert(AnimationGraphHandle(graph_handle));
                info!("VRMA: playing animation");
            } else {
                let mut player = AnimationPlayer::default();
                player.play(node).set_repeat(
                    if req.looping {
                        bevy::animation::RepeatAnimation::Forever
                    } else {
                        bevy::animation::RepeatAnimation::Never
                    },
                );
                commands.entity(mascot_entity).insert((
                    player,
                    AnimationGraphHandle(graph_handle),
                ));
                info!("VRMA: created AnimationPlayer and started playback");
            }
        }

        commands.entity(mascot_entity).insert(VrmaState { bone_map });
        false // request fulfilled, remove
    });
}

fn build_target_maps(
    gltf: &Gltf,
) -> Option<(
    HashMap<usize, bevy::animation::AnimationTargetId>,
    HashMap<bevy::animation::AnimationTargetId, usize>,
)> {
    let doc = gltf.source.as_ref()?;
    let mut node_to_id = HashMap::new();
    let mut id_to_node = HashMap::new();
    for scene in doc.scenes() {
        for node in scene.nodes() {
            let mut path = Vec::<String>::new();
            let mut visited = HashSet::<usize>::new();
            collect_target_paths(node, &mut path, &mut visited, &mut node_to_id, &mut id_to_node);
        }
    }
    Some((node_to_id, id_to_node))
}

fn collect_target_paths(
    node: gltf::Node<'_>,
    path: &mut Vec<String>,
    visited: &mut HashSet<usize>,
    node_to_id: &mut HashMap<usize, bevy::animation::AnimationTargetId>,
    id_to_node: &mut HashMap<bevy::animation::AnimationTargetId, usize>,
) {
    if visited.contains(&node.index()) {
        return;
    }
    visited.insert(node.index());

    let node_name = node
        .name()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("GltfNode{}", node.index()));
    path.push(node_name.clone());

    let target_id = bevy::animation::AnimationTargetId::from_iter(path.iter());
    node_to_id.entry(node.index()).or_insert(target_id);
    id_to_node.entry(target_id).or_insert(node.index());

    for child in node.children() {
        collect_target_paths(child, path, visited, node_to_id, id_to_node);
    }

    path.pop();
    visited.remove(&node.index());
}

fn remap_clip_targets(
    source: &AnimationClip,
    vrma_id_to_node: &HashMap<bevy::animation::AnimationTargetId, usize>,
    vrma_bone_to_node: &HashMap<String, usize>,
    model_bone_to_node: &HashMap<String, usize>,
    model_node_to_id: &HashMap<usize, bevy::animation::AnimationTargetId>,
) -> (AnimationClip, usize, usize) {
    let mut clip = source.clone();
    let mut remapped = HashMap::new();
    let mut matched = 0usize;
    let total = source.curves().len();
    let vrma_node_to_bone: HashMap<usize, &str> = vrma_bone_to_node
        .iter()
        .map(|(bone, node)| (*node, bone.as_str()))
        .collect();

    for (src_id, curves) in source.curves() {
        let Some(src_node_idx) = vrma_id_to_node.get(src_id) else {
            continue;
        };
        let Some(bone_key) = vrma_node_to_bone.get(src_node_idx) else {
            continue;
        };
        let Some(dst_node_idx) = model_bone_to_node.get(*bone_key) else {
            continue;
        };
        let Some(dst_id) = model_node_to_id.get(dst_node_idx).copied() else {
            continue;
        };

        remapped
            .entry(dst_id)
            .or_insert_with(Vec::new)
            .extend(curves.clone());
        matched += 1;
    }

    clip.curves_mut().clear();
    for (target_id, curves) in remapped {
        clip.curves_mut().insert(target_id, curves);
    }
    (clip, matched, total)
}

fn parse_vrma_humanoid_bone_node_indices(gltf: &Gltf) -> Option<HashMap<String, usize>> {
    let doc = gltf.source.as_ref()?;
    let raw: serde_json::Value = serde_json::to_value(doc.as_json()).ok()?;
    let human_bones = raw
        .get("extensions")?
        .get("VRMC_vrm_animation")?
        .get("humanoid")?
        .get("humanBones")?
        .as_object()?;

    let mut out = HashMap::new();
    for (bone_key, val) in human_bones {
        if let Some(idx) = val.get("node").and_then(|v| v.as_u64()) {
            out.insert(bone_key.clone(), idx as usize);
        }
    }
    Some(out)
}

fn parse_vrm1_humanoid_bone_node_indices(gltf: &Gltf) -> Option<HashMap<String, usize>> {
    let doc = gltf.source.as_ref()?;
    let raw: serde_json::Value = serde_json::to_value(doc.as_json()).ok()?;
    let human_bones = raw
        .get("extensions")?
        .get("VRMC_vrm")?
        .get("humanoid")?
        .get("humanBones")?
        .as_object()?;

    let mut out = HashMap::new();
    for (bone_key, val) in human_bones {
        if let Some(idx) = val.get("node").and_then(|v| v.as_u64()) {
            out.insert(bone_key.clone(), idx as usize);
        }
    }
    Some(out)
}

fn collect_entities_recursive(
    entity: Entity,
    names: &Query<(Entity, &Name)>,
    children: &Query<&Children>,
    map: &mut std::collections::HashMap<String, Entity>,
) {
    if let Ok((_, name)) = names.get(entity) {
        map.insert(name.as_str().to_string(), entity);
    }
    if let Ok(ch) = children.get(entity) {
        for child in ch.iter() {
            collect_entities_recursive(child, names, children, map);
        }
    }
}

fn assign_targets_recursive(
    entity: Entity,
    mascot_entity: Entity,
    path: &mut Vec<String>,
    children_q: &Query<&Children>,
    names_q: &Query<&Name>,
    commands: &mut Commands,
    assigned: &mut usize,
) {
    let mut pushed = false;
    if let Ok(name) = names_q.get(entity) {
        path.push(name.as_str().to_string());
        pushed = true;
        if !path.is_empty() {
            let target_id = bevy::animation::AnimationTargetId::from_iter(path.iter());
            commands.entity(entity).insert((
                target_id,
                bevy::animation::AnimatedBy(mascot_entity),
            ));
            *assigned += 1;
        }
    }

    if let Ok(children) = children_q.get(entity) {
        for child in children.iter() {
            assign_targets_recursive(
                child,
                mascot_entity,
                path,
                children_q,
                names_q,
                commands,
                assigned,
            );
        }
    }

    if pushed {
        path.pop();
    }
}

fn rebind_targets_for_clip(
    root: Entity,
    clip: &AnimationClip,
    animation_targets: &mut Query<
        (
            Entity,
            &bevy::animation::AnimationTargetId,
            Option<&mut bevy::animation::AnimatedBy>,
        ),
    >,
    commands: &mut Commands,
) -> usize {
    let wanted: HashSet<bevy::animation::AnimationTargetId> =
        clip.curves().keys().copied().collect();
    let mut count = 0usize;

    for (target_entity, target_id, animated_by) in animation_targets.iter_mut() {
        if !wanted.contains(target_id) {
            continue;
        }
        match animated_by {
            Some(mut by) => {
                if by.0 != root {
                    by.0 = root;
                }
            }
            None => {
                commands
                    .entity(target_entity)
                    .insert(bevy::animation::AnimatedBy(root));
            }
        }
        count += 1;
    }

    count
}
