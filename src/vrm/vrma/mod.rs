pub mod loader;
pub mod player;
pub mod retarget;

use bevy::prelude::*;
use bevy::gltf::Gltf;

use crate::mascot::MascotTag;
use crate::vrm::vrma::player::{VrmaPlayback, VrmaState, parse_vrma_retarget};

pub struct VrmaPlugin;

impl Plugin for VrmaPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(loader::VrmaLoaderPlugin)
            .insert_resource(VrmaPlayback::default())
            .add_systems(Update, process_vrma_requests);
    }
}

/// Process pending VRMA playback requests.
/// When the Gltf asset is ready, parse the humanoid map and play animations.
fn process_vrma_requests(
    mut playback: ResMut<VrmaPlayback>,
    gltf_assets: Res<Assets<Gltf>>,
    mascot_q: Query<(Entity, Option<&VrmaState>), With<MascotTag>>,
    names: Query<(Entity, &Name)>,
    children: Query<&Children>,
    mut commands: Commands,
    mut animation_players: Query<&mut AnimationPlayer>,
) {
    playback.requests.retain(|req| {
        let Some(gltf) = gltf_assets.get(&req.handle) else {
            return true; // still loading, keep request
        };

        let Ok((mascot_entity, _)) = mascot_q.single() else {
            return false;
        };

        // Parse retarget map: vrma_node_name → vrm0_bone_name
        let Some(retarget_map) = parse_vrma_retarget(gltf) else {
            warn!("VRMA: failed to parse humanoid extension");
            return false;
        };

        // Build bone_map: vrm0_bone_name → entity
        let mut bone_map = std::collections::HashMap::new();
        collect_entities_recursive(mascot_entity, &names, &children, &mut bone_map);

        info!(
            "VRMA: retargeting {} bones onto VRM model",
            retarget_map.len()
        );

        // Play the first animation clip via AnimationGraph
        if let Some(clip_handle) = gltf.animations.first() {
            let mut graph = AnimationGraph::new();
            let node = graph.add_clip(
                clip_handle.clone(),
                1.0,
                graph.root,
            );
            if let Ok(mut player) = animation_players.get_mut(mascot_entity) {
                player.play(node).set_repeat(
                    if req.looping {
                        bevy::animation::RepeatAnimation::Forever
                    } else {
                        bevy::animation::RepeatAnimation::Never
                    },
                );
                info!("VRMA: playing animation");
            }
        }

        commands.entity(mascot_entity).insert(VrmaState { bone_map });
        false // request fulfilled, remove
    });
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
