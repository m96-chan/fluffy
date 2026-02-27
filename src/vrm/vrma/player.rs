/// VRMA animation player.
/// Loads a .vrma file, extracts humanoid bone mapping, and plays it
/// on a VRM 0.x model by retargeting bone names.

use std::collections::HashMap;
use bevy::prelude::*;
use bevy::gltf::Gltf;

use crate::vrm::vrma::retarget::{build_retarget_map, vrma_to_vrm0_bone_name};

/// Resource: a pending VRMA playback request.
#[derive(Resource, Default)]
pub struct VrmaPlayback {
    pub requests: Vec<VrmaRequest>,
}

pub struct VrmaRequest {
    pub handle: Handle<Gltf>,
    pub looping: bool,
}

/// Component on the mascot entity: active VRMA animation state.
#[derive(Component, Default)]
pub struct VrmaState {
    /// humanoid bone name (VRM 0.x PascalCase) → entity
    pub bone_map: HashMap<String, Entity>,
}

/// Parse VRMA humanoid bone mapping from the GLTF source document.
/// Returns: VRMA node name → VRM 0.x bone name (PascalCase)
pub fn parse_vrma_retarget(gltf: &Gltf) -> Option<HashMap<String, String>> {
    let doc = gltf.source.as_ref()?;
    let raw: serde_json::Value = serde_json::to_value(doc.as_json()).ok()?;

    let anim_ext = raw
        .get("extensions")?
        .get("VRMC_vrm_animation")?;

    let human_bones = anim_ext
        .get("humanoid")?
        .get("humanBones")?
        .as_object()?;

    // Build bone_name → node_index map
    let mut bones: HashMap<String, usize> = HashMap::new();
    for (bone_name, val) in human_bones {
        if let Some(idx) = val.get("node").and_then(|v| v.as_u64()) {
            bones.insert(bone_name.clone(), idx as usize);
        }
    }

    // Build node_index → node_name from GLTF nodes
    let node_names: Vec<String> = doc
        .nodes()
        .map(|n| n.name().unwrap_or("").to_string())
        .collect();

    Some(build_retarget_map(&bones, &node_names))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vrma_retarget_map_from_json() {
        // Simulate what parse_vrma_retarget would build
        let mut bones = HashMap::new();
        bones.insert("hips".to_string(), 0usize);
        bones.insert("head".to_string(), 1usize);
        bones.insert("leftUpperArm".to_string(), 2usize);

        let node_names = vec![
            "root_hips".to_string(),
            "root_head".to_string(),
            "arm_L".to_string(),
        ];

        let map = build_retarget_map(&bones, &node_names);

        assert_eq!(map.get("root_hips").map(|s| s.as_str()), Some("Hips"));
        assert_eq!(map.get("root_head").map(|s| s.as_str()), Some("Head"));
        assert_eq!(map.get("arm_L").map(|s| s.as_str()), Some("LeftUpperArm"));
    }

    #[test]
    fn vrma_bone_name_conversion_spot_check() {
        assert_eq!(vrma_to_vrm0_bone_name("hips"), "Hips");
        assert_eq!(vrma_to_vrm0_bone_name("leftUpperArm"), "LeftUpperArm");
        assert_eq!(vrma_to_vrm0_bone_name("rightFoot"), "RightFoot");
    }

    #[test]
    fn vrma_request_looping_flag() {
        let req = VrmaRequest {
            handle: Handle::default(),
            looping: true,
        };
        assert!(req.looping);
    }

    #[test]
    fn vrma_state_default_empty() {
        let state = VrmaState::default();
        assert!(state.bone_map.is_empty());
    }
}
