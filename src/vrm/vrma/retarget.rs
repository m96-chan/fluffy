/// VRMA → VRM bone name retargeting.
///
/// VRMA uses VRM spec humanoid bone names (camelCase lowercase first):
///   "hips", "spine", "chest", "upperChest", "neck", "head",
///   "leftUpperArm", "rightUpperArm", ...
///
/// Spawned Bevy entities from VRM 0.x (Unity/VRoid export) use PascalCase:
///   "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
///   "LeftUpperArm", "RightUpperArm", ...
///
/// This module converts between the two.

use std::collections::HashMap;

/// Convert a VRMA humanoid bone name to the expected node name in a VRM 0.x scene.
/// VRM 0.x node names match Unity's HumanBodyBones naming (PascalCase).
pub fn vrma_to_vrm0_bone_name(vrma_name: &str) -> String {
    // Most names just need the first letter uppercased.
    // "leftUpperArm" → "LeftUpperArm"  (first char upper, rest unchanged)
    let mut chars = vrma_name.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Build a map from VRMA node index → expected VRM 0.x bone node name.
/// `vrma_humanoid_bones`: from `VRMC_vrm_animation.humanoid.humanBones`
///   e.g. {"hips": {"node": 0}, "spine": {"node": 1}, ...}
/// `vrma_node_names`: from the GLTF nodes array (index → name)
pub fn build_retarget_map(
    vrma_humanoid_bones: &HashMap<String, usize>, // bone_name → node_index
    vrma_node_names: &[String],                   // node_index → node_name
) -> HashMap<String, String> {
    // vrma_node_name → vrm0_bone_name
    let mut map = HashMap::new();
    for (bone_name, &node_idx) in vrma_humanoid_bones {
        if let Some(node_name) = vrma_node_names.get(node_idx) {
            let vrm0_name = vrma_to_vrm0_bone_name(bone_name);
            map.insert(node_name.clone(), vrm0_name);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── vrma_to_vrm0_bone_name ───────────────────────────────────────────────

    #[test]
    fn hips_uppercased() {
        assert_eq!(vrma_to_vrm0_bone_name("hips"), "Hips");
    }

    #[test]
    fn spine_uppercased() {
        assert_eq!(vrma_to_vrm0_bone_name("spine"), "Spine");
    }

    #[test]
    fn upper_chest_stays_camel() {
        // "upperChest" → "UpperChest"
        assert_eq!(vrma_to_vrm0_bone_name("upperChest"), "UpperChest");
    }

    #[test]
    fn left_upper_arm() {
        assert_eq!(vrma_to_vrm0_bone_name("leftUpperArm"), "LeftUpperArm");
    }

    #[test]
    fn right_lower_leg() {
        assert_eq!(vrma_to_vrm0_bone_name("rightLowerLeg"), "RightLowerLeg");
    }

    #[test]
    fn head_uppercased() {
        assert_eq!(vrma_to_vrm0_bone_name("head"), "Head");
    }

    #[test]
    fn empty_string_stays_empty() {
        assert_eq!(vrma_to_vrm0_bone_name(""), "");
    }

    #[test]
    fn already_pascal_case_unchanged() {
        // If someone passes PascalCase, first char stays upper
        assert_eq!(vrma_to_vrm0_bone_name("Hips"), "Hips");
    }

    // ── build_retarget_map ───────────────────────────────────────────────────

    #[test]
    fn retarget_map_basic() {
        let mut bones = HashMap::new();
        bones.insert("hips".to_string(), 0usize);
        bones.insert("spine".to_string(), 1usize);

        let node_names = vec!["J_Bip_C_Hips".to_string(), "J_Bip_C_Spine".to_string()];

        let map = build_retarget_map(&bones, &node_names);

        assert_eq!(map.get("J_Bip_C_Hips"), Some(&"Hips".to_string()));
        assert_eq!(map.get("J_Bip_C_Spine"), Some(&"Spine".to_string()));
    }

    #[test]
    fn retarget_map_skips_out_of_bounds_node() {
        let mut bones = HashMap::new();
        bones.insert("hips".to_string(), 99usize); // out of range

        let node_names = vec!["Node0".to_string()];
        let map = build_retarget_map(&bones, &node_names);

        assert!(map.is_empty());
    }

    #[test]
    fn retarget_map_empty_input() {
        let map = build_retarget_map(&HashMap::new(), &[]);
        assert!(map.is_empty());
    }

    #[test]
    fn all_standard_vrma_bones_convert() {
        // Spot-check common VRM humanoid bones
        let cases = [
            ("neck", "Neck"),
            ("head", "Head"),
            ("chest", "Chest"),
            ("upperChest", "UpperChest"),
            ("leftShoulder", "LeftShoulder"),
            ("rightShoulder", "RightShoulder"),
            ("leftUpperArm", "LeftUpperArm"),
            ("rightUpperArm", "RightUpperArm"),
            ("leftLowerArm", "LeftLowerArm"),
            ("rightHand", "RightHand"),
            ("leftUpperLeg", "LeftUpperLeg"),
            ("rightLowerLeg", "RightLowerLeg"),
            ("leftFoot", "LeftFoot"),
        ];
        for (vrma, expected) in cases {
            assert_eq!(
                vrma_to_vrm0_bone_name(vrma),
                expected,
                "failed for: {}",
                vrma
            );
        }
    }
}
