/// Discover and cache humanoid bone entities from a spawned VRM/GLTF scene.

use bevy::prelude::*;
use std::collections::HashMap;

/// Standard VRM humanoid bone names we care about for animation.
/// These match the node names as they appear in the spawned Bevy scene.
pub const BONE_CHEST: &str = "Chest";
pub const BONE_UPPER_CHEST: &str = "UpperChest";
pub const BONE_SPINE: &str = "Spine";
pub const BONE_NECK: &str = "Neck";
pub const BONE_HEAD: &str = "Head";

/// Cached bone entity references, inserted as a component on the mascot root.
#[derive(Component, Debug, Default)]
pub struct HumanoidBones {
    /// entity name → entity id
    pub map: HashMap<String, Entity>,
}

impl HumanoidBones {
    pub fn get(&self, name: &str) -> Option<Entity> {
        self.map.get(name).copied()
    }

    /// Returns the chest bone (prefers UpperChest, falls back to Chest, then Spine).
    pub fn chest(&self) -> Option<Entity> {
        self.get(BONE_UPPER_CHEST)
            .or_else(|| self.get(BONE_CHEST))
            .or_else(|| self.get(BONE_SPINE))
    }

    pub fn head(&self) -> Option<Entity> {
        self.get(BONE_HEAD)
    }

    pub fn is_ready(&self) -> bool {
        self.chest().is_some() && self.head().is_some()
    }
}

/// Walk the scene tree and collect bone entities by name.
pub fn collect_bones(
    root: Entity,
    names: &Query<(Entity, &Name)>,
    children: &Query<&Children>,
) -> HumanoidBones {
    let mut bones = HumanoidBones::default();
    collect_recursive(root, names, children, &mut bones);
    bones
}

fn collect_recursive(
    entity: Entity,
    names: &Query<(Entity, &Name)>,
    children: &Query<&Children>,
    bones: &mut HumanoidBones,
) {
    if let Ok((_, name)) = names.get(entity) {
        let name_str = name.as_str().to_string();
        if matches!(
            name_str.as_str(),
            BONE_CHEST | BONE_UPPER_CHEST | BONE_SPINE | BONE_NECK | BONE_HEAD
        ) {
            bones.map.insert(name_str, entity);
        }
    }
    if let Ok(ch) = children.get(entity) {
        for child in ch.iter() {
            collect_recursive(child, names, children, bones);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(id: u32) -> Entity {
        Entity::from_raw_u32(id).expect("valid entity id")
    }

    fn make_bones(entries: &[(&str, u32)]) -> HumanoidBones {
        let mut h = HumanoidBones::default();
        for (name, id) in entries {
            h.map.insert(name.to_string(), entity(*id));
        }
        h
    }

    #[test]
    fn chest_prefers_upper_chest() {
        let bones = make_bones(&[
            (BONE_CHEST, 1),
            (BONE_UPPER_CHEST, 2),
        ]);
        assert_eq!(bones.chest(), Some(entity(2)));
    }

    #[test]
    fn chest_falls_back_to_chest() {
        let bones = make_bones(&[(BONE_CHEST, 1)]);
        assert_eq!(bones.chest(), Some(entity(1)));
    }

    #[test]
    fn chest_falls_back_to_spine() {
        let bones = make_bones(&[(BONE_SPINE, 3)]);
        assert_eq!(bones.chest(), Some(entity(3)));
    }

    #[test]
    fn is_ready_needs_chest_and_head() {
        let bones = make_bones(&[(BONE_HEAD, 1)]);
        assert!(!bones.is_ready(), "need both chest and head");

        let bones = make_bones(&[(BONE_CHEST, 1), (BONE_HEAD, 2)]);
        assert!(bones.is_ready());
    }

    #[test]
    fn get_returns_correct_entity() {
        let bones = make_bones(&[(BONE_HEAD, 5)]);
        assert_eq!(bones.get(BONE_HEAD), Some(entity(5)));
    }

    #[test]
    fn is_ready_false_when_empty() {
        assert!(!HumanoidBones::default().is_ready());
    }

    #[test]
    fn get_unknown_returns_none() {
        let bones = HumanoidBones::default();
        assert!(bones.get("NonExistentBone").is_none());
    }
}
