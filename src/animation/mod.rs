pub mod bones;
pub mod procedural;

use bevy::prelude::*;

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
            .add_systems(Update, (sync_phase, discover_bones, drive_animation).chain());
    }
}

/// Tracks the current mascot phase for animation scaling.
#[derive(Resource, Default)]
pub struct CurrentPhase(pub MascotPhase);

fn sync_phase(mut phase: ResMut<CurrentPhase>, mut reader: MessageReader<PipelineMessage>) {
    for msg in reader.read() {
        if let PipelineMessage::PhaseChanged(p) = msg {
            phase.0 = p.clone();
        }
    }
}

/// After the VRM scene spawns, walk the hierarchy to find bone entities.
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

/// Apply breathing and head sway every frame.
fn drive_animation(
    time: Res<Time>,
    phase: Res<CurrentPhase>,
    mascot_q: Query<&HumanoidBones, With<MascotTag>>,
    mut transforms: Query<&mut Transform>,
) {
    let Ok(bones) = mascot_q.single() else {
        return;
    };

    let t = time.elapsed_secs();
    let breath_scale = breathing_scale_for_phase(&phase.0);

    // Breathing — chest bone X rotation
    if let Some(chest_e) = bones.chest() {
        if let Ok(mut tf) = transforms.get_mut(chest_e) {
            let angle = breathing_rotation(t, &BreathingParams::default(), breath_scale);
            // Apply as a small additive rotation on top of rest pose
            tf.rotation = Quat::from_rotation_x(angle);
        }
    }

    // Head sway — head bone X/Y rotation
    if let Some(head_e) = bones.head() {
        if let Ok(mut tf) = transforms.get_mut(head_e) {
            let (rx, ry) = head_sway(t, &HeadSwayParams::default());
            tf.rotation = Quat::from_euler(EulerRot::XYZ, rx, ry, 0.0);
        }
    }
}
