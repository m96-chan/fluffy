pub mod expressions;
pub mod loader;
pub mod vrma;

use bevy::prelude::*;
use bevy::gltf::Gltf;

pub use expressions::VrmExpressionMap;
pub use loader::VrmLoaderPlugin;

pub struct VrmPlugin;

impl Plugin for VrmPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(VrmLoaderPlugin)
            .add_plugins(vrma::VrmaPlugin)
            .add_systems(Update, setup_vrm_expressions);
    }
}

/// Marker: this entity holds a Handle<Gltf> for a VRM file.
/// After the Gltf loads, expressions are parsed and VrmExpressionMap is inserted.
#[derive(Component)]
pub struct VrmGltfHandle(pub Handle<Gltf>);

/// Once the Gltf asset is ready, parse VRM expressions and spawn the scene.
fn setup_vrm_expressions(
    mut commands: Commands,
    gltf_assets: Res<Assets<Gltf>>,
    query: Query<(Entity, &VrmGltfHandle), Without<VrmExpressionMap>>,
) {
    for (entity, handle) in query.iter() {
        let Some(gltf) = gltf_assets.get(&handle.0) else {
            continue;
        };

        let expression_map = expressions::parse_expressions(gltf);
        info!(
            "VRM expressions ready: {} presets",
            expression_map.binds.len()
        );

        // Spawn the default scene as a child
        let scene = gltf
            .default_scene
            .clone()
            .or_else(|| gltf.scenes.first().cloned());

        if let Some(scene_handle) = scene {
            commands.entity(entity).insert((
                expression_map,
                SceneRoot(scene_handle),
            ));
        } else {
            warn!("VRM: no scenes found in GLTF");
            commands.entity(entity).insert(expression_map);
        }
    }
}
