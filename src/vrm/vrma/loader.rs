/// VRMA (.vrma) asset loader.
/// Registers `.vrma` extension using Bevy's GltfLoader, same pattern as VrmGltfLoader.

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::gltf::{DefaultGltfImageSampler, GltfError, GltfLoader, GltfLoaderSettings};
use bevy::gltf::extensions::GltfExtensionHandlers;
use bevy::image::{CompressedImageFormatSupport, CompressedImageFormats, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::reflect::TypePath;

pub struct VrmaLoaderPlugin;

impl Plugin for VrmaLoaderPlugin {
    fn build(&self, app: &mut App) {
        app.preregister_asset_loader::<VrmaGltfLoader>(&["vrma"]);
    }

    fn finish(&self, app: &mut App) {
        let supported_compressed_formats =
            if let Some(r) = app.world().get_resource::<CompressedImageFormatSupport>() {
                r.0
            } else {
                CompressedImageFormats::NONE
            };

        let default_sampler =
            if let Some(r) = app.world().get_resource::<DefaultGltfImageSampler>() {
                r.get_internal()
            } else {
                let r = DefaultGltfImageSampler::new(&ImageSamplerDescriptor::default());
                let s = r.get_internal();
                app.insert_resource(r);
                s
            };

        let extensions = if let Some(r) = app.world().get_resource::<GltfExtensionHandlers>() {
            r.0.clone()
        } else {
            let r = GltfExtensionHandlers::default();
            let h = r.0.clone();
            app.insert_resource(r);
            h
        };

        app.register_asset_loader(VrmaGltfLoader(GltfLoader {
            supported_compressed_formats,
            custom_vertex_attributes: Default::default(),
            default_sampler,
            default_convert_coordinates: Default::default(),
            extensions,
        }));
    }
}

/// Thin wrapper around GltfLoader that handles .vrma extension.
#[derive(TypePath)]
pub struct VrmaGltfLoader(GltfLoader);

impl AssetLoader for VrmaGltfLoader {
    type Asset = bevy::gltf::Gltf;
    type Settings = GltfLoaderSettings;
    type Error = GltfError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &GltfLoaderSettings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<bevy::gltf::Gltf, GltfError> {
        self.0.load(reader, settings, load_context).await
    }

    fn extensions(&self) -> &[&str] {
        &["vrma"]
    }
}
