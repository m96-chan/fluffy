/// Startup system that loads the Whisper model from disk.

use bevy::prelude::*;
use std::sync::Arc;

use crate::state::{AppConfig, WhisperModel};

pub struct WhisperLoaderPlugin;

impl Plugin for WhisperLoaderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, load_whisper_model);
    }
}

fn load_whisper_model(mut commands: Commands, config: Res<AppConfig>) {
    let path = &config.whisper_model_path;

    if !path.exists() {
        warn!(
            "Whisper model not found at {:?} — voice input disabled. \
             Set FLUFFY_MODEL_DIR or use the download feature (issue #2).",
            path
        );
        return;
    }

    info!("Loading Whisper model from {:?}...", path);

    match crate::stt::whisper::load_whisper_context(path) {
        Ok(ctx) => {
            commands.insert_resource(WhisperModel {
                ctx,
                model_path: path.clone(),
            });
            info!("Whisper model loaded.");
        }
        Err(e) => {
            error!("Failed to load Whisper model: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ConfigReady;
    use std::path::PathBuf;

    #[test]
    fn missing_model_path_detected_before_load() {
        let cfg = AppConfig {
            api_key: "sk-test".to_string(),
            whisper_model_path: PathBuf::from("/nonexistent/ggml.bin"),
            ..AppConfig::default()
        };
        // is_ready() catches this before we even try to load
        assert!(matches!(cfg.is_ready(), ConfigReady::MissingWhisperModel(_)));
    }

    #[test]
    fn existing_file_passes_readiness_check() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let cfg = AppConfig {
            api_key: "sk-test".to_string(),
            whisper_model_path: tmp.path().to_path_buf(),
            ..AppConfig::default()
        };
        assert_eq!(cfg.is_ready(), ConfigReady::Ok);
    }

    #[test]
    fn model_path_default_uses_data_dir() {
        let cfg = AppConfig::default();
        // Should contain "fluffy/models/ggml-base.bin" somewhere in the path
        let path_str = cfg.whisper_model_path.to_string_lossy();
        assert!(path_str.contains("fluffy"), "Expected 'fluffy' in path: {}", path_str);
        assert!(path_str.ends_with("ggml-base.bin"), "Expected ggml-base.bin: {}", path_str);
    }
}
