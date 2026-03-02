/// Startup system that downloads and loads the Whisper model.

use bevy::prelude::*;
use std::sync::Arc;

use crate::stt::download::WhisperModelId;
use crate::state::{AppConfig, WhisperModel};

pub struct WhisperLoaderPlugin;

impl Plugin for WhisperLoaderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, load_whisper_model);
    }
}

fn load_whisper_model(mut commands: Commands, config: Res<AppConfig>) {
    let Some(model_id) = WhisperModelId::from_str(&config.whisper_model_id) else {
        warn!(
            "Unknown Whisper model '{}' — voice input disabled. \
             Use one of: tiny, base, small, medium, large-v3-turbo",
            config.whisper_model_id
        );
        return;
    };

    info!("Downloading/caching Whisper model: {} ...", model_id);

    let files = match crate::stt::download::ensure_whisper_model(model_id) {
        Ok(f) => f,
        Err(e) => {
            error!("Failed to download Whisper model: {}", e);
            return;
        }
    };

    // Select device: CUDA if available, else CPU
    let device = select_device();
    info!("Loading Whisper model on {:?}...", device);

    match crate::stt::whisper::WhisperEngine::load(&files, &device) {
        Ok(engine) => {
            commands.insert_resource(WhisperModel {
                engine: Arc::new(engine),
            });
            info!("Whisper model loaded on {:?}.", device);
        }
        Err(e) => {
            error!("Failed to load Whisper model: {}", e);
        }
    }
}

fn select_device() -> candle_core::Device {
    #[cfg(feature = "cuda")]
    {
        match candle_core::Device::new_cuda(0) {
            Ok(d) => return d,
            Err(e) => {
                tracing::warn!("CUDA not available, falling back to CPU: {}", e);
            }
        }
    }
    candle_core::Device::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ConfigReady;

    #[test]
    fn unknown_model_id_detected() {
        let cfg = AppConfig {
            api_key: "sk-test".to_string(),
            whisper_model_id: "nonexistent".to_string(),
            ..AppConfig::default()
        };
        assert!(matches!(
            cfg.is_ready(),
            ConfigReady::InvalidWhisperModel(_)
        ));
    }

    #[test]
    fn valid_model_id_passes_readiness() {
        let cfg = AppConfig {
            api_key: "sk-test".to_string(),
            whisper_model_id: "medium".to_string(),
            ..AppConfig::default()
        };
        assert_eq!(cfg.is_ready(), ConfigReady::Ok);
    }

    #[test]
    fn model_id_default_is_medium() {
        std::env::remove_var("WHISPER_MODEL_ID");
        let cfg = AppConfig::default();
        assert_eq!(cfg.whisper_model_id, "medium");
    }
}
