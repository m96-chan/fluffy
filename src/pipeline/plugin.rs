/// Bevy plugin that handles pipeline start/stop via keyboard input.

use bevy::prelude::*;
use std::sync::Arc;

use crate::events::PipelineMessage;
use crate::state::{AppConfig, ConfigReady, PipelineState, WhisperModel};

pub struct PipelinePlugin;

impl Plugin for PipelinePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, toggle_pipeline);
    }
}

/// Space → start pipeline, Space again → stop.
fn toggle_pipeline(
    keys: Res<ButtonInput<KeyCode>>,
    config: Res<AppConfig>,
    whisper: Option<Res<WhisperModel>>,
    mut pipeline: ResMut<PipelineState>,
    mut writer: MessageWriter<PipelineMessage>,
) {
    if !keys.just_pressed(KeyCode::Space) {
        return;
    }

    if pipeline.is_running() {
        stop_pipeline(&mut pipeline, &mut writer);
    } else {
        start_pipeline(config, whisper, &mut pipeline, &mut writer);
    }
}

fn start_pipeline(
    config: Res<AppConfig>,
    whisper: Option<Res<WhisperModel>>,
    pipeline: &mut PipelineState,
    writer: &mut MessageWriter<PipelineMessage>,
) {
    match config.is_ready() {
        ConfigReady::MissingApiKey => {
            warn!("Pipeline: ANTHROPIC_API_KEY is not set");
            writer.write(PipelineMessage::PipelineError {
                source: "config".into(),
                message: "ANTHROPIC_API_KEY is not set".into(),
            });
            return;
        }
        ConfigReady::MissingWhisperModel(path) => {
            warn!("Pipeline: Whisper model not found at {:?}", path);
            writer.write(PipelineMessage::PipelineError {
                source: "config".into(),
                message: format!("Whisper model not found: {}", path.display()),
            });
            return;
        }
        ConfigReady::Ok => {}
    }

    let Some(whisper) = whisper else {
        warn!("Pipeline: WhisperModel resource not loaded yet");
        return;
    };

    info!("Pipeline: starting");

    let config_arc = Arc::new(config.clone());
    let ctx_arc = whisper.ctx.clone();

    let runtime = tokio::runtime::Handle::current();
    let (cancel, rx) = runtime.block_on(async {
        crate::pipeline::coordinator::start_pipeline(config_arc, ctx_arc)
            .await
            .expect("Failed to start pipeline")
    });

    pipeline.cancel_token = Some(cancel);
    pipeline.receiver = Some(Arc::new(tokio::sync::Mutex::new(rx)));

    info!("Pipeline: started");
}

fn stop_pipeline(pipeline: &mut PipelineState, _writer: &mut MessageWriter<PipelineMessage>) {
    if let Some(token) = pipeline.cancel_token.take() {
        token.cancel();
        info!("Pipeline: stopped");
    }
    pipeline.receiver = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ConfigReady;
    use std::path::PathBuf;
    use tokio_util::sync::CancellationToken;

    // Helper: config that is ready (has api_key, model file exists)
    fn ready_config() -> AppConfig {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        AppConfig {
            api_key: "sk-test".to_string(),
            whisper_model_path: tmp.path().to_path_buf(),
            ..AppConfig::default()
        }
    }

    #[test]
    fn config_not_ready_without_api_key() {
        std::env::remove_var("ANTHROPIC_API_KEY");
        let cfg = AppConfig {
            api_key: String::new(),
            ..AppConfig::default()
        };
        assert_eq!(cfg.is_ready(), ConfigReady::MissingApiKey);
    }

    #[test]
    fn config_not_ready_without_model_file() {
        let cfg = AppConfig {
            api_key: "sk-test".to_string(),
            whisper_model_path: PathBuf::from("/does/not/exist.bin"),
            ..AppConfig::default()
        };
        assert!(matches!(cfg.is_ready(), ConfigReady::MissingWhisperModel(_)));
    }

    #[test]
    fn pipeline_state_default_not_running() {
        let state = PipelineState::default();
        assert!(!state.is_running());
    }

    #[test]
    fn pipeline_state_running_when_token_active() {
        let state = PipelineState {
            cancel_token: Some(CancellationToken::new()),
            receiver: None,
        };
        assert!(state.is_running());
    }

    #[test]
    fn pipeline_state_not_running_after_cancel() {
        let token = CancellationToken::new();
        token.cancel();
        let state = PipelineState {
            cancel_token: Some(token),
            receiver: None,
        };
        assert!(!state.is_running());
    }

    #[test]
    fn stop_clears_cancel_token() {
        let token = CancellationToken::new();
        let mut pipeline = PipelineState {
            cancel_token: Some(token),
            receiver: None,
        };
        assert!(pipeline.is_running());

        // Simulate stop
        if let Some(t) = pipeline.cancel_token.take() {
            t.cancel();
        }
        pipeline.receiver = None;

        assert!(!pipeline.is_running());
        assert!(pipeline.cancel_token.is_none());
    }
}
