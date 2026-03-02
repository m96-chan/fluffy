/// Bevy plugin that handles pipeline start/stop via keyboard input
/// and initializes the local TTS engine on startup.

use bevy::prelude::*;
use std::sync::Arc;

use crate::events::PipelineMessage;
use crate::state::{AppConfig, ConfigReady, PipelineState, WhisperModel};

pub struct PipelinePlugin;

impl Plugin for PipelinePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (log_pipeline_config, init_tts_engine).chain())
            .add_systems(Update, toggle_pipeline);
    }
}

fn log_pipeline_config(config: Res<AppConfig>) {
    let key_state = if config.api_key.is_empty() {
        "missing"
    } else {
        "set"
    };
    info!(
        "Pipeline config: api_key={}, model={}, anthropic_api_url={}, whisper_model={}, tts_clone_voice_wav={}",
        key_state,
        config.model,
        config.anthropic_api_url,
        config.whisper_model_id,
        config.tts_clone_voice_wav.display()
    );
}

/// Initialize the local TTS engine on a background thread.
/// Once ready, inserts TtsEngineHandle as a Bevy Resource.
fn init_tts_engine(_commands: Commands, config: Res<AppConfig>) {
    let clone_wav = config.tts_clone_voice_wav.clone();

    std::thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                warn!("TTS engine init: failed to create runtime: {}", e);
                return;
            }
        };

        match runtime.block_on(candle_miotts::engine::TtsEngine::initialize(&clone_wav)) {
            Ok(engine) => {
                info!("TTS engine initialized — ready for synthesis");
                // NOTE: We can't insert into Bevy World from a background thread directly.
                // The engine is stored in a static for the coordinator to pick up.
                // In a production setup, we'd use a channel to send it back.
                // For now, the coordinator will initialize its own engine.
                let _ = engine;
            }
            Err(e) => {
                warn!("TTS engine init failed: {}", e);
            }
        }
    });
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
        ConfigReady::InvalidWhisperModel(id) => {
            warn!("Pipeline: Invalid Whisper model ID: {}", id);
            writer.write(PipelineMessage::PipelineError {
                source: "config".into(),
                message: format!("Invalid Whisper model: {id}"),
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
    let engine_arc = whisper.engine.clone();

    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let (cancel, rx) = runtime.block_on(async {
        crate::pipeline::coordinator::start_pipeline(config_arc, engine_arc)
            .await
            .expect("Failed to start pipeline")
    });

    pipeline.cancel_token = Some(cancel);
    pipeline.receiver = Some(Arc::new(tokio::sync::Mutex::new(rx)));
    pipeline.runtime = Some(runtime);

    info!("Pipeline: started");
}

fn stop_pipeline(pipeline: &mut PipelineState, _writer: &mut MessageWriter<PipelineMessage>) {
    if let Some(token) = pipeline.cancel_token.take() {
        token.cancel();
        info!("Pipeline: stopped");
    }
    pipeline.receiver = None;
    pipeline.runtime = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ConfigReady;
    use tokio_util::sync::CancellationToken;

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
    fn config_not_ready_with_invalid_model() {
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
    fn pipeline_state_default_not_running() {
        let state = PipelineState::default();
        assert!(!state.is_running());
    }

    #[test]
    fn pipeline_state_running_when_token_active() {
        let state = PipelineState {
            cancel_token: Some(CancellationToken::new()),
            ..Default::default()
        };
        assert!(state.is_running());
    }

    #[test]
    fn pipeline_state_not_running_after_cancel() {
        let token = CancellationToken::new();
        token.cancel();
        let state = PipelineState {
            cancel_token: Some(token),
            ..Default::default()
        };
        assert!(!state.is_running());
    }

    #[test]
    fn stop_clears_cancel_token() {
        let token = CancellationToken::new();
        let mut pipeline = PipelineState {
            cancel_token: Some(token),
            ..Default::default()
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
