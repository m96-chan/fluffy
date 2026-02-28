use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Application configuration — stored as a Bevy Resource.
#[derive(Debug, Clone, Serialize, Deserialize, Resource)]
pub struct AppConfig {
    pub api_key: String,
    pub anthropic_api_url: String,
    pub model: String,
    pub tts_clone_voice_wav: PathBuf,
    pub vad_threshold: f32,
    pub vad_silence_hold_frames: usize,
    pub audio_device: Option<String>,
    pub system_prompt: String,
    /// Path to the Whisper GGML model file.
    pub whisper_model_path: PathBuf,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("ANTHROPIC_API_KEY")
                .or_else(|_| std::env::var("CLAUDE_API_KEY"))
                .unwrap_or_default(),
            anthropic_api_url: std::env::var("ANTHROPIC_API_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com/v1/messages".to_string()),
            model: std::env::var("ANTHROPIC_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-6".to_string()),
            tts_clone_voice_wav: std::env::var("FLUFFY_TTS_CLONE_VOICE_WAV")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("assets/voice/このボイス.wav")),
            vad_threshold: 0.02,
            vad_silence_hold_frames: 25,
            audio_device: None,
            system_prompt: default_system_prompt(),
            whisper_model_path: default_whisper_model_path(),
        }
    }
}

fn default_whisper_model_path() -> PathBuf {
    if let Ok(p) = std::env::var("WHISPER_MODEL_PATH") {
        return PathBuf::from(p);
    }
    // XDG data dir or fallback to ~/.local/share/fluffy/models/
    let base = std::env::var("FLUFFY_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_next::data_dir()
                .unwrap_or_else(|| PathBuf::from("~/.local/share"))
                .join("fluffy")
                .join("models")
        });
    base.join("ggml-base.bin")
}

impl AppConfig {
    /// Returns true if the minimum required fields are set.
    pub fn is_ready(&self) -> ConfigReady {
        if self.api_key.is_empty() {
            return ConfigReady::MissingApiKey;
        }
        if !self.whisper_model_path.exists() {
            return ConfigReady::MissingWhisperModel(self.whisper_model_path.clone());
        }
        ConfigReady::Ok
    }
}

fn default_system_prompt() -> String {
    r#"You are a friendly desktop mascot assistant named Fluffy.

Guidelines:
- Keep spoken responses concise (≤2 sentences before a code block)
- Prefix your response with an emotion hint in square brackets: [happy], [thinking], [surprised], [sad], or [neutral]
- Put code in markdown fenced blocks with the language specified (e.g. ```rust)
- Code blocks will be shown in the chat overlay but not spoken aloud
- Be warm, encouraging, and enthusiastic about helping

Available tools: read_file, write_file, list_files, run_command
"#
    .to_string()
}

#[derive(Debug, PartialEq, Eq)]
pub enum ConfigReady {
    Ok,
    MissingApiKey,
    MissingWhisperModel(PathBuf),
}

/// Handle to the initialized TTS engine — stored as a Bevy Resource.
#[derive(Resource)]
pub struct TtsEngineHandle {
    pub engine: std::sync::Arc<tokio::sync::Mutex<candle_miotts::engine::TtsEngine>>,
}

/// Handle to an initialized Whisper model — stored as a Bevy Resource.
#[derive(Resource)]
pub struct WhisperModel {
    pub ctx: Arc<whisper_rs::WhisperContext>,
    pub model_path: PathBuf,
}

unsafe impl Send for WhisperModel {}
unsafe impl Sync for WhisperModel {}

/// Active pipeline state — stored as a Bevy Resource.
#[derive(Resource, Default)]
pub struct PipelineState {
    pub cancel_token: Option<CancellationToken>,
    pub receiver: Option<
        Arc<
            tokio::sync::Mutex<
                tokio::sync::mpsc::Receiver<crate::events::PipelineMessage>,
            >,
        >,
    >,
}

impl PipelineState {
    pub fn is_running(&self) -> bool {
        self.cancel_token
            .as_ref()
            .map(|t| !t.is_cancelled())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_reads_api_key_from_env() {
        std::env::set_var("ANTHROPIC_API_KEY", "test-key-123");
        let cfg = AppConfig::default();
        assert_eq!(cfg.api_key, "test-key-123");
        std::env::remove_var("ANTHROPIC_API_KEY");
    }

    #[test]
    fn default_config_missing_api_key() {
        std::env::remove_var("ANTHROPIC_API_KEY");
        let cfg = AppConfig::default();
        // model path won't exist either, but api_key check comes first
        assert_eq!(cfg.is_ready(), ConfigReady::MissingApiKey);
    }

    #[test]
    fn config_ready_missing_model() {
        let cfg = AppConfig {
            api_key: "sk-test".to_string(),
            whisper_model_path: PathBuf::from("/nonexistent/model.bin"),
            ..AppConfig::default()
        };
        assert!(matches!(cfg.is_ready(), ConfigReady::MissingWhisperModel(_)));
    }

    #[test]
    fn config_ready_ok_with_real_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let cfg = AppConfig {
            api_key: "sk-test".to_string(),
            whisper_model_path: tmp.path().to_path_buf(),
            ..AppConfig::default()
        };
        assert_eq!(cfg.is_ready(), ConfigReady::Ok);
    }

    #[test]
    fn pipeline_state_not_running_by_default() {
        let state = PipelineState::default();
        assert!(!state.is_running());
    }

    #[test]
    fn pipeline_state_running_after_token_set() {
        let token = CancellationToken::new();
        let state = PipelineState {
            cancel_token: Some(token),
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
}
