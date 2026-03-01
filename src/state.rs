use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Application configuration — stored as a Bevy Resource.
///
/// Loaded from `~/.config/fluffy/config.toml`.
/// Missing fields fall back to defaults (env vars, then hardcoded).
#[derive(Debug, Clone, Serialize, Deserialize, Resource)]
#[serde(default)]
pub struct AppConfig {
    pub api_key: String,
    pub anthropic_api_url: String,
    pub model: String,
    pub tts_clone_voice_wav: PathBuf,
    pub vad_threshold: f32,
    pub vad_silence_hold_frames: usize,
    pub audio_device: Option<String>,
    pub system_prompt: String,
    /// Whisper model size (e.g. "medium", "small", "large-v3-turbo").
    pub whisper_model_id: String,
    /// Whisper STT language code (e.g. "ja", "en", "auto").
    pub stt_language: String,
    /// Delay (ms) after turn start before VAD is treated as barge-in.
    /// VAD within this window is assumed to be speaker echo and ignored.
    pub barge_in_delay_ms: u64,
    /// Whether to show a drop shadow under the mascot.
    pub shadow_enabled: bool,
    /// Opacity of the drop shadow (0.0–1.0).
    pub shadow_opacity: f32,
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
            whisper_model_id: std::env::var("WHISPER_MODEL_ID")
                .unwrap_or_else(|_| "medium".to_string()),
            stt_language: "ja".to_string(),
            barge_in_delay_ms: 500,
            shadow_enabled: true,
            shadow_opacity: 0.35,
        }
    }
}

impl AppConfig {
    /// Path to the config file.
    pub fn config_path() -> PathBuf {
        dirs_next::config_dir()
            .unwrap_or_else(|| PathBuf::from("~/.config"))
            .join("fluffy")
            .join("config.toml")
    }

    /// Load from `~/.config/fluffy/config.toml`.
    /// Missing fields fall back to `Default` (env vars, then hardcoded).
    pub fn load() -> Self {
        let path = Self::config_path();
        match std::fs::read_to_string(&path) {
            Ok(contents) => {
                let mut cfg: Self = toml::from_str(&contents).unwrap_or_else(|e| {
                    tracing::warn!("Failed to parse {}: {} — using defaults", path.display(), e);
                    Self::default()
                });
                // api_key: env var always takes precedence over TOML
                let env_key = std::env::var("ANTHROPIC_API_KEY")
                    .or_else(|_| std::env::var("CLAUDE_API_KEY"))
                    .unwrap_or_default();
                if !env_key.is_empty() {
                    cfg.api_key = env_key;
                }
                // system_prompt: if not set in TOML, load from file
                if cfg.system_prompt.is_empty() {
                    cfg.system_prompt = default_system_prompt();
                }
                cfg
            }
            Err(_) => {
                tracing::info!("No config found at {} — using defaults", path.display());
                Self::default()
            }
        }
    }

    /// Returns true if the minimum required fields are set.
    pub fn is_ready(&self) -> ConfigReady {
        if self.api_key.is_empty() {
            return ConfigReady::MissingApiKey;
        }
        if crate::stt::download::WhisperModelId::from_str(&self.whisper_model_id).is_none() {
            return ConfigReady::InvalidWhisperModel(self.whisper_model_id.clone());
        }
        ConfigReady::Ok
    }
}

fn default_system_prompt() -> String {
    // ~/.config/fluffy/system_prompt.txt があればそちらを使う
    if let Some(path) = dirs_next::config_dir().map(|d| d.join("fluffy").join("system_prompt.txt"))
    {
        if let Ok(text) = std::fs::read_to_string(&path) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }

    // 環境変数 FLUFFY_SYSTEM_PROMPT_FILE でもパス指定可
    if let Ok(path) = std::env::var("FLUFFY_SYSTEM_PROMPT_FILE") {
        if let Ok(text) = std::fs::read_to_string(&path) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }

    r#"You are a friendly desktop mascot assistant named Fluffy.

Guidelines:
- Keep spoken responses concise (≤2 sentences before a code block)
- Prefix your response with an emotion hint in square brackets: [happy], [thinking], [surprised], [sad], or [neutral]
- Put code in markdown fenced blocks with the language specified (e.g. ```rust)
- Code blocks will be shown in the chat overlay but not spoken aloud
- Be warm, encouraging, and enthusiastic about helping
- Reply in the same language the user speaks

Available tools: read_file, write_file, list_files, run_command
"#
    .to_string()
}

#[derive(Debug, PartialEq, Eq)]
pub enum ConfigReady {
    Ok,
    MissingApiKey,
    InvalidWhisperModel(String),
}

/// Handle to an initialized Whisper model — stored as a Bevy Resource.
#[derive(Resource)]
pub struct WhisperModel {
    pub engine: Arc<crate::stt::whisper::WhisperEngine>,
}

/// Active pipeline state — stored as a Bevy Resource.
///
/// Owns a tokio runtime that lives as long as the pipeline is running.
/// Bevy systems run on the Compute Task Pool which has no tokio reactor,
/// so we need our own runtime for async pipeline tasks.
pub struct PipelineState {
    pub cancel_token: Option<CancellationToken>,
    pub receiver: Option<
        Arc<
            tokio::sync::Mutex<
                tokio::sync::mpsc::Receiver<crate::events::PipelineMessage>,
            >,
        >,
    >,
    /// Tokio runtime that keeps spawned pipeline tasks alive.
    pub runtime: Option<tokio::runtime::Runtime>,
}

impl Resource for PipelineState {}

impl Default for PipelineState {
    fn default() -> Self {
        Self {
            cancel_token: None,
            receiver: None,
            runtime: None,
        }
    }
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
    fn config_ready_invalid_model() {
        let cfg = AppConfig {
            api_key: "sk-test".to_string(),
            whisper_model_id: "nonexistent-model".to_string(),
            ..AppConfig::default()
        };
        assert!(matches!(
            cfg.is_ready(),
            ConfigReady::InvalidWhisperModel(_)
        ));
    }

    #[test]
    fn config_ready_ok_with_valid_model() {
        let cfg = AppConfig {
            api_key: "sk-test".to_string(),
            whisper_model_id: "medium".to_string(),
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
}
