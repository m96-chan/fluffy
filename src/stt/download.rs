/// Whisper model downloader using hf-hub.
/// Downloads safetensors, config, and tokenizer from Hugging Face.

use std::path::PathBuf;
use tracing::info;

use crate::error::AppError;

/// Known Whisper model sizes, mapped to HF repo IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperModelId {
    Tiny,
    Base,
    Small,
    Medium,
    LargeV3Turbo,
}

impl WhisperModelId {
    pub fn repo_id(&self) -> &'static str {
        match self {
            Self::Tiny => "openai/whisper-tiny",
            Self::Base => "openai/whisper-base",
            Self::Small => "openai/whisper-small",
            Self::Medium => "openai/whisper-medium",
            Self::LargeV3Turbo => "openai/whisper-large-v3-turbo",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "tiny" | "openai/whisper-tiny" => Some(Self::Tiny),
            "base" | "openai/whisper-base" => Some(Self::Base),
            "small" | "openai/whisper-small" => Some(Self::Small),
            "medium" | "openai/whisper-medium" => Some(Self::Medium),
            "large-v3-turbo" | "openai/whisper-large-v3-turbo" => Some(Self::LargeV3Turbo),
            _ => None,
        }
    }

}

impl std::fmt::Display for WhisperModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.repo_id())
    }
}

/// Paths to downloaded Whisper model files.
pub struct WhisperModelFiles {
    pub config: PathBuf,
    pub model: PathBuf,
    pub tokenizer: PathBuf,
}

/// Download/cache Whisper model files via hf-hub.
/// Returns paths to the cached files.
pub fn ensure_whisper_model(model_id: WhisperModelId) -> Result<WhisperModelFiles, AppError> {
    let repo_id = model_id.repo_id();
    info!("Ensuring Whisper model files for {}", repo_id);

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| AppError::Stt(format!("hf-hub API init failed: {e}")))?;
    let repo = api.model(repo_id.to_string());

    let config = repo
        .get("config.json")
        .map_err(|e| AppError::Stt(format!("Failed to download config.json: {e}")))?;
    let model = repo
        .get("model.safetensors")
        .map_err(|e| AppError::Stt(format!("Failed to download model.safetensors: {e}")))?;
    let tokenizer = repo
        .get("tokenizer.json")
        .map_err(|e| AppError::Stt(format!("Failed to download tokenizer.json: {e}")))?;

    info!(
        "Whisper model files ready: config={}, model={}, tokenizer={}",
        config.display(),
        model.display(),
        tokenizer.display()
    );

    Ok(WhisperModelFiles {
        config,
        model,
        tokenizer,
    })
}

/// Load embedded mel filter coefficients for the given mel bin count.
pub fn load_mel_filters(num_mel_bins: usize) -> Result<Vec<f32>, AppError> {
    use byteorder::{ByteOrder, LittleEndian};

    let bytes = match num_mel_bins {
        80 => include_bytes!("../../assets/whisper/melfilters.bytes").as_slice(),
        128 => include_bytes!("../../assets/whisper/melfilters128.bytes").as_slice(),
        n => return Err(AppError::Stt(format!("Unsupported num_mel_bins: {n}"))),
    };

    let mut filters = vec![0f32; bytes.len() / 4];
    LittleEndian::read_f32_into(bytes, &mut filters);
    Ok(filters)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_repo_ids_are_correct() {
        assert_eq!(WhisperModelId::Tiny.repo_id(), "openai/whisper-tiny");
        assert_eq!(WhisperModelId::Base.repo_id(), "openai/whisper-base");
        assert_eq!(WhisperModelId::Small.repo_id(), "openai/whisper-small");
        assert_eq!(WhisperModelId::Medium.repo_id(), "openai/whisper-medium");
        assert_eq!(
            WhisperModelId::LargeV3Turbo.repo_id(),
            "openai/whisper-large-v3-turbo"
        );
    }

    #[test]
    fn from_str_roundtrip() {
        for id in [
            WhisperModelId::Tiny,
            WhisperModelId::Base,
            WhisperModelId::Small,
            WhisperModelId::Medium,
            WhisperModelId::LargeV3Turbo,
        ] {
            // Short name
            let short = id.repo_id().strip_prefix("openai/whisper-").unwrap();
            assert_eq!(WhisperModelId::from_str(short), Some(id));
            // Full repo ID
            assert_eq!(WhisperModelId::from_str(id.repo_id()), Some(id));
        }
    }

    #[test]
    fn from_str_unknown_returns_none() {
        assert_eq!(WhisperModelId::from_str("unknown"), None);
    }

    #[test]
    fn mel_filters_80_loads() {
        let filters = load_mel_filters(80).unwrap();
        // 80 mel bins * 201 frequency bins = 16080 values
        assert_eq!(filters.len(), 80 * 201);
    }

    #[test]
    fn mel_filters_128_loads() {
        let filters = load_mel_filters(128).unwrap();
        assert_eq!(filters.len(), 128 * 201);
    }

    #[test]
    fn mel_filters_unsupported() {
        assert!(load_mel_filters(64).is_err());
    }
}
