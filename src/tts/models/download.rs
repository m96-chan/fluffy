use std::path::PathBuf;
use tracing::info;

/// HuggingFace model repository identifiers.
const LFM2_REPO: &str = "Aratako/MioTTS-2.6B";
const MIOCODEC_REPO: &str = "Aratako/MioCodec-25Hz-44.1kHz-v2";
const WAVLM_REPO: &str = "microsoft/wavlm-base-plus";

/// Paths to downloaded model files.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// LFM2 SafeTensors model files (may be sharded)
    pub lfm2_safetensors: Vec<PathBuf>,
    /// LFM2 tokenizer.json
    pub lfm2_tokenizer: PathBuf,
    /// LFM2 config.json
    pub lfm2_config: PathBuf,
    /// MioCodec model.safetensors
    pub miocodec_safetensors: PathBuf,
    /// MioCodec config.yaml
    pub miocodec_config: PathBuf,
    /// WavLM pytorch_model.bin (no safetensors available)
    pub wavlm_pth: PathBuf,
}

/// Download all required models from HuggingFace Hub.
/// Uses the hf-hub crate with the default cache directory (~/.cache/huggingface).
/// Files are only downloaded on first run; subsequent calls are instant.
pub async fn ensure_models_downloaded() -> Result<ModelPaths, crate::error::AppError> {
    info!("Ensuring TTS models are downloaded...");

    let (lfm2, miocodec, wavlm) = tokio::join!(
        download_lfm2(),
        download_miocodec(),
        download_wavlm(),
    );

    let lfm2 = lfm2?;
    let miocodec = miocodec?;
    let wavlm = wavlm?;

    info!("All TTS models ready");

    Ok(ModelPaths {
        lfm2_safetensors: lfm2.0,
        lfm2_tokenizer: lfm2.1,
        lfm2_config: lfm2.2,
        miocodec_safetensors: miocodec.0,
        miocodec_config: miocodec.1,
        wavlm_pth: wavlm,
    })
}

async fn download_lfm2() -> Result<(Vec<PathBuf>, PathBuf, PathBuf), crate::error::AppError> {
    tokio::task::spawn_blocking(|| {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::error::AppError::Download(format!("HF API init: {e}")))?;
        let repo = api.model(LFM2_REPO.to_string());

        info!("Downloading LFM2 model from {LFM2_REPO}...");

        // Download config.json first to check for sharded model
        let config_path = repo.get("config.json")
            .map_err(|e| crate::error::AppError::Download(format!("LFM2 config.json: {e}")))?;

        let tokenizer_path = repo.get("tokenizer.json")
            .map_err(|e| crate::error::AppError::Download(format!("LFM2 tokenizer.json: {e}")))?;

        // Try model.safetensors.index.json for sharded model
        let safetensor_paths = match repo.get("model.safetensors.index.json") {
            Ok(index_path) => {
                // Sharded model — parse index to get shard filenames
                let index_json: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(&index_path)
                        .map_err(|e| crate::error::AppError::Download(format!("Read index: {e}")))?
                )
                .map_err(|e| crate::error::AppError::Download(format!("Parse index: {e}")))?;

                let weight_map = index_json
                    .get("weight_map")
                    .and_then(|m| m.as_object())
                    .ok_or_else(|| crate::error::AppError::Download("Missing weight_map".into()))?;

                // Collect unique shard filenames
                let mut shards: Vec<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                shards.sort();
                shards.dedup();

                info!("LFM2: downloading {} shards", shards.len());

                let mut paths = Vec::with_capacity(shards.len());
                for shard in &shards {
                    let path = repo.get(shard)
                        .map_err(|e| crate::error::AppError::Download(format!("LFM2 {shard}: {e}")))?;
                    paths.push(path);
                }
                paths
            }
            Err(_) => {
                // Single file model
                let path = repo.get("model.safetensors")
                    .map_err(|e| crate::error::AppError::Download(format!("LFM2 model.safetensors: {e}")))?;
                vec![path]
            }
        };

        info!("LFM2 model ready ({} files)", safetensor_paths.len());
        Ok((safetensor_paths, tokenizer_path, config_path))
    })
    .await
    .map_err(|e| crate::error::AppError::Download(format!("Join error: {e}")))?
}

async fn download_miocodec() -> Result<(PathBuf, PathBuf), crate::error::AppError> {
    tokio::task::spawn_blocking(|| {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::error::AppError::Download(format!("HF API init: {e}")))?;
        let repo = api.model(MIOCODEC_REPO.to_string());

        info!("Downloading MioCodec from {MIOCODEC_REPO}...");

        let model_path = repo.get("model.safetensors")
            .map_err(|e| crate::error::AppError::Download(format!("MioCodec model: {e}")))?;
        let config_path = repo.get("config.yaml")
            .map_err(|e| crate::error::AppError::Download(format!("MioCodec config: {e}")))?;

        info!("MioCodec model ready");
        Ok((model_path, config_path))
    })
    .await
    .map_err(|e| crate::error::AppError::Download(format!("Join error: {e}")))?
}

async fn download_wavlm() -> Result<PathBuf, crate::error::AppError> {
    tokio::task::spawn_blocking(|| {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::error::AppError::Download(format!("HF API init: {e}")))?;
        let repo = api.model(WAVLM_REPO.to_string());

        info!("Downloading WavLM from {WAVLM_REPO}...");

        let model_path = repo.get("pytorch_model.bin")
            .map_err(|e| crate::error::AppError::Download(format!("WavLM model: {e}")))?;

        info!("WavLM model ready");
        Ok(model_path)
    })
    .await
    .map_err(|e| crate::error::AppError::Download(format!("Join error: {e}")))?
}
