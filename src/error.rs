use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("Audio error: {0}")]
    Audio(String),

    #[error("STT error: {0}")]
    Stt(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Sandbox violation: access to {0} is not allowed")]
    SandboxViolation(String),
}
