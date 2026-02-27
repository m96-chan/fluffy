/// Messages flowing between the pipeline and the mascot rendering.
/// Bevy 0.18 uses Message/MessageReader/MessageWriter instead of Event/EventReader/EventWriter.

use bevy::prelude::*;

#[derive(Message, Debug, Clone)]
pub enum PipelineMessage {
    /// Pipeline phase changed
    PhaseChanged(MascotPhase),
    /// STT produced text
    SttResult { text: String },
    /// LLM streaming token
    LlmToken { token: String },
    /// LLM finished a full response
    LlmDone,
    /// Emotion hint detected in LLM output
    EmotionChange { emotion: String },
    /// PCM audio chunk for lip sync amplitude
    LipSyncAmplitude { amplitude: f32 },
    /// Error from any pipeline stage
    PipelineError { source: String, message: String },
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum MascotPhase {
    #[default]
    Idle,
    Listening,
    Processing,
    Thinking,
    Speaking,
}
