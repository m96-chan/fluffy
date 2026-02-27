use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::audio::{capture, playback::PlaybackEngine, vad::VadState};
use crate::error::AppError;
use crate::events::{MascotPhase, PipelineMessage};
use crate::llm::{client as llm_client, tool_use};
use crate::pipeline::lip_sync;
use crate::state::AppConfig;
use crate::tts::{client as tts_client, sentence::SentenceSplitter};
use crate::stt::whisper as stt_whisper;

/// Message types flowing through the internal pipeline channels.
enum PipelineInput {
    Utterance(Vec<f32>),
    TextInput(String),
}

/// Starts the voice pipeline as an async task.
///
/// Returns:
/// - `CancellationToken` to stop the pipeline
/// - `mpsc::Receiver<PipelineMessage>` that yields messages for Bevy to consume
pub async fn start_pipeline(
    config: Arc<AppConfig>,
    whisper_ctx: Arc<whisper_rs::WhisperContext>,
) -> Result<(CancellationToken, mpsc::Receiver<PipelineMessage>), AppError> {
    let (msg_tx, msg_rx) = mpsc::channel::<PipelineMessage>(64);
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();

    tokio::spawn(async move {
        if let Err(e) = run_pipeline(config, whisper_ctx, msg_tx, cancel_clone).await {
            error!("Pipeline error: {}", e);
        }
    });

    Ok((cancel, msg_rx))
}

async fn run_pipeline(
    config: Arc<AppConfig>,
    whisper_ctx: Arc<whisper_rs::WhisperContext>,
    msg_tx: mpsc::Sender<PipelineMessage>,
    cancel: CancellationToken,
) -> Result<(), AppError> {
    send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Idle)).await;

    // Start audio capture
    let audio_rx = capture::start_capture(config.audio_device.clone()).await?;

    let mut vad = VadState::new(
        config.vad_threshold,
        config.vad_silence_hold_frames,
        8000, // ~0.5s pre-roll at 16kHz
    );

    let (utterance_tx, mut utterance_rx) = mpsc::channel::<PipelineInput>(4);
    let utt_tx = utterance_tx.clone();

    // VAD task
    let cancel_vad = cancel.clone();
    let msg_tx_vad = msg_tx.clone();
    let mut audio_rx = audio_rx;

    tokio::spawn(async move {
        loop {
            tokio::select! {
                Some(frame) = audio_rx.recv() => {
                    if let Some(utterance) = vad.process_frame(&frame) {
                        info!("VAD: utterance detected ({} samples)", utterance.len());
                        send(&msg_tx_vad, PipelineMessage::PhaseChanged(MascotPhase::Processing)).await;
                        let _ = utt_tx.send(PipelineInput::Utterance(utterance)).await;
                    }
                }
                _ = cancel_vad.cancelled() => {
                    if let Some(utterance) = vad.flush() {
                        let _ = utt_tx.send(PipelineInput::Utterance(utterance)).await;
                    }
                    break;
                }
            }
        }
    });

    send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Listening)).await;

    let mut conversation: Vec<llm_client::Message> = Vec::new();

    loop {
        tokio::select! {
            Some(input) = utterance_rx.recv() => {
                let text = match input {
                    PipelineInput::TextInput(t) => t,
                    PipelineInput::Utterance(pcm) => {
                        send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Processing)).await;

                        match stt_whisper::transcribe(whisper_ctx.clone(), pcm).await {
                            Ok(t) if t.is_empty() => {
                                info!("STT returned empty, skipping");
                                send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Listening)).await;
                                continue;
                            }
                            Ok(t) => {
                                info!("STT: {}", t);
                                send(&msg_tx, PipelineMessage::SttResult { text: t.clone() }).await;
                                t
                            }
                            Err(e) => {
                                error!("STT error: {}", e);
                                send(&msg_tx, PipelineMessage::PipelineError {
                                    source: "stt".into(),
                                    message: e.to_string(),
                                }).await;
                                send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Listening)).await;
                                continue;
                            }
                        }
                    }
                };

                conversation.push(llm_client::Message {
                    role: "user".to_string(),
                    content: llm_client::MessageContent::Text(text),
                });

                send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Thinking)).await;

                process_llm_turn(&config, &mut conversation, &msg_tx, cancel.clone()).await;

                if !cancel.is_cancelled() {
                    send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Listening)).await;
                }
            }
            _ = cancel.cancelled() => {
                info!("Pipeline cancelled");
                break;
            }
        }
    }

    send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Idle)).await;
    Ok(())
}

async fn process_llm_turn(
    config: &Arc<AppConfig>,
    conversation: &mut Vec<llm_client::Message>,
    msg_tx: &mpsc::Sender<PipelineMessage>,
    cancel: CancellationToken,
) {
    let tools = llm_client::make_tool_definitions();
    let (llm_tx, mut llm_rx) = mpsc::channel(64);

    let api_key = config.api_key.clone();
    let model = config.model.clone();
    let system = config.system_prompt.clone();
    let msgs = conversation.clone();
    let tools_clone = tools.clone();

    tokio::spawn(async move {
        if let Err(e) = llm_client::stream_completion(
            &api_key, &model, &system, msgs, tools_clone, llm_tx,
        ).await {
            error!("LLM error: {}", e);
        }
    });

    let mut sentence_splitter = SentenceSplitter::new();
    let mut full_response = String::new();

    // PCM bridge: tokio mpsc → std mpsc → rodio (which is !Send)
    let (pcm_bridge_tx, pcm_bridge_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let msg_tx_play = msg_tx.clone();

    std::thread::spawn(move || {
        match PlaybackEngine::new() {
            Ok(engine) => {
                while let Ok(chunk) = pcm_bridge_rx.recv() {
                    let amp = lip_sync::compute_lip_sync_amplitude(&chunk);
                    // Fire-and-forget; Bevy reads these next frame
                    let _ = msg_tx_play.try_send(PipelineMessage::LipSyncAmplitude { amplitude: amp });
                    engine.queue_chunk(chunk);
                }
            }
            Err(e) => error!("Playback engine init failed: {}", e),
        }
    });

    let (tts_tx, mut tts_rx) = mpsc::channel::<Vec<f32>>(8);
    tokio::spawn(async move {
        while let Some(chunk) = tts_rx.recv().await {
            let _ = pcm_bridge_tx.send(chunk);
        }
    });

    let tts_url = config.tts_url.clone();
    let (sentence_tts_tx, mut sentence_tts_rx) = mpsc::channel::<String>(4);
    let msg_tx_tts = msg_tx.clone();

    tokio::spawn(async move {
        while let Some(sentence) = sentence_tts_rx.recv().await {
            match tts_client::synthesize(&tts_url, &sentence).await {
                Ok(mut pcm_rx) => {
                    while let Some(chunk) = pcm_rx.recv().await {
                        let _ = tts_tx.send(chunk).await;
                    }
                }
                Err(e) => {
                    error!("TTS error: {}", e);
                    send(&msg_tx_tts, PipelineMessage::PipelineError {
                        source: "tts".into(),
                        message: e.to_string(),
                    }).await;
                }
            }
        }
    });

    loop {
        tokio::select! {
            Some(chunk) = llm_rx.recv() => {
                match chunk {
                    llm_client::LlmChunk::Token(token) => {
                        send(msg_tx, PipelineMessage::LlmToken { token: token.clone() }).await;
                        full_response.push_str(&token);

                        parse_and_emit_emotion(msg_tx, &full_response).await;

                        let tts_text = strip_emotion_prefix(&token);
                        if !tts_text.is_empty() {
                            for s in sentence_splitter.push(&tts_text) {
                                let _ = sentence_tts_tx.send(s).await;
                            }
                        }
                    }

                    llm_client::LlmChunk::ToolCall { id, name, input } => {
                        match tool_use::dispatch_tool(&name, &input).await {
                            Ok(result) => {
                                conversation.push(llm_client::Message {
                                    role: "user".to_string(),
                                    content: llm_client::MessageContent::Blocks(vec![
                                        llm_client::ContentBlock {
                                            block_type: "tool_result".to_string(),
                                            tool_use_id: Some(id),
                                            content: Some(serde_json::Value::String(result)),
                                            text: None,
                                            id: None,
                                            name: None,
                                            input: None,
                                        },
                                    ]),
                                });
                            }
                            Err(e) => error!("Tool dispatch error: {}", e),
                        }
                    }

                    llm_client::LlmChunk::Done => {
                        if let Some(remaining) = sentence_splitter.flush() {
                            let _ = sentence_tts_tx.send(remaining).await;
                        }
                        conversation.push(llm_client::Message {
                            role: "assistant".to_string(),
                            content: llm_client::MessageContent::Text(full_response.clone()),
                        });
                        send(msg_tx, PipelineMessage::LlmDone).await;
                        break;
                    }

                    llm_client::LlmChunk::Sentence(_) | llm_client::LlmChunk::Error(_) => {}
                }
            }
            _ = cancel.cancelled() => break,
        }
    }
}

async fn parse_and_emit_emotion(msg_tx: &mpsc::Sender<PipelineMessage>, text: &str) {
    if let Some(start) = text.find('[') {
        if let Some(end) = text.find(']') {
            if start < end {
                let emotion = &text[start + 1..end];
                let known = ["happy", "sad", "surprised", "angry", "thinking", "neutral"];
                if known.contains(&emotion) {
                    send(msg_tx, PipelineMessage::EmotionChange { emotion: emotion.to_string() }).await;
                }
            }
        }
    }
}

fn strip_emotion_prefix(token: &str) -> String {
    let trimmed = token.trim();
    if trimmed.starts_with('[') || trimmed.ends_with(']') {
        return String::new();
    }
    token.to_string()
}

/// Helper: send a message, logging if the channel is closed.
async fn send(tx: &mpsc::Sender<PipelineMessage>, msg: PipelineMessage) {
    if tx.send(msg).await.is_err() {
        warn!("Pipeline message channel closed");
    }
}
