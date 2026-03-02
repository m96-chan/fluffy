use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::audio::{capture, playback::PlaybackEngine, vad::VadState};
use crate::error::AppError;
use crate::events::{MascotPhase, PipelineMessage};
use crate::llm::{client as llm_client, tool_use};
use crate::state::AppConfig;
use candle_miotts::engine::TtsEngine;
use candle_miotts::sentence::SentenceSplitter;
use crate::tts::client as tts_client;
use crate::stt::whisper::{self as stt_whisper, WhisperEngine};

/// Message types flowing through the internal pipeline channels.
enum PipelineInput {
    Utterance(Vec<f32>),
}

/// Starts the voice pipeline as an async task.
///
/// Returns:
/// - `CancellationToken` to stop the pipeline
/// - `mpsc::Receiver<PipelineMessage>` that yields messages for Bevy to consume
pub async fn start_pipeline(
    config: Arc<AppConfig>,
    whisper_engine: Arc<WhisperEngine>,
) -> Result<(CancellationToken, mpsc::Receiver<PipelineMessage>), AppError> {
    let (msg_tx, msg_rx) = mpsc::channel::<PipelineMessage>(64);
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();

    tokio::spawn(async move {
        if let Err(e) = run_pipeline(config, whisper_engine, msg_tx, cancel_clone).await {
            error!("Pipeline error: {}", e);
        }
    });

    Ok((cancel, msg_rx))
}

async fn run_pipeline(
    config: Arc<AppConfig>,
    whisper_engine: Arc<WhisperEngine>,
    msg_tx: mpsc::Sender<PipelineMessage>,
    cancel: CancellationToken,
) -> Result<(), AppError> {
    send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Idle)).await;

    // Initialize TTS engine
    info!("Pipeline: initializing TTS engine...");
    let tts_engine = TtsEngine::initialize(&config.tts_clone_voice_wav)
        .await
        .map_err(|e| AppError::Pipeline(format!("TTS engine init: {e}")))?;
    let tts_engine = Arc::new(tokio::sync::Mutex::new(tts_engine));
    info!("Pipeline: TTS engine ready");

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

    // Pre-synthesize interjection audio ("あ、ごめんね") for barge-in responses
    let interjection_pcm: Option<Vec<f32>> = {
        let eng = tts_engine.clone();
        match tokio::task::spawn_blocking(move || {
            let engine = eng.blocking_lock();
            engine.synthesize_blocking("あ、ごめんね")
        })
        .await
        {
            Ok(Ok(pcm)) => {
                info!("Pipeline: interjection audio cached ({} samples)", pcm.len());
                Some(pcm)
            }
            Ok(Err(e)) => {
                warn!("Pipeline: failed to pre-synthesize interjection: {}", e);
                None
            }
            Err(e) => {
                warn!("Pipeline: interjection task panicked: {}", e);
                None
            }
        }
    };

    send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Listening)).await;

    let mut conversation: Vec<llm_client::Message> = Vec::new();

    loop {
        tokio::select! {
            Some(input) = utterance_rx.recv() => {
                let pcm = match input {
                    PipelineInput::Utterance(pcm) => pcm,
                };

                let text = match transcribe_utterance(&whisper_engine, pcm, &config, &msg_tx).await {
                    Some(t) => t,
                    None => continue,
                };

                conversation.push(llm_client::Message {
                    role: "user".to_string(),
                    content: llm_client::MessageContent::Text(text),
                });

                send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Thinking)).await;

                // Run LLM turn with barge-in detection
                let turn_cancel = cancel.child_token();
                let turn_start = Instant::now();
                let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();

                let mut turn_fut = Box::pin(process_llm_turn(
                    &config,
                    &mut conversation,
                    &msg_tx,
                    turn_cancel.clone(),
                    &tts_engine,
                    stop_tx,
                    stop_rx,
                ));

                let mut interrupted_pcm: Option<Vec<f32>> = None;

                loop {
                    tokio::select! {
                        _completed = &mut turn_fut => break,
                        Some(PipelineInput::Utterance(pcm)) = utterance_rx.recv() => {
                            if turn_start.elapsed() > std::time::Duration::from_millis(config.barge_in_delay_ms) {
                                info!("Barge-in detected (elapsed {:?}), cancelling turn", turn_start.elapsed());
                                turn_cancel.cancel();
                                interrupted_pcm = Some(pcm);
                                // Wait for process_llm_turn to clean up
                                let _ = (&mut turn_fut).await;
                                break;
                            }
                            info!("VAD within echo guard window ({:?}), ignoring", turn_start.elapsed());
                        }
                        _ = cancel.cancelled() => break,
                    }
                }

                // Drop turn_fut to release &mut conversation borrow
                drop(turn_fut);

                // Drain echo utterances
                while utterance_rx.try_recv().is_ok() {}

                if let Some(pcm) = interrupted_pcm {
                    // Play interjection
                    if let Some(ref ij_pcm) = interjection_pcm {
                        play_interjection(ij_pcm, &msg_tx).await;
                    }
                    send(&msg_tx, PipelineMessage::Interrupted).await;

                    // Transcribe the interrupting utterance
                    if let Some(text) = transcribe_utterance(&whisper_engine, pcm, &config, &msg_tx).await {
                        conversation.push(llm_client::Message {
                            role: "user".to_string(),
                            content: llm_client::MessageContent::Text(text),
                        });

                        send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Thinking)).await;

                        // Process the interrupting utterance (no barge-in nesting)
                        let turn_cancel2 = cancel.child_token();
                        let (stop_tx2, stop_rx2) = std::sync::mpsc::channel::<()>();
                        process_llm_turn(
                            &config, &mut conversation, &msg_tx,
                            turn_cancel2, &tts_engine, stop_tx2, stop_rx2,
                        ).await;

                        while utterance_rx.try_recv().is_ok() {}
                    }
                }

                if cancel.is_cancelled() {
                    info!("Pipeline cancelled");
                    break;
                }

                send(&msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Listening)).await;
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
    turn_cancel: CancellationToken,
    tts_engine: &Arc<tokio::sync::Mutex<TtsEngine>>,
    stop_tx: std::sync::mpsc::Sender<()>,
    stop_rx: std::sync::mpsc::Receiver<()>,
) {
    let tools = llm_client::make_tool_definitions();
    let (llm_tx, mut llm_rx) = mpsc::channel(64);

    let api_key = config.api_key.clone();
    let api_url = config.anthropic_api_url.clone();
    let model = config.model.clone();
    let system = config.system_prompt.clone();
    let msgs = conversation.clone();
    let tools_clone = tools.clone();

    tokio::spawn(async move {
        if let Err(e) = llm_client::stream_completion(
            &api_key, &api_url, &model, &system, msgs, tools_clone, llm_tx,
        ).await {
            error!("LLM error: {}", e);
        }
    });

    let mut sentence_splitter = SentenceSplitter::new();
    let mut full_response = String::new();
    // Buffer for accumulating bracket content (e.g. "[happy]") across streamed tokens.
    // Emotion tags like [happy] may arrive split across multiple tokens: "[", "happy", "]".
    let mut bracket_buf: Option<String> = None;

    // PCM bridge: tokio mpsc -> std mpsc -> rodio (which is !Send)
    let (pcm_bridge_tx, pcm_bridge_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let msg_tx_play = msg_tx.clone();

    // Playback thread checks for stop signal alongside PCM chunks
    std::thread::spawn(move || {
        match PlaybackEngine::new() {
            Ok(engine) => {
                loop {
                    if stop_rx.try_recv().is_ok() {
                        engine.stop();
                        break;
                    }
                    match pcm_bridge_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                        Ok(chunk) => engine.queue_chunk(chunk, &msg_tx_play),
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
                engine.wait_until_end();
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

    let tts_engine_clone = tts_engine.clone();
    let (sentence_tts_tx, mut sentence_tts_rx) = mpsc::channel::<String>(4);
    let msg_tx_tts = msg_tx.clone();

    tokio::spawn(async move {
        while let Some(sentence) = sentence_tts_rx.recv().await {
            match tts_client::synthesize(&tts_engine_clone, &sentence).await {
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

    let interrupted = loop {
        tokio::select! {
            Some(chunk) = llm_rx.recv() => {
                match chunk {
                    llm_client::LlmChunk::Token(token) => {
                        send(msg_tx, PipelineMessage::LlmToken { token: token.clone() }).await;
                        full_response.push_str(&token);

                        parse_and_emit_emotion(msg_tx, &full_response).await;

                        // Feed characters through bracket-aware filter.
                        // Emotion tags like [happy] are buffered across tokens and discarded.
                        let tts_text = filter_for_tts(&token, &mut bracket_buf);
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
                        break false;
                    }

                }
            }
            _ = turn_cancel.cancelled() => break true,
        }
    };

    if interrupted {
        // Stop playback immediately
        let _ = stop_tx.send(());
        // Record partial response in conversation history
        if !full_response.is_empty() {
            let truncated = format!("{} [interrupted]", full_response.trim());
            conversation.push(llm_client::Message {
                role: "assistant".to_string(),
                content: llm_client::MessageContent::Text(truncated),
            });
        }
        send(msg_tx, PipelineMessage::LlmDone).await;
    }
}

/// Transcribe a PCM utterance via Whisper STT. Returns `None` on empty/error.
async fn transcribe_utterance(
    whisper_engine: &Arc<WhisperEngine>,
    pcm: Vec<f32>,
    config: &Arc<AppConfig>,
    msg_tx: &mpsc::Sender<PipelineMessage>,
) -> Option<String> {
    send(msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Processing)).await;

    match stt_whisper::transcribe(whisper_engine.clone(), pcm, config.stt_language.clone()).await {
        Ok(t) if t.is_empty() => {
            info!("STT returned empty, skipping");
            send(msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Listening)).await;
            None
        }
        Ok(t) => {
            info!("STT: {}", t);
            send(msg_tx, PipelineMessage::SttResult { text: t.clone() }).await;
            Some(t)
        }
        Err(e) => {
            error!("STT error: {}", e);
            send(msg_tx, PipelineMessage::PipelineError {
                source: "stt".into(),
                message: e.to_string(),
            }).await;
            send(msg_tx, PipelineMessage::PhaseChanged(MascotPhase::Listening)).await;
            None
        }
    }
}

/// Play pre-synthesized interjection audio and wait for it to finish.
async fn play_interjection(pcm: &[f32], msg_tx: &mpsc::Sender<PipelineMessage>) {
    let pcm = pcm.to_vec();
    let msg_tx = msg_tx.clone();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();

    std::thread::spawn(move || {
        match PlaybackEngine::new() {
            Ok(engine) => {
                engine.queue_chunk(pcm, &msg_tx);
                engine.wait_until_end();
            }
            Err(e) => error!("Interjection playback failed: {}", e),
        }
        let _ = done_tx.send(());
    });

    let _ = done_rx.await;
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

/// Filter a streamed token for TTS, accumulating bracket content across tokens.
///
/// `bracket_buf`:
///   - `None` → normal mode
///   - `Some(buf)` → inside `[...]`, accumulating content
///
/// When `]` arrives, the buffered content is checked against known emotion tags.
/// Emotion tags are discarded; unknown brackets are flushed to output.
fn filter_for_tts(token: &str, bracket_buf: &mut Option<String>) -> String {
    let known_emotions: &[&str] = &[
        "happy", "thinking", "surprised", "sad", "neutral",
        "angry", "excited", "confused", "shy", "embarrassed",
    ];

    let mut result = String::new();

    for c in token.chars() {
        if let Some(buf) = bracket_buf {
            if c == ']' {
                // End of bracket — check if it's an emotion tag
                let tag = buf.trim().to_lowercase();
                if !known_emotions.contains(&tag.as_str()) {
                    // Not an emotion tag — flush as normal text
                    result.push('[');
                    result.push_str(buf);
                    result.push(']');
                }
                *bracket_buf = None;
            } else {
                buf.push(c);
            }
        } else if c == '[' {
            *bracket_buf = Some(String::new());
        } else {
            // Normal character — apply emoji/kaomoji stripping
            if !is_emoji(c) {
                result.push(c);
            }
        }
    }

    strip_kaomoji(&result)
}

/// Strip kaomoji in parentheses from text.
fn strip_kaomoji(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '(' | '（' => {
                let close = if c == '(' { ')' } else { '）' };
                let mut group = String::new();
                let mut has_kaomoji_char = false;
                let mut found_close = false;
                for inner in chars.by_ref() {
                    if inner == close {
                        found_close = true;
                        break;
                    }
                    group.push(inner);
                    if is_kaomoji_char(inner) {
                        has_kaomoji_char = true;
                    }
                }
                if !has_kaomoji_char {
                    result.push(c);
                    result.push_str(&group);
                    if found_close {
                        result.push(close);
                    }
                }
            }
            _ => result.push(c),
        }
    }

    result
}

fn is_emoji(c: char) -> bool {
    matches!(c as u32,
        0x1F600..=0x1F64F   // Emoticons
        | 0x1F300..=0x1F5FF // Misc Symbols and Pictographs
        | 0x1F680..=0x1F6FF // Transport
        | 0x1F1E0..=0x1F1FF // Flags
        | 0x2600..=0x26FF   // Misc symbols
        | 0x2700..=0x27BF   // Dingbats
        | 0xFE00..=0xFE0F   // Variation Selectors
        | 0x1F900..=0x1F9FF // Supplemental Symbols
        | 0x1FA00..=0x1FA6F // Chess Symbols
        | 0x1FA70..=0x1FAFF // Symbols Extended-A
    )
}

fn is_kaomoji_char(c: char) -> bool {
    matches!(c,
        'ω' | '∀' | '▽' | '△' | '´' | '`' | '゜' | '°'
        | '＊' | '☆' | '★' | '♪' | '♡' | '♥' | '←' | '→'
        | '↑' | '↓' | '＾' | '＿' | 'ﾉ' | 'ヾ' | 'σ' | 'д'
        | '◕' | '◉' | '●' | '○' | '◎' | '⊂' | '⊃' | '⌒'
        | 'ε' | 'Д' | '∇' | '∩' | '∪' | '╥' | '╹'
    )
}

/// Helper: send a message, logging if the channel is closed.
async fn send(tx: &mpsc::Sender<PipelineMessage>, msg: PipelineMessage) {
    if tx.send(msg).await.is_err() {
        warn!("Pipeline message channel closed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: feed tokens through filter_for_tts and return concatenated result.
    fn feed_tokens(tokens: &[&str]) -> String {
        let mut buf: Option<String> = None;
        let mut out = String::new();
        for t in tokens {
            out.push_str(&filter_for_tts(t, &mut buf));
        }
        out
    }

    #[test]
    fn emotion_single_token() {
        assert_eq!(feed_tokens(&["[happy]"]), "");
        assert_eq!(feed_tokens(&["[sad]"]), "");
    }

    #[test]
    fn emotion_split_across_tokens() {
        // LLM often streams "[", "happy", "]" separately
        assert_eq!(feed_tokens(&["[", "happy", "]"]), "");
        assert_eq!(feed_tokens(&["[think", "ing]"]), "");
        assert_eq!(feed_tokens(&["[", "surprised", "] こんにちは"]), " こんにちは");
    }

    #[test]
    fn emotion_prefix_then_text() {
        assert_eq!(feed_tokens(&["[happy] ", "やっほー！"]), " やっほー！");
    }

    #[test]
    fn unknown_bracket_kept() {
        // Non-emotion brackets should pass through
        assert_eq!(feed_tokens(&["[注意]"]), "[注意]");
        assert_eq!(feed_tokens(&["[", "URL", "]"]), "[URL]");
    }

    #[test]
    fn strip_kaomoji_in_parens() {
        assert_eq!(feed_tokens(&["やったね(´・ω・`)"]), "やったね");
        assert_eq!(feed_tokens(&["(＊∀＊)すごい"]), "すごい");
        assert_eq!(feed_tokens(&["わーい（＾▽＾）！"]), "わーい！");
    }

    #[test]
    fn keep_normal_parens() {
        assert_eq!(feed_tokens(&["明日（あした）"]), "明日（あした）");
        assert_eq!(feed_tokens(&["test (hello)"]), "test (hello)");
    }

    #[test]
    fn strip_emoji() {
        assert_eq!(feed_tokens(&["楽しい😊よ"]), "楽しいよ");
    }

    #[test]
    fn normal_text_unchanged() {
        assert_eq!(feed_tokens(&["こんにちは世界"]), "こんにちは世界");
        assert_eq!(feed_tokens(&["Hello world!"]), "Hello world!");
    }
}
