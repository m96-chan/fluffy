use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::error::AppError;

const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Debug, Clone)]
pub enum LlmChunk {
    Token(String),
    Sentence(String),
    ToolCall { id: String, name: String, input: Value },
    Done,
    Error(String),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
}

/// Stream LLM tokens from Claude API.
/// Returns a channel of LlmChunk events.
pub async fn stream_completion(
    api_key: &str,
    api_url: &str,
    model: &str,
    system: &str,
    messages: Vec<Message>,
    tools: Vec<Value>,
    tx: mpsc::Sender<LlmChunk>,
) -> Result<(), AppError> {
    let client = Client::new();

    let body = json!({
        "model": model,
        "max_tokens": 4096,
        "stream": true,
        "system": system,
        "messages": messages,
        "tools": tools,
    });

    let response = client
        .post(api_url)
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| AppError::Llm(format!("Request failed: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let _body = response.text().await.unwrap_or_default();
        let user_msg = match status.as_u16() {
            401 => "通信に失敗しました。認証エラーです。",
            429 => "通信に失敗しました。リクエストが多すぎます。少し待ってください。",
            500..=599 => "通信に失敗しました。サーバーエラーです。",
            _ => "通信に失敗しました。",
        };
        let _ = tx.send(LlmChunk::Token(format!("[neutral] {user_msg}"))).await;
        let _ = tx.send(LlmChunk::Done).await;
        return Ok(());
    }

    parse_sse_stream(response, tx).await
}

async fn parse_sse_stream(
    response: reqwest::Response,
    tx: mpsc::Sender<LlmChunk>,
) -> Result<(), AppError> {
    use tokio_stream::StreamExt;

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    // Track active tool use block being built
    let mut current_tool_id: Option<String> = None;
    let mut current_tool_name: Option<String> = None;
    let mut current_tool_input: Option<String> = None;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| AppError::Llm(format!("Stream error: {}", e)))?;
        let text = String::from_utf8_lossy(&chunk);
        buffer.push_str(&text);

        // Process complete SSE lines
        while let Some(pos) = buffer.find("\n\n") {
            let event_str = buffer[..pos].to_string();
            buffer.drain(..pos + 2);

            if let Some(chunk) = parse_sse_event(&event_str) {
                match process_sse_event(
                    &chunk,
                    &mut current_tool_id,
                    &mut current_tool_name,
                    &mut current_tool_input,
                ) {
                    Some(LlmChunk::Done) => {
                        let _ = tx.send(LlmChunk::Done).await;
                        return Ok(());
                    }
                    Some(c) => {
                        if tx.send(c).await.is_err() {
                            return Ok(()); // receiver dropped
                        }
                    }
                    None => {}
                }
            }
        }
    }

    let _ = tx.send(LlmChunk::Done).await;
    Ok(())
}

fn parse_sse_event(event_str: &str) -> Option<Value> {
    let mut event_type = None;
    let mut data = None;

    for line in event_str.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            event_type = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data: ") {
            data = Some(rest.trim().to_string());
        }
    }

    let data_str = data?;
    if data_str == "[DONE]" {
        return None;
    }

    serde_json::from_str(&data_str).ok()
}

fn process_sse_event(
    event: &Value,
    current_tool_id: &mut Option<String>,
    current_tool_name: &mut Option<String>,
    current_tool_input: &mut Option<String>,
) -> Option<LlmChunk> {
    let event_type = event.get("type")?.as_str()?;

    match event_type {
        "content_block_start" => {
            let block = event.get("content_block")?;
            let block_type = block.get("type")?.as_str()?;

            if block_type == "tool_use" {
                *current_tool_id = block.get("id").and_then(|v| v.as_str()).map(String::from);
                *current_tool_name =
                    block.get("name").and_then(|v| v.as_str()).map(String::from);
                *current_tool_input = Some(String::new());
            }
            None
        }

        "content_block_delta" => {
            let delta = event.get("delta")?;
            let delta_type = delta.get("type")?.as_str()?;

            match delta_type {
                "text_delta" => {
                    let text = delta.get("text")?.as_str()?;
                    Some(LlmChunk::Token(text.to_string()))
                }
                "input_json_delta" => {
                    // Accumulate tool input JSON
                    if let Some(ref mut input) = current_tool_input {
                        if let Some(partial) = delta.get("partial_json").and_then(|v| v.as_str()) {
                            input.push_str(partial);
                        }
                    }
                    None
                }
                _ => None,
            }
        }

        "content_block_stop" => {
            // If we were building a tool use block, emit it now
            if let (Some(id), Some(name), Some(input_str)) = (
                current_tool_id.take(),
                current_tool_name.take(),
                current_tool_input.take(),
            ) {
                let input: Value = serde_json::from_str(&input_str).unwrap_or(Value::Null);
                return Some(LlmChunk::ToolCall { id, name, input });
            }
            None
        }

        "message_stop" => Some(LlmChunk::Done),

        _ => None,
    }
}

/// Build the standard tool definitions for the pipeline
pub fn make_tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "read_file",
            "description": "Read the contents of a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to file"},
                    "start_line": {"type": "integer", "description": "Start line (1-indexed, optional)"},
                    "end_line": {"type": "integer", "description": "End line (inclusive, optional)"}
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "write_file",
            "description": "Write content to a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }),
        json!({
            "name": "list_files",
            "description": "List files in a directory",
            "input_schema": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory path"},
                    "pattern": {"type": "string", "description": "Glob pattern (optional)"}
                },
                "required": ["directory"]
            }
        }),
        json!({
            "name": "run_command",
            "description": "Run a shell command (requires user approval)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "cwd": {"type": "string", "description": "Working directory (optional)"}
                },
                "required": ["command"]
            }
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_SSE: &str = r#"event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_stop
data: {"type":"message_stop"}

"#;

    const FIXTURE_TOOL_SSE: &str = r#"event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"read_file","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"test.txt\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_stop
data: {"type":"message_stop"}

"#;

    fn collect_chunks_from_fixture(fixture: &str) -> Vec<LlmChunk> {
        let mut chunks = Vec::new();
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let mut current_tool_input = None;

        for event_block in fixture.split("\n\n").filter(|s| !s.trim().is_empty()) {
            if let Some(event) = parse_sse_event(event_block) {
                if let Some(chunk) = process_sse_event(
                    &event,
                    &mut current_tool_id,
                    &mut current_tool_name,
                    &mut current_tool_input,
                ) {
                    chunks.push(chunk);
                }
            }
        }

        chunks
    }

    #[test]
    fn sse_text_tokens_extracted_correctly() {
        let chunks = collect_chunks_from_fixture(FIXTURE_SSE);
        let tokens: Vec<&str> = chunks
            .iter()
            .filter_map(|c| match c {
                LlmChunk::Token(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(tokens, vec!["Hello", " world"]);
    }

    #[test]
    fn sse_message_stop_signals_done() {
        let chunks = collect_chunks_from_fixture(FIXTURE_SSE);
        assert!(chunks.iter().any(|c| matches!(c, LlmChunk::Done)));
    }

    #[test]
    fn sse_tool_use_block_parsed() {
        let chunks = collect_chunks_from_fixture(FIXTURE_TOOL_SSE);
        let tool_call = chunks
            .iter()
            .find(|c| matches!(c, LlmChunk::ToolCall { .. }));

        assert!(tool_call.is_some(), "Should have a tool call");
        if let Some(LlmChunk::ToolCall { id, name, input }) = tool_call {
            assert_eq!(id, "toolu_01");
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "test.txt");
        }
    }
}
