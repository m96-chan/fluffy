use serde_json::Value;
use tracing::{info, warn};

use crate::error::AppError;
use crate::llm::tools::{file_ops, shell_exec};

/// Dispatch a tool call and return the result as a JSON string.
pub async fn dispatch_tool(name: &str, input: &Value) -> Result<String, AppError> {
    info!("Dispatching tool: {}", name);

    match name {
        "read_file" => {
            let path = input["path"]
                .as_str()
                .ok_or_else(|| AppError::Llm("read_file: missing path".to_string()))?;
            let start_line = input["start_line"].as_u64().map(|v| v as usize);
            let end_line = input["end_line"].as_u64().map(|v| v as usize);
            file_ops::read_file(path, start_line, end_line).await
        }

        "write_file" => {
            let path = input["path"]
                .as_str()
                .ok_or_else(|| AppError::Llm("write_file: missing path".to_string()))?;
            let content = input["content"]
                .as_str()
                .ok_or_else(|| AppError::Llm("write_file: missing content".to_string()))?;
            file_ops::write_file(path, content).await
        }

        "list_files" => {
            let directory = input["directory"]
                .as_str()
                .ok_or_else(|| AppError::Llm("list_files: missing directory".to_string()))?;
            let pattern = input["pattern"].as_str();
            file_ops::list_files(directory, pattern).await
        }

        "run_command" => {
            let command = input["command"]
                .as_str()
                .ok_or_else(|| AppError::Llm("run_command: missing command".to_string()))?;
            let cwd = input["cwd"].as_str();
            shell_exec::run_command(command, cwd).await
        }

        unknown => {
            warn!("Unknown tool: {}", unknown);
            Err(AppError::Llm(format!("Unknown tool: {}", unknown)))
        }
    }
}
