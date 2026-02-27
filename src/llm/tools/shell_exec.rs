use tokio::process::Command;
use tracing::{info, warn};

use crate::error::AppError;

const MAX_OUTPUT_BYTES: usize = 64 * 1024; // 64KB output limit
const COMMAND_TIMEOUT_SECS: u64 = 30;

/// Execute a shell command and return its output
pub async fn run_command(command: &str, cwd: Option<&str>) -> Result<String, AppError> {
    info!("Executing command: {}", command);

    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(command);

    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }

    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(COMMAND_TIMEOUT_SECS),
        cmd.output(),
    )
    .await
    .map_err(|_| AppError::Llm(format!("Command timed out after {}s", COMMAND_TIMEOUT_SECS)))?
    .map_err(|e| AppError::Io(e))?;

    let mut result = String::new();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !stdout.is_empty() {
        result.push_str(&stdout[..stdout.len().min(MAX_OUTPUT_BYTES)]);
    }

    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push_str("\n--- stderr ---\n");
        }
        result.push_str(&stderr[..stderr.len().min(MAX_OUTPUT_BYTES / 2)]);
    }

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        result.push_str(&format!("\n[Exit code: {}]", code));
    }

    Ok(result)
}
