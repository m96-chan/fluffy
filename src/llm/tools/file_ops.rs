use std::path::Path;
use tokio::fs;
use tracing::info;

use crate::error::AppError;

const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024; // 10MB
const MAX_LINE_COUNT: usize = 5000;

/// Read file contents, optionally sliced by line range
pub async fn read_file(
    path: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> Result<String, AppError> {
    let path = Path::new(path);

    // Sandbox check: reject paths that look like they escape to sensitive areas
    validate_path(path)?;

    let metadata = fs::metadata(path).await.map_err(|e| AppError::Io(e))?;
    if metadata.len() > MAX_FILE_SIZE {
        return Err(AppError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("File too large ({}MB limit)", MAX_FILE_SIZE / 1024 / 1024),
        )));
    }

    let content = fs::read_to_string(path)
        .await
        .map_err(|e| AppError::Io(e))?;

    if start_line.is_none() && end_line.is_none() {
        // Limit total lines returned
        let lines: Vec<&str> = content.lines().take(MAX_LINE_COUNT).collect();
        return Ok(lines.join("\n"));
    }

    let start = start_line.unwrap_or(1).max(1) - 1; // convert to 0-indexed
    let lines: Vec<&str> = content.lines().collect();
    let end = end_line.unwrap_or(lines.len()).min(lines.len());

    Ok(lines[start..end].join("\n"))
}

/// Write content to file
pub async fn write_file(path: &str, content: &str) -> Result<String, AppError> {
    let path = Path::new(path);

    validate_path(path)?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|e| AppError::Io(e))?;
    }

    fs::write(path, content)
        .await
        .map_err(|e| AppError::Io(e))?;

    info!("Wrote {} bytes to {:?}", content.len(), path);
    Ok(format!("Successfully wrote {} bytes to {}", content.len(), path.display()))
}

/// List files in directory
pub async fn list_files(directory: &str, pattern: Option<&str>) -> Result<String, AppError> {
    let path = Path::new(directory);
    validate_path(path)?;

    let mut entries = fs::read_dir(path).await.map_err(|e| AppError::Io(e))?;
    let mut files = Vec::new();

    while let Some(entry) = entries.next_entry().await.map_err(|e| AppError::Io(e))? {
        let file_name = entry.file_name().to_string_lossy().to_string();

        if let Some(pat) = pattern {
            // Simple glob: check if filename matches the pattern
            if !simple_glob_match(pat, &file_name) {
                continue;
            }
        }

        let metadata = entry.metadata().await.map_err(|e| AppError::Io(e))?;
        let prefix = if metadata.is_dir() { "d" } else { "f" };
        files.push(format!("{} {}", prefix, file_name));
    }

    files.sort();
    Ok(files.join("\n"))
}

fn validate_path(path: &Path) -> Result<(), AppError> {
    // Reject paths that contain suspicious patterns
    let path_str = path.to_string_lossy();

    // Block absolute paths to sensitive system directories
    let blocked_prefixes = ["/etc/", "/proc/", "/sys/", "/dev/"];
    for prefix in &blocked_prefixes {
        if path_str.starts_with(prefix) {
            return Err(AppError::SandboxViolation(path_str.to_string()));
        }
    }

    // Block path traversal that escapes too far
    let canonical_parts: Vec<_> = path.components().collect();
    let mut depth: i32 = 0;
    for part in &canonical_parts {
        match part {
            std::path::Component::ParentDir => {
                depth -= 1;
                if depth < -2 {
                    return Err(AppError::SandboxViolation(path_str.to_string()));
                }
            }
            std::path::Component::Normal(_) => depth += 1,
            _ => {}
        }
    }

    Ok(())
}

fn simple_glob_match(pattern: &str, name: &str) -> bool {
    // Very simple glob: only handles leading/trailing *
    if pattern == "*" {
        return true;
    }
    if let Some(suffix) = pattern.strip_prefix("*.") {
        return name.ends_with(&format!(".{}", suffix));
    }
    if let Some(prefix) = pattern.strip_suffix("*") {
        return name.starts_with(prefix);
    }
    pattern == name
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn read_file_returns_contents() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.txt");
        tokio::fs::write(&path, "hello world\nline 2").await.unwrap();

        let result = read_file(path.to_str().unwrap(), None, None)
            .await
            .unwrap();
        assert!(result.contains("hello world"));
    }

    #[tokio::test]
    async fn write_file_creates_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("out.txt");

        let result = write_file(path.to_str().unwrap(), "test content")
            .await
            .unwrap();
        assert!(result.contains("12 bytes"));

        let contents = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(contents, "test content");
    }

    #[tokio::test]
    async fn read_file_outside_sandbox_rejected() {
        let result = read_file("/etc/passwd", None, None).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AppError::SandboxViolation(_)));
    }

    #[test]
    fn glob_match_extension() {
        assert!(simple_glob_match("*.rs", "main.rs"));
        assert!(!simple_glob_match("*.rs", "main.ts"));
    }

    #[test]
    fn glob_match_wildcard() {
        assert!(simple_glob_match("*", "anything"));
    }
}
