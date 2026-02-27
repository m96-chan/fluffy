/// Window configuration: position, passthrough mode.
/// Persisted to ~/.config/fluffy/window.toml

use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Resource)]
pub struct WindowConfig {
    /// Window X position in physical pixels (top-left corner).
    pub x: i32,
    /// Window Y position in physical pixels.
    pub y: i32,
    /// Whether click-through (hit-test passthrough) is enabled.
    pub click_through: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            x: 100,
            y: 100,
            click_through: false, // default: interactive (hit_test=true). Press C to enable click-through.
        }
    }
}

impl WindowConfig {
    /// Path to the config file.
    pub fn config_path() -> PathBuf {
        dirs_next::config_dir()
            .unwrap_or_else(|| PathBuf::from("~/.config"))
            .join("fluffy")
            .join("window.toml")
    }

    /// Load from file, or return default if not found / parse error.
    pub fn load() -> Self {
        let path = Self::config_path();
        match std::fs::read_to_string(&path) {
            Ok(contents) => toml::from_str(&contents).unwrap_or_else(|e| {
                warn!("Failed to parse window config: {} — using defaults", e);
                Self::default()
            }),
            Err(_) => {
                info!("No window config found at {:?} — using defaults", path);
                Self::default()
            }
        }
    }

    /// Save to file, creating parent directories as needed.
    pub fn save(&self) -> Result<(), String> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let contents = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, contents).map_err(|e| e.to_string())?;
        info!("Window config saved to {:?}", path);
        Ok(())
    }

    /// Save with a custom path (for testing).
    pub fn save_to(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let contents = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, contents).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Load from a custom path (for testing).
    pub fn load_from(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => toml::from_str(&contents).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let cfg = WindowConfig::default();
        assert_eq!(cfg.x, 100);
        assert_eq!(cfg.y, 100);
        assert!(!cfg.click_through); // default: interactive
    }

    #[test]
    fn serialize_roundtrip() {
        let cfg = WindowConfig { x: 200, y: 300, click_through: false };
        let toml_str = toml::to_string_pretty(&cfg).unwrap();
        let restored: WindowConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(cfg, restored);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let cfg = WindowConfig { x: 42, y: 99, click_through: false };
        cfg.save_to(tmp.path()).unwrap();
        let loaded = WindowConfig::load_from(tmp.path());
        assert_eq!(cfg, loaded);
    }

    #[test]
    fn load_from_missing_file_returns_default() {
        let loaded = WindowConfig::load_from(std::path::Path::new("/nonexistent/path.toml"));
        assert_eq!(loaded, WindowConfig::default());
    }

    #[test]
    fn load_from_corrupt_file_returns_default() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "NOT VALID TOML ][[[").unwrap();
        let loaded = WindowConfig::load_from(tmp.path());
        assert_eq!(loaded, WindowConfig::default());
    }

    #[test]
    fn config_path_contains_fluffy() {
        let path = WindowConfig::config_path();
        assert!(path.to_string_lossy().contains("fluffy"));
        assert!(path.to_string_lossy().ends_with("window.toml"));
    }

    #[test]
    fn save_creates_parent_dirs() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("subdir").join("window.toml");
        let cfg = WindowConfig::default();
        cfg.save_to(&path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn click_through_default_false() {
        // Default is interactive (hit_test=true). Press C in-app to enable click-through.
        assert!(!WindowConfig::default().click_through);
    }

    #[test]
    fn position_update() {
        let mut cfg = WindowConfig::default();
        cfg.x = 500;
        cfg.y = 250;
        assert_eq!(cfg.x, 500);
        assert_eq!(cfg.y, 250);
    }
}
