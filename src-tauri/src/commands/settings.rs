//! Tauri IPC commands for the Settings panel.
//!
//! Reads model configuration from `_models/config.yaml` (the same source
//! of truth used by the inference client at runtime) and provides live
//! MCP server status from the running McpClient.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

static SETTINGS_CHANGED: AtomicBool = AtomicBool::new(false);

pub fn settings_changed() {
    SETTINGS_CHANGED.store(true, Ordering::SeqCst);
}

pub fn has_settings_changed() -> bool {
    SETTINGS_CHANGED.swap(false, Ordering::SeqCst)
}

/// Unified app settings that persist across restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    /// Currently active model key from _models/config.yaml
    pub active_model_key: Option<String>,
    /// Allowed filesystem paths for sandboxed operations
    pub allowed_paths: Vec<String>,
    /// UI theme preference
    pub theme: String,
    /// Whether to show tool traces
    pub show_tool_traces: bool,
    /// Sampling config (integrated from existing system)
    pub sampling: SamplingConfig,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            active_model_key: None,
            allowed_paths: Vec::new(),
            theme: "system".to_string(),
            show_tool_traces: true,
            sampling: SamplingConfig::default(),
        }
        // Default allowed paths
    }
}

impl AppSettings {
    const FILE_NAME: &'static str = "settings.json";

    fn persist_path() -> PathBuf {
        crate::data_dir().join(Self::FILE_NAME)
    }

    pub fn load_or_default() -> Self {
        let path = Self::persist_path();
        if !path.exists() {
            return Self::default();
        }
        match std::fs::read_to_string(&path) {
            Ok(content) => match serde_json::from_str::<Self>(&content) {
                Ok(mut settings) => {
                    // Validate and clean up non-existent paths
                    let original_count = settings.allowed_paths.len();
                    settings.allowed_paths.retain(|p| {
                        std::path::PathBuf::from(p).exists()
                    });
                    let removed = original_count - settings.allowed_paths.len();
                    if removed > 0 {
                        tracing::warn!(removed_paths = removed, "removed non-existent allowed paths");
                        settings.save();
                    }
                    tracing::info!(path = %path.display(), "loaded app settings");
                    settings
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to parse settings, using defaults");
                    Self::default()
                }
            },
            Err(e) => {
                tracing::warn!(error = %e, "failed to read settings, using defaults");
                Self::default()
            }
        }
    }

    pub fn save(&self) {
        let path = Self::persist_path();
        let content = match serde_json::to_string_pretty(self) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(error = %e, "failed to serialize settings");
                return;
            }
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp_path = path.with_extension("json.tmp");
        if let Err(e) = std::fs::write(&tmp_path, &content) {
            tracing::error!(error = %e, "failed to write settings temp file");
            return;
        }
        if let Err(e) = std::fs::rename(&tmp_path, &path) {
            tracing::error!(error = %e, "failed to rename settings file");
            return;
        }
        settings_changed();
        tracing::debug!("saved app settings");
    }

    pub fn export_to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("export failed: {}", e))
    }

    pub fn import_from_json(json: &str) -> Result<Self, String> {
        let settings: Self =
            serde_json::from_str(json).map_err(|e| format!("invalid settings JSON: {}", e))?;
        settings.sampling.save();
        settings.save();
        Ok(settings)
    }
}

/// Model configuration exposed to the frontend.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelConfigInfo {
    pub key: String,
    pub display_name: String,
    pub runtime: String,
    pub base_url: String,
    pub context_window: u32,
    pub temperature: f64,
    pub max_tokens: u32,
    pub estimated_vram_gb: Option<f64>,
    pub capabilities: Vec<String>,
    pub tool_call_format: String,
}

/// Models overview returned to the frontend.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsOverviewInfo {
    pub active_model: String,
    pub models: Vec<ModelConfigInfo>,
    pub fallback_chain: Vec<String>,
}

/// MCP server status.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerStatusInfo {
    pub name: String,
    pub status: String,
    pub tool_count: u32,
    pub tool_names: Vec<String>,
    pub last_check: String,
    pub error: Option<String>,
}

/// Get the models configuration overview.
///
/// Reads from `_models/config.yaml` using the same config loader
/// that the inference client uses at runtime.
#[tauri::command]
pub fn get_models_config() -> Result<ModelsOverviewInfo, String> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let config_path = crate::inference::config::find_config_path(&cwd)
        .map_err(|e| format!("Config not found: {e}"))?;
    let config = crate::inference::config::load_models_config(&config_path)
        .map_err(|e| format!("Config load error: {e}"))?;

    let models: Vec<ModelConfigInfo> = config
        .models
        .iter()
        .map(|(key, m)| ModelConfigInfo {
            key: key.clone(),
            display_name: m.display_name.clone(),
            runtime: m.runtime.clone(),
            base_url: m.base_url.clone(),
            context_window: m.context_window,
            temperature: f64::from(m.temperature),
            max_tokens: m.max_tokens,
            estimated_vram_gb: m.estimated_vram_gb.map(f64::from),
            capabilities: m.capabilities.clone(),
            tool_call_format: format!("{:?}", m.tool_call_format),
        })
        .collect();

    Ok(ModelsOverviewInfo {
        active_model: config.active_model.clone(),
        models,
        fallback_chain: config.fallback_chain.clone(),
    })
}

/// Get the status of all MCP servers from the running McpClient.
///
/// Queries actual server state — no hardcoded stubs. Returns configured
/// servers with their running status and tool count.
#[tauri::command]
pub async fn get_mcp_servers_status(
    mcp_state: tauri::State<'_, crate::TokioMutex<crate::mcp_client::McpClient>>,
) -> Result<Vec<McpServerStatusInfo>, String> {
    let mcp = mcp_state.lock().await;
    let now = chrono::Utc::now().to_rfc3339();

    let configured = mcp.configured_servers();
    let mut statuses: Vec<McpServerStatusInfo> = configured
        .into_iter()
        .map(|name| {
            let is_running = mcp.is_server_running(&name);
            let tool_count = mcp.registry.tools_for_server(&name) as u32;
            let tool_names = mcp.registry.tool_names_for_server(&name);

            McpServerStatusInfo {
                status: if is_running {
                    "initialized".to_string()
                } else {
                    "failed".to_string()
                },
                tool_count,
                tool_names,
                last_check: now.clone(),
                error: if is_running {
                    None
                } else {
                    Some("Server not running".to_string())
                },
                name,
            }
        })
        .collect();

    statuses.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(statuses)
}

// ─── Permission Grant Management ────────────────────────────────────────────

/// A permission grant exposed to the frontend.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionGrantInfo {
    pub tool_name: String,
    pub scope: String,
    pub granted_at: String,
}

/// List all persistent permission grants.
///
/// Reads from the PermissionStore in Tauri state.
#[tauri::command]
pub async fn list_permission_grants(
    perms: tauri::State<'_, crate::TokioMutex<crate::agent_core::PermissionStore>>,
) -> Result<Vec<PermissionGrantInfo>, String> {
    let store = perms.lock().await;
    let grants = store
        .list_persistent()
        .into_iter()
        .map(|g| PermissionGrantInfo {
            tool_name: g.tool_name.clone(),
            scope: format!("{:?}", g.scope).to_lowercase(),
            granted_at: g.granted_at.clone(),
        })
        .collect();
    Ok(grants)
}

/// Revoke a persistent permission grant by tool name.
///
/// Removes the grant from the PermissionStore and persists the change to disk.
#[tauri::command]
pub async fn revoke_permission(
    tool_name: String,
    perms: tauri::State<'_, crate::TokioMutex<crate::agent_core::PermissionStore>>,
) -> Result<bool, String> {
    let mut store = perms.lock().await;
    let removed = store.revoke(&tool_name);
    tracing::info!(tool = %tool_name, removed, "revoke_permission");
    Ok(removed)
}

// ─── Sampling Configuration ─────────────────────────────────────────────────

/// Runtime sampling hyperparameters exposed to the frontend.
///
/// Persisted to `sampling_config.json` in the app data directory.
/// The agent loop reads these at the start of each `send_message` call
/// instead of using hardcoded constants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplingConfig {
    pub tool_temperature: f32,
    pub tool_top_p: f32,
    pub conversational_temperature: f32,
    pub conversational_top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            tool_temperature: 0.1,
            tool_top_p: 0.2,
            conversational_temperature: 0.7,
            conversational_top_p: 0.9,
        }
    }
}

impl SamplingConfig {
    /// Load from disk or return defaults.
    pub fn load_or_default() -> Self {
        let path = Self::persist_path();
        if !path.exists() {
            return Self::default();
        }
        match std::fs::read_to_string(&path) {
            Ok(content) => match serde_json::from_str::<Self>(&content) {
                Ok(cfg) => {
                    tracing::info!(path = %path.display(), "loaded sampling config");
                    cfg
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to parse sampling config, using defaults");
                    Self::default()
                }
            },
            Err(e) => {
                tracing::warn!(error = %e, "failed to read sampling config, using defaults");
                Self::default()
            }
        }
    }

    /// Save to disk (atomic write).
    pub fn save(&self) {
        let path = Self::persist_path();
        let content = match serde_json::to_string_pretty(self) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(error = %e, "failed to serialize sampling config");
                return;
            }
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp_path = path.with_extension("json.tmp");
        if let Err(e) = std::fs::write(&tmp_path, &content) {
            tracing::error!(error = %e, "failed to write sampling config temp file");
            return;
        }
        if let Err(e) = std::fs::rename(&tmp_path, &path) {
            tracing::error!(error = %e, "failed to rename sampling config file");
            return;
        }
        tracing::debug!("saved sampling config");
    }

    fn persist_path() -> PathBuf {
        crate::data_dir().join("sampling_config.json")
    }
}

/// Get the current sampling configuration.
#[tauri::command]
pub async fn get_sampling_config(
    state: tauri::State<'_, crate::TokioMutex<SamplingConfig>>,
) -> Result<SamplingConfig, String> {
    let cfg = state.lock().await;
    Ok(cfg.clone())
}

/// Update the sampling configuration and persist to disk.
#[tauri::command]
pub async fn update_sampling_config(
    config: SamplingConfig,
    state: tauri::State<'_, crate::TokioMutex<SamplingConfig>>,
) -> Result<SamplingConfig, String> {
    let mut cfg = state.lock().await;
    *cfg = config;
    cfg.save();
    tracing::info!(
        tool_temp = cfg.tool_temperature,
        tool_top_p = cfg.tool_top_p,
        conv_temp = cfg.conversational_temperature,
        conv_top_p = cfg.conversational_top_p,
        "sampling config updated"
    );
    Ok(cfg.clone())
}

/// Reset the sampling configuration to defaults and persist.
#[tauri::command]
pub async fn reset_sampling_config(
    state: tauri::State<'_, crate::TokioMutex<SamplingConfig>>,
) -> Result<SamplingConfig, String> {
    let mut cfg = state.lock().await;
    *cfg = SamplingConfig::default();
    cfg.save();
    tracing::info!("sampling config reset to defaults");
    Ok(cfg.clone())
}

// ─── Unified App Settings ────────────────────────────────────────────────────

/// Get the current app settings.
#[tauri::command]
pub fn get_app_settings() -> AppSettings {
    AppSettings::load_or_default()
}

/// Update app settings and persist to disk.
#[tauri::command]
pub fn update_app_settings(settings: AppSettings) -> AppSettings {
    settings.save();
    tracing::info!(
        active_model = ?settings.active_model_key,
        theme = %settings.theme,
        allowed_paths = settings.allowed_paths.len(),
        "app settings updated"
    );
    settings
}

/// Add an allowed path to settings.
#[tauri::command]
pub fn add_allowed_path(path: String) -> Result<AppSettings, String> {
    // Validate path exists or can be created
    let path_buf = std::path::PathBuf::from(&path);
    if !path_buf.exists() {
        // Check if parent exists (to allow creating new directories)
        if let Some(parent) = path_buf.parent() {
            if !parent.exists() {
                return Err(format!("Parent directory does not exist: {}", parent.display()));
            }
        } else {
            return Err("Invalid path".to_string());
        }
    }
    
    let mut settings = AppSettings::load_or_default();
    if !settings.allowed_paths.contains(&path) {
        settings.allowed_paths.push(path.clone());
        settings.save();
        tracing::info!(path = %path, "allowed path added");
    }
    Ok(settings)
}

/// Remove an allowed path from settings.
#[tauri::command]
pub fn remove_allowed_path(path: String) -> AppSettings {
    let mut settings = AppSettings::load_or_default();
    let path_clone = path.clone();
    settings.allowed_paths.retain(|p| p != &path);
    settings.save();
    tracing::info!(path = %path_clone, "allowed path removed");
    settings
}

/// Export settings to JSON string.
#[tauri::command]
pub fn export_settings() -> Result<String, String> {
    let settings = AppSettings::load_or_default();
    settings.export_to_json()
}

/// Import settings from JSON string.
#[tauri::command]
pub fn import_settings(json: String) -> Result<AppSettings, String> {
    AppSettings::import_from_json(&json)
}

/// Check if settings have changed since last check (for file watching).
#[tauri::command]
pub fn poll_settings_changed() -> bool {
    has_settings_changed()
}

// ─── Config Hot Reload ──────────────────────────────────────────────────────

use std::sync::atomic::AtomicU64;
use std::time::SystemTime;

static CONFIG_LAST_MODIFIED: AtomicU64 = AtomicU64::new(0);

/// Check if config file has been modified since last check.
#[tauri::command]
pub fn check_config_reload() -> Result<bool, String> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let config_path = crate::inference::config::find_config_path(&cwd)
        .map_err(|e| format!("Config not found: {e}"))?;
    
    let metadata = std::fs::metadata(&config_path)
        .map_err(|e| format!("Failed to read config metadata: {}", e))?;
    
    let modified = metadata.modified()
        .map_err(|e| format!("Failed to get modification time: {}", e))?;
    
    let modified_secs = modified
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|e| format!("Time error: {}", e))?
        .as_secs();
    
    let last_modified = CONFIG_LAST_MODIFIED.load(Ordering::SeqCst);
    
    if modified_secs > last_modified {
        CONFIG_LAST_MODIFIED.store(modified_secs, Ordering::SeqCst);
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Force reload the model config (for manual refresh).
#[tauri::command]
pub fn reload_model_config() -> Result<ModelsOverviewInfo, String> {
    get_models_config()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_sampling_config_default() {
        let cfg = SamplingConfig::default();
        assert_eq!(cfg.tool_temperature, 0.1);
        assert_eq!(cfg.tool_top_p, 0.2);
        assert_eq!(cfg.conversational_temperature, 0.7);
        assert_eq!(cfg.conversational_top_p, 0.9);
    }

    #[test]
    fn test_sampling_config_serialization() {
        let cfg = SamplingConfig {
            tool_temperature: 0.5,
            tool_top_p: 0.3,
            conversational_temperature: 0.8,
            conversational_top_p: 0.95,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("0.5"));
        assert!(json.contains("0.3"));
        assert!(json.contains("0.8"));
        assert!(json.contains("0.95"));
    }

    #[test]
    fn test_sampling_config_deserialization() {
        let json = r#"{
            "toolTemperature": 0.3,
            "toolTopP": 0.4,
            "conversationalTemperature": 0.6,
            "conversationalTopP": 0.8
        }"#;
        let cfg: SamplingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.tool_temperature, 0.3);
        assert_eq!(cfg.tool_top_p, 0.4);
        assert_eq!(cfg.conversational_temperature, 0.6);
        assert_eq!(cfg.conversational_top_p, 0.8);
    }

    #[test]
    fn test_app_settings_default() {
        let settings = AppSettings::default();
        assert_eq!(settings.active_model_key, None);
        assert!(settings.allowed_paths.is_empty());
        assert_eq!(settings.theme, "system");
        assert!(settings.show_tool_traces);
        // Sampling should be default
        assert_eq!(settings.sampling.tool_temperature, 0.1);
    }

    #[test]
    fn test_app_settings_serialization() {
        let mut settings = AppSettings::default();
        settings.active_model_key = Some("test-model".to_string());
        settings.allowed_paths = vec!["/home/user/docs".to_string()];
        settings.theme = "dark".to_string();
        settings.show_tool_traces = false;

        let json = serde_json::to_string(&settings).unwrap();
        assert!(json.contains("test-model"));
        assert!(json.contains("dark"));
        assert!(json.contains("docs"));
    }

    #[test]
    fn test_app_settings_deserialization() {
        let json = r#"{
            "activeModelKey": "lm-studio-model",
            "allowedPaths": ["/tmp", "/var"],
            "theme": "light",
            "showToolTraces": false,
            "sampling": {
                "toolTemperature": 0.2,
                "toolTopP": 0.3,
                "conversationalTemperature": 0.8,
                "conversationalTopP": 0.9
            }
        }"#;
        let settings: AppSettings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.active_model_key, Some("lm-studio-model".to_string()));
        assert_eq!(settings.allowed_paths.len(), 2);
        assert_eq!(settings.theme, "light");
        assert!(!settings.show_tool_traces);
    }

    #[test]
    fn test_config_last_modified_atomic() {
        // Test that CONFIG_LAST_MODIFIED is properly initialized
        let initial = CONFIG_LAST_MODIFIED.load(Ordering::SeqCst);
        assert_eq!(initial, 0);

        // Store a value and verify
        CONFIG_LAST_MODIFIED.store(12345, Ordering::SeqCst);
        let after = CONFIG_LAST_MODIFIED.load(Ordering::SeqCst);
        assert_eq!(after, 12345);

        // Reset
        CONFIG_LAST_MODIFIED.store(0, Ordering::SeqCst);
    }

    #[test]
    fn test_settings_changed_atomic() {
        // Test the SETTINGS_CHANGED flag
        settings_changed();
        assert!(has_settings_changed());
        assert!(!has_settings_changed()); // Should clear after check

        // Setting it again should work
        settings_changed();
        assert!(has_settings_changed());
    }

    #[test]
    fn test_model_config_info_fields() {
        let info = ModelConfigInfo {
            key: "test-key".to_string(),
            display_name: "Test Model".to_string(),
            runtime: "lm-studio".to_string(),
            base_url: "http://localhost:1234/v1".to_string(),
            context_window: 32768,
            temperature: 0.7,
            max_tokens: 4096,
            estimated_vram_gb: Some(24.0),
            capabilities: vec!["chat".to_string(), "tools".to_string()],
            tool_call_format: "json".to_string(),
        };

        assert_eq!(info.key, "test-key");
        assert_eq!(info.runtime, "lm-studio");
        assert_eq!(info.context_window, 32768);
    }

    #[test]
    fn test_models_overview_info_serialization() {
        let overview = ModelsOverviewInfo {
            active_model: "qwen2.5".to_string(),
            models: vec![
                ModelConfigInfo {
                    key: "qwen2.5".to_string(),
                    display_name: "Qwen 2.5".to_string(),
                    runtime: "ollama".to_string(),
                    base_url: "http://localhost:11434/v1".to_string(),
                    context_window: 32768,
                    temperature: 0.7,
                    max_tokens: 4096,
                    estimated_vram_gb: Some(20.0),
                    capabilities: vec!["chat".to_string()],
                    tool_call_format: "json".to_string(),
                }
            ],
            fallback_chain: vec!["gpt-oss".to_string()],
        };

        let json = serde_json::to_string(&overview).unwrap();
        assert!(json.contains("qwen2.5"));
        assert!(json.contains("ollama"));
    }

    #[test]
    fn test_mcp_server_status_info() {
        let status = McpServerStatusInfo {
            name: "filesystem".to_string(),
            status: "initialized".to_string(),
            tool_count: 10,
            tool_names: vec!["list_dir".to_string(), "read_file".to_string()],
            last_check: "2024-01-01T00:00:00Z".to_string(),
            error: None,
        };

        assert_eq!(status.name, "filesystem");
        assert_eq!(status.status, "initialized");
        assert_eq!(status.tool_count, 10);

        // Test with error
        let status_with_error = McpServerStatusInfo {
            error: Some("Connection refused".to_string()),
            ..status
        };
        assert!(status_with_error.error.is_some());
    }

    #[test]
    fn test_permission_grant_info() {
        let grant = PermissionGrantInfo {
            tool_name: "filesystem.write_file".to_string(),
            scope: "session".to_string(),
            granted_at: "2024-01-01T12:00:00Z".to_string(),
        };

        assert_eq!(grant.tool_name, "filesystem.write_file");
        assert_eq!(grant.scope, "session");
    }

    #[test]
    fn test_app_settings_export_import_roundtrip() {
        let original = AppSettings {
            active_model_key: Some("test-model".to_string()),
            allowed_paths: vec!["/home/user".to_string()],
            theme: "dark".to_string(),
            show_tool_traces: true,
            sampling: SamplingConfig {
                tool_temperature: 0.15,
                tool_top_p: 0.25,
                conversational_temperature: 0.75,
                conversational_top_p: 0.85,
            },
        };

        let json = original.export_to_json().unwrap();
        let imported = AppSettings::import_from_json(&json).unwrap();

        assert_eq!(imported.active_model_key, original.active_model_key);
        assert_eq!(imported.allowed_paths, original.allowed_paths);
        assert_eq!(imported.theme, original.theme);
        assert_eq!(imported.sampling.tool_temperature, original.sampling.tool_temperature);
    }
}
