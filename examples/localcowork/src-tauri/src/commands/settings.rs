//! Tauri IPC commands for the Settings panel.
//!
//! Reads model configuration from `_models/config.yaml` (the same source
//! of truth used by the inference client at runtime) and provides live
//! MCP server status from the running McpClient.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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

            McpServerStatusInfo {
                status: if is_running {
                    "initialized".to_string()
                } else {
                    "failed".to_string()
                },
                tool_count,
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
