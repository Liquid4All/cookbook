//! Tauri IPC commands for the Settings panel.
//!
//! Reads model configuration from `_models/config.yaml` (the same source
//! of truth used by the inference client at runtime) and provides live
//! MCP server status from the running McpClient.

use serde::Serialize;

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
/// In the full integration, this reads from the ToolRouter's PermissionStore.
/// For now, returns a stub (empty list).
#[tauri::command]
pub fn list_permission_grants() -> Result<Vec<PermissionGrantInfo>, String> {
    // Stub: no grants until ToolRouter is fully wired into Tauri state.
    // When integrated, this will read from the ToolRouter's permissions field.
    Ok(vec![])
}

/// Revoke a persistent permission grant by tool name.
///
/// In the full integration, this calls PermissionStore::revoke() on the
/// ToolRouter. For now, returns a stub.
#[tauri::command]
pub fn revoke_permission(tool_name: String) -> Result<bool, String> {
    // Stub: return false (nothing to revoke) until ToolRouter is wired in.
    tracing::info!(tool = %tool_name, "revoke_permission called (stub)");
    Ok(false)
}
