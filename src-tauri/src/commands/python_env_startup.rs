//! Startup-time Python venv provisioning (called from `lib.rs`).
//!
//! Unlike the IPC commands in `python_env.rs`, this runs before the frontend
//! is connected and logs progress to `agent.log` instead of emitting events.

use std::path::{Path, PathBuf};
use tokio::process::Command;

use super::python_env::{find_system_python, is_venv_ready, pip_executable, venv_bin_dir};

// ─── Discovery ──────────────────────────────────────────────────────────────

/// Find all Python MCP server directories (those with `pyproject.toml`).
fn discover_python_servers(project_root: &Path) -> Vec<(String, PathBuf)> {
    let mcp_dir = project_root.join("mcp-servers");
    let mut servers = Vec::new();

    let entries = match std::fs::read_dir(&mcp_dir) {
        Ok(e) => e,
        Err(_) => return servers,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        if name.starts_with('_') || name.starts_with('.') {
            continue;
        }

        if path.join("pyproject.toml").exists() {
            servers.push((name, path));
        }
    }

    servers
}

// ─── Startup Provisioning ───────────────────────────────────────────────────

/// Provision missing Python venvs at app startup.
///
/// Scans `mcp-servers/` for Python servers without `.venv` directories,
/// creates venvs and installs dependencies. Idempotent — skips servers
/// that already have a working venv.
pub async fn provision_missing_venvs(project_root: &Path) {
    let servers = discover_python_servers(project_root);

    let missing: Vec<_> = servers
        .iter()
        .filter(|(_, dir)| !is_venv_ready(dir))
        .collect();

    if missing.is_empty() {
        tracing::info!("all Python server venvs already provisioned");
        return;
    }

    tracing::info!(
        count = missing.len(),
        servers = ?missing.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        "provisioning missing Python server venvs at startup"
    );

    let system_python = match find_system_python().await {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "cannot provision Python venvs — python not found");
            return;
        }
    };

    for (name, server_dir) in &missing {
        let venv_dir = server_dir.join(".venv");
        let bin_dir = venv_dir.join(venv_bin_dir());
        let pip_path = bin_dir.join(pip_executable());

        tracing::info!(server = %name, "creating venv...");

        let create_result = Command::new(&system_python)
            .args(["-m", "venv"])
            .arg(&venv_dir)
            .current_dir(server_dir)
            .output()
            .await;

        match create_result {
            Ok(output) if output.status.success() => {}
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                tracing::warn!(server = %name, stderr = %stderr.trim(), "failed to create venv");
                continue;
            }
            Err(e) => {
                tracing::warn!(server = %name, error = %e, "failed to run python -m venv");
                continue;
            }
        }

        // Upgrade pip (non-fatal)
        let _ = Command::new(&pip_path)
            .args(["install", "--quiet", "--upgrade", "pip"])
            .current_dir(server_dir)
            .output()
            .await;

        tracing::info!(server = %name, "installing dependencies...");

        let install_result = Command::new(&pip_path)
            .args(["install", "--quiet", "-e", "."])
            .current_dir(server_dir)
            .output()
            .await;

        match install_result {
            Ok(output) if output.status.success() => {
                tracing::info!(server = %name, "venv provisioned successfully");
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                tracing::warn!(server = %name, stderr = %stderr.trim(), "pip install failed");
            }
            Err(e) => {
                tracing::warn!(server = %name, error = %e, "failed to run pip install");
            }
        }
    }
}
