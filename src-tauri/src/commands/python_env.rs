//! IPC commands for Python MCP server environment provisioning.
//!
//! Ensures per-server `.venv` directories exist with all dependencies installed.
//! Called during onboarding (Setup step) and from Settings > Servers (Repair).
//!
//! The flow:
//!   1. Detect Python servers by scanning `mcp-servers/` for `pyproject.toml`
//!   2. For each server: check `.venv/bin/python` (or `Scripts\python.exe` on Windows)
//!   3. If missing: create venv → pip install -e .
//!   4. Emit progress events via Tauri so the frontend can show real-time status

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tauri::Emitter;
use tokio::process::Command;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Status of a single Python server's environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonEnvStatus {
    /// Server name (e.g., "ocr", "document").
    pub server: String,
    /// Whether the venv is ready (exists + deps installed).
    pub ready: bool,
    /// Error message if provisioning failed.
    pub error: Option<String>,
}

/// Progress event emitted during venv provisioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonEnvProgress {
    /// Server name.
    pub server: String,
    /// Current stage: "checking", "creating_venv", "installing_deps", "done", "failed".
    pub stage: String,
    /// Human-readable message.
    pub message: String,
}

// ─── Platform Helpers ───────────────────────────────────────────────────────

/// Return the platform-specific venv binary subdirectory name.
pub(super) fn venv_bin_dir() -> &'static str {
    if cfg!(target_os = "windows") {
        "Scripts"
    } else {
        "bin"
    }
}

/// Return the Python executable name for this platform.
pub(super) fn python_executable() -> &'static str {
    if cfg!(target_os = "windows") {
        "python.exe"
    } else {
        "python"
    }
}

/// Return the pip executable name for this platform.
pub(super) fn pip_executable() -> &'static str {
    if cfg!(target_os = "windows") {
        "pip.exe"
    } else {
        "pip"
    }
}

/// Find the system python3 command.
///
/// Checks `python3` first, then falls back to `python`.
pub(super) async fn find_system_python() -> Result<String, String> {
    for candidate in ["python3", "python"] {
        let result = Command::new(candidate)
            .arg("--version")
            .output()
            .await;
        if let Ok(output) = result {
            if output.status.success() {
                return Ok(candidate.to_string());
            }
        }
    }
    Err("Python 3 not found. Install Python 3.11+ from https://python.org".to_string())
}

// ─── Core Provisioning ──────────────────────────────────────────────────────

/// Check if a Python server's venv is already provisioned.
pub(super) fn is_venv_ready(server_dir: &Path) -> bool {
    let venv_dir = server_dir.join(".venv");
    let python_path = venv_dir.join(venv_bin_dir()).join(python_executable());
    python_path.exists()
}

/// Provision a single Python server's venv.
///
/// 1. Creates `.venv` via `python3 -m venv`
/// 2. Upgrades pip
/// 3. Runs `pip install -e .` to install the server's deps
async fn provision_server_env(
    server_name: &str,
    server_dir: &Path,
    app: &tauri::AppHandle,
) -> Result<(), String> {
    let venv_dir = server_dir.join(".venv");
    let bin_dir = venv_dir.join(venv_bin_dir());
    let pip_path = bin_dir.join(pip_executable());

    // Stage 1: Create venv
    emit_progress(app, server_name, "creating_venv", "Creating virtual environment...");

    let system_python = find_system_python().await?;

    let create_output = Command::new(&system_python)
        .args(["-m", "venv"])
        .arg(&venv_dir)
        .current_dir(server_dir)
        .output()
        .await
        .map_err(|e| format!("Failed to run python -m venv: {e}"))?;

    if !create_output.status.success() {
        let stderr = String::from_utf8_lossy(&create_output.stderr);
        return Err(format!("Failed to create venv: {}", stderr.trim()));
    }

    // Stage 2: Upgrade pip (prevents compatibility issues with hatchling)
    emit_progress(app, server_name, "installing_deps", "Upgrading pip...");

    let pip_upgrade = Command::new(&pip_path)
        .args(["install", "--quiet", "--upgrade", "pip"])
        .current_dir(server_dir)
        .output()
        .await
        .map_err(|e| format!("Failed to upgrade pip: {e}"))?;

    if !pip_upgrade.status.success() {
        let stderr = String::from_utf8_lossy(&pip_upgrade.stderr);
        tracing::warn!(
            server = server_name,
            stderr = %stderr.trim(),
            "pip upgrade failed (non-fatal, continuing with existing pip)"
        );
    }

    // Stage 3: Install server dependencies
    emit_progress(
        app,
        server_name,
        "installing_deps",
        "Installing dependencies...",
    );

    let install_output = Command::new(&pip_path)
        .args(["install", "--quiet", "-e", "."])
        .current_dir(server_dir)
        .output()
        .await
        .map_err(|e| format!("Failed to run pip install: {e}"))?;

    if !install_output.status.success() {
        let stderr = String::from_utf8_lossy(&install_output.stderr);
        return Err(format!(
            "Failed to install dependencies: {}",
            stderr.trim()
        ));
    }

    emit_progress(app, server_name, "done", "Ready");
    Ok(())
}

/// Emit a progress event to the frontend.
fn emit_progress(app: &tauri::AppHandle, server: &str, stage: &str, message: &str) {
    let payload = PythonEnvProgress {
        server: server.to_string(),
        stage: stage.to_string(),
        message: message.to_string(),
    };
    let _ = app.emit("python-env-progress", &payload);
    tracing::info!(
        server = server,
        stage = stage,
        message = message,
        "python env provisioning"
    );
}

// ─── Discovery Helper ───────────────────────────────────────────────────────

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

        // Skip internal/hidden directories
        if name.starts_with('_') || name.starts_with('.') {
            continue;
        }

        // Only Python servers (have pyproject.toml)
        if path.join("pyproject.toml").exists() {
            servers.push((name, path));
        }
    }

    servers
}

// ─── Tauri IPC Commands ─────────────────────────────────────────────────────

/// Ensure a single Python server's venv is provisioned.
///
/// Idempotent: if the venv already exists and has python, returns immediately.
/// Creates the venv and installs deps if missing.
#[tauri::command]
pub async fn ensure_python_server_env(
    server_name: String,
    app: tauri::AppHandle,
) -> Result<PythonEnvStatus, String> {
    let project_root = crate::resolve_project_root();
    let server_dir = project_root.join("mcp-servers").join(&server_name);

    if !server_dir.join("pyproject.toml").exists() {
        return Ok(PythonEnvStatus {
            server: server_name,
            ready: false,
            error: Some("Not a Python server (no pyproject.toml)".to_string()),
        });
    }

    emit_progress(&app, &server_name, "checking", "Checking environment...");

    if is_venv_ready(&server_dir) {
        emit_progress(&app, &server_name, "done", "Already provisioned");
        return Ok(PythonEnvStatus {
            server: server_name,
            ready: true,
            error: None,
        });
    }

    match provision_server_env(&server_name, &server_dir, &app).await {
        Ok(()) => Ok(PythonEnvStatus {
            server: server_name,
            ready: true,
            error: None,
        }),
        Err(e) => {
            emit_progress(&app, &server_name, "failed", &e);
            Ok(PythonEnvStatus {
                server: server_name,
                ready: false,
                error: Some(e),
            })
        }
    }
}

/// Ensure all Python MCP servers have their venvs provisioned.
///
/// Discovers Python servers from `mcp-servers/`, provisions each sequentially,
/// and returns aggregate status. Continues past individual failures.
#[tauri::command]
pub async fn ensure_all_python_envs(
    app: tauri::AppHandle,
) -> Result<Vec<PythonEnvStatus>, String> {
    let project_root = crate::resolve_project_root();
    let servers = discover_python_servers(&project_root);

    tracing::info!(
        count = servers.len(),
        servers = ?servers.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        "provisioning Python server environments"
    );

    let mut results = Vec::new();

    for (name, server_dir) in &servers {
        emit_progress(&app, name, "checking", "Checking environment...");

        if is_venv_ready(server_dir) {
            emit_progress(&app, name, "done", "Already provisioned");
            results.push(PythonEnvStatus {
                server: name.clone(),
                ready: true,
                error: None,
            });
            continue;
        }

        match provision_server_env(name, server_dir, &app).await {
            Ok(()) => {
                results.push(PythonEnvStatus {
                    server: name.clone(),
                    ready: true,
                    error: None,
                });
            }
            Err(e) => {
                emit_progress(&app, name, "failed", &e);
                tracing::warn!(
                    server = %name,
                    error = %e,
                    "failed to provision Python server env (non-fatal)"
                );
                results.push(PythonEnvStatus {
                    server: name.clone(),
                    ready: false,
                    error: Some(e),
                });
            }
        }
    }

    let ready_count = results.iter().filter(|r| r.ready).count();
    tracing::info!(
        ready = ready_count,
        total = results.len(),
        "Python server environment provisioning complete"
    );

    Ok(results)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_discover_python_servers() {
        let tmp = TempDir::new().unwrap();
        let mcp = tmp.path().join("mcp-servers");
        std::fs::create_dir(&mcp).unwrap();

        // Python server
        let ocr = mcp.join("ocr");
        std::fs::create_dir(&ocr).unwrap();
        std::fs::write(ocr.join("pyproject.toml"), "[project]\nname = \"ocr\"").unwrap();

        // TypeScript server (should be skipped)
        let fs_srv = mcp.join("filesystem");
        std::fs::create_dir(&fs_srv).unwrap();
        std::fs::write(fs_srv.join("package.json"), "{}").unwrap();

        // Hidden dir (should be skipped)
        let hidden = mcp.join("_shared");
        std::fs::create_dir(&hidden).unwrap();
        std::fs::write(hidden.join("pyproject.toml"), "").unwrap();

        let servers = discover_python_servers(tmp.path());
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].0, "ocr");
    }

    #[test]
    fn test_is_venv_ready_false_when_missing() {
        let tmp = TempDir::new().unwrap();
        assert!(!is_venv_ready(tmp.path()));
    }

    #[test]
    fn test_is_venv_ready_true_when_python_exists() {
        let tmp = TempDir::new().unwrap();
        let venv_bin = tmp.path().join(".venv").join(venv_bin_dir());
        std::fs::create_dir_all(&venv_bin).unwrap();
        std::fs::write(venv_bin.join(python_executable()), "").unwrap();
        assert!(is_venv_ready(tmp.path()));
    }

    #[test]
    fn test_venv_bin_dir_platform() {
        let dir = venv_bin_dir();
        if cfg!(target_os = "windows") {
            assert_eq!(dir, "Scripts");
        } else {
            assert_eq!(dir, "bin");
        }
    }
}
