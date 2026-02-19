//! MCP Server Auto-Discovery — scan `mcp-servers/` and build configs from conventions.
//!
//! Eliminates the need to manually maintain `mcp-servers.json` for every server.
//! Servers are detected by the presence of `package.json` (TypeScript) or
//! `pyproject.toml` (Python). The JSON file becomes an optional override.

use std::collections::HashMap;
use std::path::Path;

use super::types::ServerConfig;

// ─── Language Detection ──────────────────────────────────────────────────────

/// Detected language of an MCP server.
#[derive(Debug, PartialEq, Eq)]
enum ServerLanguage {
    TypeScript,
    Python,
}

/// Detect the language of a server directory by checking marker files.
///
/// - `package.json` → TypeScript
/// - `pyproject.toml` → Python
/// - Neither → `None` (not a server)
fn detect_language(server_dir: &Path) -> Option<ServerLanguage> {
    if server_dir.join("package.json").exists() {
        Some(ServerLanguage::TypeScript)
    } else if server_dir.join("pyproject.toml").exists() {
        Some(ServerLanguage::Python)
    } else {
        None
    }
}

// ─── Platform Helpers ────────────────────────────────────────────────────────

/// Platform-correct npx command.
///
/// Windows requires `npx.cmd` because `npx` is a batch script;
/// `Command::new("npx")` fails without the extension on Windows.
fn default_npx_command() -> &'static str {
    if cfg!(target_os = "windows") {
        "npx.cmd"
    } else {
        "npx"
    }
}

/// Platform-correct Python command.
///
/// macOS 12.3+ removed the `python` symlink; only `python3` exists.
/// Windows installs Python as `python.exe` via the official installer.
fn default_python_command() -> &'static str {
    if cfg!(target_os = "windows") {
        "python"
    } else {
        "python3"
    }
}

// ─── Config Generation ──────────────────────────────────────────────────────

/// Generate a `ServerConfig` for a TypeScript MCP server.
fn ts_config(name: &str) -> ServerConfig {
    ServerConfig {
        command: default_npx_command().to_string(),
        args: vec!["tsx".to_string(), "src/index.ts".to_string()],
        env: HashMap::new(),
        cwd: Some(format!("mcp-servers/{name}")),
        venv: None,
    }
}

/// Detect the Python entry point module for a server.
///
/// Checks for `src/server.py` first (preferred convention), then
/// `src/main.py` (used by knowledge, security, meeting, screenshot-pipeline).
fn detect_py_entry_module(server_dir: &Path) -> String {
    if server_dir.join("src").join("server.py").exists() {
        "src.server".to_string()
    } else if server_dir.join("src").join("main.py").exists() {
        "src.main".to_string()
    } else {
        // Fallback to convention — will fail at runtime with a clear error
        "src.server".to_string()
    }
}

/// Generate a `ServerConfig` for a Python MCP server.
///
/// If a `.venv` directory exists inside the server dir, sets `venv: ".venv"`.
/// Detects the entry point module by checking for `src/server.py` or `src/main.py`.
fn py_config(name: &str, server_dir: &Path) -> ServerConfig {
    let venv = if server_dir.join(".venv").is_dir() {
        Some(".venv".to_string())
    } else {
        None
    };

    let entry_module = detect_py_entry_module(server_dir);

    ServerConfig {
        command: default_python_command().to_string(),
        args: vec!["-m".to_string(), entry_module],
        env: HashMap::new(),
        cwd: Some(format!("mcp-servers/{name}")),
        venv,
    }
}

// ─── Discovery ──────────────────────────────────────────────────────────────

/// Scan the `mcp-servers/` directory and generate `ServerConfig` entries.
///
/// Skips directories starting with `_` or `.`. Returns an empty map if the
/// directory doesn't exist (graceful degradation).
pub fn discover_servers(mcp_servers_dir: &Path) -> HashMap<String, ServerConfig> {
    let mut configs = HashMap::new();

    let entries = match std::fs::read_dir(mcp_servers_dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(
                path = %mcp_servers_dir.display(),
                error = %e,
                "mcp-servers directory not found, skipping auto-discovery"
            );
            return configs;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let dir_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };

        // Skip internal/hidden directories
        if dir_name.starts_with('_') || dir_name.starts_with('.') {
            continue;
        }

        if let Some(language) = detect_language(&path) {
            let config = match language {
                ServerLanguage::TypeScript => ts_config(&dir_name),
                ServerLanguage::Python => py_config(&dir_name, &path),
            };
            tracing::debug!(
                server = %dir_name,
                language = ?language,
                "auto-discovered MCP server"
            );
            configs.insert(dir_name, config);
        }
    }

    configs
}

// ─── Merge ──────────────────────────────────────────────────────────────────

/// Merge auto-discovered configs with manual overrides from `mcp-servers.json`.
///
/// Override entries **fully replace** discovered entries for the same server name.
/// Override-only servers (not on disk) are added as-is (supports external servers).
pub fn merge_configs(
    mut discovered: HashMap<String, ServerConfig>,
    overrides: HashMap<String, ServerConfig>,
) -> HashMap<String, ServerConfig> {
    for (name, override_config) in overrides {
        // Override fully replaces the discovered config
        discovered.insert(name, override_config);
    }
    discovered
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_discover_ts_server() {
        let tmp = TempDir::new().unwrap();
        let server_dir = tmp.path().join("filesystem");
        std::fs::create_dir(&server_dir).unwrap();
        std::fs::write(server_dir.join("package.json"), "{}").unwrap();

        let configs = discover_servers(tmp.path());
        assert_eq!(configs.len(), 1);
        assert!(configs.contains_key("filesystem"));

        let cfg = &configs["filesystem"];
        assert_eq!(cfg.command, default_npx_command());
        assert_eq!(cfg.args, vec!["tsx", "src/index.ts"]);
        assert_eq!(cfg.cwd, Some("mcp-servers/filesystem".to_string()));
        assert_eq!(cfg.venv, None);
    }

    #[test]
    fn test_discover_py_server_with_server_py() {
        let tmp = TempDir::new().unwrap();
        let server_dir = tmp.path().join("document");
        std::fs::create_dir_all(server_dir.join("src")).unwrap();
        std::fs::write(server_dir.join("pyproject.toml"), "").unwrap();
        std::fs::write(server_dir.join("src").join("server.py"), "").unwrap();

        let configs = discover_servers(tmp.path());
        assert_eq!(configs.len(), 1);

        let cfg = &configs["document"];
        assert_eq!(cfg.command, default_python_command());
        assert_eq!(cfg.args, vec!["-m", "src.server"]);
        assert_eq!(cfg.venv, None);
    }

    #[test]
    fn test_discover_py_server_with_main_py() {
        let tmp = TempDir::new().unwrap();
        let server_dir = tmp.path().join("knowledge");
        std::fs::create_dir_all(server_dir.join("src")).unwrap();
        std::fs::write(server_dir.join("pyproject.toml"), "").unwrap();
        std::fs::write(server_dir.join("src").join("main.py"), "").unwrap();

        let configs = discover_servers(tmp.path());
        assert_eq!(configs.len(), 1);

        let cfg = &configs["knowledge"];
        assert_eq!(cfg.command, default_python_command());
        assert_eq!(cfg.args, vec!["-m", "src.main"]);
        assert_eq!(cfg.venv, None);
    }

    #[test]
    fn test_discover_py_server_with_venv() {
        let tmp = TempDir::new().unwrap();
        let server_dir = tmp.path().join("ocr");
        std::fs::create_dir(&server_dir).unwrap();
        std::fs::write(server_dir.join("pyproject.toml"), "").unwrap();
        std::fs::create_dir(server_dir.join(".venv")).unwrap();

        let configs = discover_servers(tmp.path());
        let cfg = &configs["ocr"];
        assert_eq!(cfg.command, default_python_command());
        assert_eq!(cfg.venv, Some(".venv".to_string()));
    }

    #[test]
    fn test_skip_underscore_dirs() {
        let tmp = TempDir::new().unwrap();
        let shared = tmp.path().join("_shared");
        std::fs::create_dir(&shared).unwrap();
        std::fs::write(shared.join("package.json"), "{}").unwrap();

        let configs = discover_servers(tmp.path());
        assert!(configs.is_empty());
    }

    #[test]
    fn test_skip_dot_dirs() {
        let tmp = TempDir::new().unwrap();
        let hidden = tmp.path().join(".hidden");
        std::fs::create_dir(&hidden).unwrap();
        std::fs::write(hidden.join("package.json"), "{}").unwrap();

        let configs = discover_servers(tmp.path());
        assert!(configs.is_empty());
    }

    #[test]
    fn test_skip_non_directory_files() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("README.md"), "# readme").unwrap();

        let configs = discover_servers(tmp.path());
        assert!(configs.is_empty());
    }

    #[test]
    fn test_missing_directory() {
        let configs = discover_servers(Path::new("/nonexistent/path/mcp-servers"));
        assert!(configs.is_empty());
    }

    #[test]
    fn test_merge_override_replaces() {
        let mut discovered = HashMap::new();
        discovered.insert(
            "filesystem".to_string(),
            ts_config("filesystem"),
        );

        let mut overrides = HashMap::new();
        overrides.insert(
            "filesystem".to_string(),
            ServerConfig {
                command: "node".to_string(),
                args: vec!["dist/index.js".to_string()],
                env: HashMap::new(),
                cwd: Some("/custom/path".to_string()),
                venv: None,
            },
        );

        let merged = merge_configs(discovered, overrides);
        assert_eq!(merged["filesystem"].command, "node");
        assert_eq!(merged["filesystem"].cwd, Some("/custom/path".to_string()));
    }

    #[test]
    fn test_merge_adds_override_only_servers() {
        let discovered = HashMap::new();
        let mut overrides = HashMap::new();
        overrides.insert(
            "external".to_string(),
            ServerConfig {
                command: "custom-mcp".to_string(),
                args: vec![],
                env: HashMap::new(),
                cwd: None,
                venv: None,
            },
        );

        let merged = merge_configs(discovered, overrides);
        assert!(merged.contains_key("external"));
        assert_eq!(merged["external"].command, "custom-mcp");
    }

    #[test]
    fn test_merge_preserves_non_overridden() {
        let mut discovered = HashMap::new();
        discovered.insert("fs".to_string(), ts_config("fs"));
        discovered.insert("ocr".to_string(), ts_config("ocr"));

        let mut overrides = HashMap::new();
        overrides.insert(
            "fs".to_string(),
            ServerConfig {
                command: "node".to_string(),
                args: vec![],
                env: HashMap::new(),
                cwd: None,
                venv: None,
            },
        );

        let merged = merge_configs(discovered, overrides);
        // fs was overridden
        assert_eq!(merged["fs"].command, "node");
        // ocr was NOT overridden — preserved from discovery
        assert_eq!(merged["ocr"].command, "npx");
    }
}
