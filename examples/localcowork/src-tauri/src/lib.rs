pub mod agent_core;
pub mod commands;
pub mod inference;
pub mod mcp_client;

use std::sync::Mutex;

use agent_core::{AgentDatabase, ConversationManager, ConfirmationResponse, PermissionStore};
use commands::settings::SamplingConfig;
use mcp_client::McpClient;
use tauri::Manager;

/// Pending confirmation channel — holds a oneshot sender while the agent loop
/// awaits a user response via the ConfirmationDialog.
pub type PendingConfirmation =
    TokioMutex<Option<tokio::sync::oneshot::Sender<ConfirmationResponse>>>;

/// Async mutex for types that require `.await` inside their methods.
pub type TokioMutex<T> = tokio::sync::Mutex<T>;

/// Return the platform-standard data directory for LocalCowork.
///
/// - macOS: `~/Library/Application Support/com.localcowork.app/`
/// - Windows: `{FOLDERID_RoamingAppData}\localcowork\`
/// - Linux: `$XDG_DATA_HOME/com.localcowork.app/` (fallback `~/.local/share/...`)
///
/// Falls back to `~/.localcowork/` only if none of the above can be resolved.
pub(crate) fn data_dir() -> std::path::PathBuf {
    if let Some(dir) = dirs::data_dir() {
        return dir.join("com.localcowork.app");
    }
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".localcowork")
}

/// Initialize the tracing subscriber — writes structured logs to the app data directory.
///
/// On each app startup:
/// 1. Rotates existing logs (agent.log → agent.log.1 → .2 → .3, keeps last 3).
/// 2. Opens a fresh agent.log with a line-flushing writer for crash resilience.
/// 3. Logs a startup banner with the data directory path for discoverability.
fn init_tracing() {
    use tracing_subscriber::fmt;
    use tracing_subscriber::EnvFilter;

    let log_dir = data_dir();
    let _ = std::fs::create_dir_all(&log_dir);

    let log_path = log_dir.join("agent.log");

    // Rotate: agent.log.2 → .3, .1 → .2, agent.log → .1
    rotate_log_file(&log_path, 3);

    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .expect("failed to open agent.log");

    let flushing_writer = FlushingWriter::new(log_file);

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("localcowork=info,warn"));

    fmt::fmt()
        .with_env_filter(filter)
        .with_writer(flushing_writer)
        .with_ansi(false)
        .with_target(true)
        .with_thread_ids(false)
        .init();

    // Startup banner — makes it easy to find the right log file
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        data_dir = %log_dir.display(),
        log_file = %log_path.display(),
        pid = std::process::id(),
        "=== LocalCowork starting ==="
    );
}

/// Rotate log files: `agent.log` → `agent.log.1` → `.2` → … → `.{keep}`.
///
/// Oldest file beyond `keep` is deleted. Missing files in the chain are skipped.
fn rotate_log_file(base_path: &std::path::Path, keep: u32) {
    // Delete the oldest
    let oldest = format!("{}.{keep}", base_path.display());
    let _ = std::fs::remove_file(&oldest);

    // Shift: .{n-1} → .{n}
    for i in (1..keep).rev() {
        let from = format!("{}.{i}", base_path.display());
        let to = format!("{}.{}", base_path.display(), i + 1);
        let _ = std::fs::rename(&from, &to);
    }

    // Current → .1
    if base_path.exists() {
        let to = format!("{}.1", base_path.display());
        let _ = std::fs::rename(base_path, &to);
    }
}

/// A writer that wraps `std::fs::File` and flushes after every write.
///
/// `tracing-subscriber` buffers log output internally. Without explicit
/// flushing, log entries may sit in OS buffers and be lost on crash.
/// This wrapper ensures each log line is on disk immediately.
///
/// Performance impact is minimal for a desktop app (~100 log lines/minute).
#[derive(Clone)]
struct FlushingWriter {
    file: std::sync::Arc<std::sync::Mutex<std::fs::File>>,
}

impl FlushingWriter {
    fn new(file: std::fs::File) -> Self {
        Self {
            file: std::sync::Arc::new(std::sync::Mutex::new(file)),
        }
    }
}

impl std::io::Write for FlushingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut f = self.file.lock().map_err(|e| {
            std::io::Error::other(format!("lock poisoned: {e}"))
        })?;
        let n = std::io::Write::write(&mut *f, buf)?;
        std::io::Write::flush(&mut *f)?;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut f = self.file.lock().map_err(|e| {
            std::io::Error::other(format!("lock poisoned: {e}"))
        })?;
        std::io::Write::flush(&mut *f)
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for FlushingWriter {
    type Writer = FlushingWriter;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// Resolve the path for the agent SQLite database.
///
/// Uses the platform-standard data directory (creates it if needed).
fn resolve_db_path() -> String {
    let dir = data_dir();
    if !dir.exists() {
        let _ = std::fs::create_dir_all(&dir);
    }
    dir.join("agent.db").to_string_lossy().into_owned()
}

/// Resolve the MCP servers configuration using auto-discovery + optional overrides.
///
/// 1. Auto-discovers servers by scanning `mcp-servers/` for `package.json` (TS)
///    or `pyproject.toml` (Python) markers.
/// 2. Loads `mcp-servers.json` as optional overrides (missing file is fine).
/// 3. Merges: override entries fully replace discovered entries.
/// 4. Resolves relative paths, venvs, and injects vision model env vars.
fn resolve_mcp_config() -> mcp_client::types::McpServersConfig {
    let project_root = resolve_project_root();

    // 1. Auto-discover servers from mcp-servers/ directory
    let mcp_servers_dir = project_root.join("mcp-servers");
    let discovered = mcp_client::discovery::discover_servers(&mcp_servers_dir);
    tracing::info!(
        discovered = discovered.len(),
        servers = ?discovered.keys().collect::<Vec<_>>(),
        "auto-discovered MCP servers"
    );

    // 2. Load optional override file
    let overrides = load_override_file(&project_root);

    // 3. Merge: overrides win
    let mut merged = mcp_client::discovery::merge_configs(discovered, overrides);

    // 4. Filter by enabled_servers allowlist from _models/config.yaml (if set)
    filter_by_enabled_servers(&mut merged, &project_root);

    let mut config = mcp_client::types::McpServersConfig { servers: merged };

    // 5. Post-process: resolve paths, venvs, inject vision env vars
    resolve_paths_and_env(&mut config, &project_root);

    tracing::info!(
        server_count = config.servers.len(),
        servers = ?config.servers.keys().collect::<Vec<_>>(),
        "final MCP server config"
    );

    config
}

/// Filter discovered servers by the `enabled_servers` allowlist in `_models/config.yaml`.
///
/// When `enabled_servers` is set, only servers whose names appear in the list
/// are kept. All others are removed. When absent or empty, all servers pass through.
fn filter_by_enabled_servers(
    servers: &mut std::collections::HashMap<String, mcp_client::ServerConfig>,
    project_root: &std::path::Path,
) {
    let config_path = project_root.join("_models/config.yaml");
    let content = match std::fs::read_to_string(&config_path) {
        Ok(c) => c,
        Err(_) => return, // No config file — skip filtering
    };

    // Parse just enough YAML to extract enabled_servers without requiring
    // the full ModelsConfig (which needs model configs to be valid).
    let yaml: serde_json::Value = match serde_yaml::from_str(&content) {
        Ok(v) => v,
        Err(_) => return,
    };

    let enabled = match yaml.get("enabled_servers").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return, // Field absent — no filtering
    };

    let allowlist: std::collections::HashSet<String> = enabled
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    if allowlist.is_empty() {
        return;
    }

    let before = servers.len();
    servers.retain(|name, _| allowlist.contains(name));
    let after = servers.len();

    tracing::info!(
        before,
        after,
        enabled = ?allowlist,
        "filtered MCP servers by enabled_servers allowlist"
    );
}

/// Filter tools by the `enabled_tools` allowlist in `_models/config.yaml`.
///
/// When `enabled_tools` is set, only tools whose fully-qualified names appear
/// in the list are kept in the registry. All others are removed. This allows
/// curating a tight tool surface for specific demos or deployments.
///
/// Must be called AFTER `McpClient::start_all()` has populated the registry.
fn filter_tools_by_allowlist(mcp_client: &mut McpClient, project_root: &std::path::Path) {
    let config_path = project_root.join("_models/config.yaml");
    let content = match std::fs::read_to_string(&config_path) {
        Ok(c) => c,
        Err(_) => return, // No config file — skip filtering
    };

    let yaml: serde_json::Value = match serde_yaml::from_str(&content) {
        Ok(v) => v,
        Err(_) => return,
    };

    let enabled = match yaml.get("enabled_tools").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return, // Field absent — no filtering
    };

    let allowlist: std::collections::HashSet<String> = enabled
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    if allowlist.is_empty() {
        return;
    }

    mcp_client.registry.retain_tools(&allowlist);
}

/// Determine the project root directory.
///
/// Resolution order:
/// 1. `mcp-servers/` relative to cwd (dev mode, running from project root).
/// 2. `../mcp-servers/` relative to cwd (dev mode, running from `src-tauri/`).
/// 3. `mcp-servers/` relative to the executable (packaged app).
/// 4. Fallback: cwd parent directory.
pub(crate) fn resolve_project_root() -> std::path::PathBuf {
    let cwd = std::env::current_dir().unwrap_or_default();

    // Dev mode: cwd is the project root
    if cwd.join("mcp-servers").is_dir() {
        return cwd;
    }

    // Dev mode: cwd is src-tauri/
    if cwd.join("..").join("mcp-servers").is_dir() {
        return cwd.join("..").canonicalize().unwrap_or(cwd);
    }

    // Packaged app: check relative to the executable location.
    // macOS: .app/Contents/MacOS/localcowork → .app/Contents/Resources/
    // Windows: install_dir/localcowork.exe → install_dir/
    // Linux: install_dir/localcowork → install_dir/
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            // macOS .app bundle: Resources/ is a sibling of MacOS/
            let macos_resources = exe_dir.join("../Resources");
            if macos_resources.join("mcp-servers").is_dir() {
                if let Ok(resolved) = macos_resources.canonicalize() {
                    return resolved;
                }
            }
            // Flat layout (Windows/Linux or dev binary)
            if exe_dir.join("mcp-servers").is_dir() {
                return exe_dir.to_path_buf();
            }
        }
    }

    // Last resort: cwd parent
    cwd.parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf()
}

/// Load the optional `mcp-servers.json` override file.
///
/// Returns an empty map if the file doesn't exist or can't be parsed.
fn load_override_file(
    project_root: &std::path::Path,
) -> std::collections::HashMap<String, mcp_client::ServerConfig> {
    let candidates = [
        project_root.join("src-tauri/mcp-servers.json"),
        project_root.join("mcp-servers.json"),
    ];

    for path in &candidates {
        if let Ok(content) = std::fs::read_to_string(path) {
            match serde_json::from_str::<mcp_client::types::McpServersConfig>(&content) {
                Ok(cfg) => {
                    tracing::info!(
                        path = %path.display(),
                        count = cfg.servers.len(),
                        "loaded MCP override config"
                    );
                    return cfg.servers;
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "failed to parse MCP override config"
                    );
                }
            }
        }
    }

    std::collections::HashMap::new()
}

/// Resolve relative paths, venvs, and inject vision env vars into all server configs.
fn resolve_paths_and_env(
    config: &mut mcp_client::types::McpServersConfig,
    project_root: &std::path::Path,
) {
    for server_config in config.servers.values_mut() {
        // Resolve relative cwd to absolute
        if let Some(ref cwd) = server_config.cwd {
            if !std::path::Path::new(cwd).is_absolute() {
                let abs_cwd = project_root.join(cwd);
                server_config.cwd = Some(abs_cwd.to_string_lossy().into_owned());
            }
        }

        // Resolve venv: rewrite command to venv binary and inject env vars
        if let Some(ref venv) = server_config.venv {
            let base_dir = server_config
                .cwd
                .as_ref()
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| project_root.to_path_buf());

            let abs_venv = if std::path::Path::new(venv).is_absolute() {
                std::path::PathBuf::from(venv)
            } else {
                base_dir.join(venv)
            };
            // Windows venvs use Scripts\ instead of bin/
            let venv_bin = if cfg!(target_os = "windows") {
                abs_venv.join("Scripts")
            } else {
                abs_venv.join("bin")
            };
            let venv_command = venv_bin.join(&server_config.command);

            if venv_command.exists() {
                server_config.command = venv_command.to_string_lossy().into_owned();
                server_config.env.insert(
                    "VIRTUAL_ENV".to_string(),
                    abs_venv.to_string_lossy().into_owned(),
                );
                let system_path = std::env::var("PATH").unwrap_or_default();
                server_config.env.insert(
                    "PATH".to_string(),
                    if cfg!(target_os = "windows") {
                        format!("{};{system_path}", venv_bin.to_string_lossy())
                    } else {
                        format!("{}:{system_path}", venv_bin.to_string_lossy())
                    },
                );
                tracing::info!(
                    venv = %abs_venv.display(),
                    command = %server_config.command,
                    "resolved venv for MCP server"
                );
            } else {
                tracing::warn!(
                    venv = %abs_venv.display(),
                    command = %server_config.command,
                    "venv binary not found, using command as-is"
                );
            }

            server_config.venv = Some(abs_venv.to_string_lossy().into_owned());
        }
    }

    // Inject LOCALCOWORK_DATA_DIR so MCP servers use platform-standard paths
    let app_data = data_dir().to_string_lossy().into_owned();
    for server_config in config.servers.values_mut() {
        server_config
            .env
            .entry("LOCALCOWORK_DATA_DIR".to_string())
            .or_insert_with(|| app_data.clone());
    }

    // Inject vision model endpoint env vars
    if let Some((vision_endpoint, vision_model)) = resolve_vision_model(project_root) {
        for server_config in config.servers.values_mut() {
            server_config
                .env
                .entry("LOCALCOWORK_VISION_ENDPOINT".to_string())
                .or_insert_with(|| vision_endpoint.clone());
            server_config
                .env
                .entry("LOCALCOWORK_VISION_MODEL".to_string())
                .or_insert_with(|| vision_model.clone());
        }
        tracing::info!(
            endpoint = %vision_endpoint,
            model = %vision_model,
            "injected vision model env vars into MCP servers"
        );
    }
}

/// Find the first vision-capable model from `_models/config.yaml`.
///
/// Returns `(base_url, model_name)` if a model with the "vision" capability is found.
/// Checks: (1) active model, (2) fallback chain, (3) any model in the config.
fn resolve_vision_model(project_root: &std::path::Path) -> Option<(String, String)> {
    let config_path = project_root.join("_models/config.yaml");
    let content = std::fs::read_to_string(&config_path).ok()?;
    let yaml: serde_json::Value = serde_yaml::from_str(&content).ok()?;

    let models = yaml.get("models")?.as_object()?;
    let active = yaml.get("active_model")?.as_str()?;

    // Helper: check if a model has vision capability
    let has_vision = |key: &str| -> Option<(String, String)> {
        let model = models.get(key)?;
        let caps = model.get("capabilities")?.as_array()?;
        let is_vision = caps.iter().any(|c| c.as_str() == Some("vision"));
        if !is_vision {
            return None;
        }
        let base_url = model.get("base_url")?.as_str()?.to_string();
        let model_name = model
            .get("model_name")
            .and_then(|v| v.as_str())
            .unwrap_or(key)
            .to_string();
        Some((base_url, model_name))
    };

    // 1. Check active model first
    if let Some(result) = has_vision(active) {
        return Some(result);
    }

    // 2. Check fallback chain
    if let Some(chain) = yaml.get("fallback_chain").and_then(|c| c.as_array()) {
        for entry in chain {
            if let Some(key) = entry.as_str() {
                if let Some(result) = has_vision(key) {
                    return Some(result);
                }
            }
        }
    }

    // 3. Scan all models for any with vision capability (e.g., dedicated VL model)
    for key in models.keys() {
        if let Some(result) = has_vision(key) {
            return Some(result);
        }
    }

    None
}

/// Run the Tauri application.
pub fn run() {
    // Initialize tracing FIRST — before any tracing::info!() calls
    init_tracing();

    // Initialize the SQLite-backed ConversationManager
    let db_path = resolve_db_path();
    let db = AgentDatabase::open(&db_path).expect("failed to open agent database");
    let conversation_manager = ConversationManager::new(db);

    tracing::info!(db_path = %db_path, "agent database initialized");

    // Register an empty MCP client synchronously so that TokioMutex<McpClient>
    // is always available in Tauri state. The async setup task will replace the
    // empty client with a fully initialized one once servers are started.
    // This prevents panics if start_session is called before MCP init completes.
    let empty_mcp_config = mcp_client::types::McpServersConfig {
        servers: std::collections::HashMap::new(),
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(Mutex::new(conversation_manager))
        .manage(TokioMutex::new(McpClient::new(empty_mcp_config, None)))
        .manage(TokioMutex::new(PermissionStore::new()))
        .manage(TokioMutex::new(SamplingConfig::load_or_default()))
        .manage(TokioMutex::new(None::<tokio::sync::oneshot::Sender<ConfirmationResponse>>)
            as PendingConfirmation)
        .setup(|app| {
            // Initialize MCP client asynchronously during app setup.
            // Once servers are started, replace the empty client via lock.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                // Provision missing Python venvs BEFORE resolving MCP config,
                // so that discovery picks up the newly created .venv directories.
                let project_root = resolve_project_root();
                commands::python_env_startup::provision_missing_venvs(&project_root).await;

                let config = resolve_mcp_config();
                let mut mcp_client = McpClient::new(config, None);

                let errors = mcp_client.start_all().await;
                for (name, err) in &errors {
                    tracing::warn!(
                        server = %name,
                        error = %err,
                        "MCP server failed to start (non-fatal)"
                    );
                }

                // Filter tools by enabled_tools allowlist (if configured)
                filter_tools_by_allowlist(&mut mcp_client, &project_root);


                let running = mcp_client.running_server_count();
                let tools = mcp_client.tool_count();
                tracing::info!(
                    running_servers = running,
                    total_tools = tools,
                    "MCP client initialized"
                );

                // Replace the empty placeholder with the fully initialized client
                let state: tauri::State<'_, TokioMutex<McpClient>> = handle.state();
                let mut lock = state.lock().await;
                *lock = mcp_client;
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::greet,
            commands::chat::start_session,
            commands::chat::send_message,
            commands::chat::respond_to_confirmation,
            commands::session::list_sessions,
            commands::session::load_session,
            commands::session::delete_session,
            commands::session::get_context_budget,
            commands::session::cleanup_empty_sessions,
            commands::filesystem::list_directory,
            commands::filesystem::get_home_dir,
            commands::settings::get_models_config,
            commands::settings::get_mcp_servers_status,
            commands::settings::list_permission_grants,
            commands::settings::revoke_permission,
            commands::settings::get_sampling_config,
            commands::settings::update_sampling_config,
            commands::settings::reset_sampling_config,
            commands::hardware::detect_hardware,
            commands::model_download::download_model,
            commands::model_download::verify_model,
            commands::model_download::get_model_dir,
            commands::ollama::check_llama_server_status,
            commands::ollama::check_ollama_status,
            commands::ollama::list_ollama_models,
            commands::ollama::pull_ollama_model,
            commands::python_env::ensure_python_server_env,
            commands::python_env::ensure_all_python_envs,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
