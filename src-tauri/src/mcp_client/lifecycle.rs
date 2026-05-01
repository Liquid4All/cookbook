//! Server process lifecycle management.
//!
//! Handles spawning, monitoring, restarting, and shutting down MCP server
//! child processes. Each server runs as a separate OS process communicating
//! via JSON-RPC over stdio.

use std::collections::HashMap;
use std::time::Duration;

use tokio::process::{Child, Command};
use tokio::time::sleep;

use super::errors::McpError;
use super::transport::StdioTransport;
use super::types::{InitializeResult, McpToolDefinition, ServerConfig};

// ─── Constants ───────────────────────────────────────────────────────────────

/// Maximum restart attempts before giving up on a server.
const MAX_RESTART_ATTEMPTS: u32 = 3;

/// Base delay between restart attempts (doubles each time).
const RESTART_BASE_DELAY: Duration = Duration::from_secs(1);

/// Timeout for the initialize handshake.
///
/// Set to 30s to accommodate ML-heavy servers (meeting, ocr) that import
/// PyTorch, Whisper, and other large frameworks at startup.
const INIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Timeout for graceful shutdown before force-killing.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

// ─── ManagedServer ───────────────────────────────────────────────────────────

/// A running MCP server process with its transport and tool definitions.
pub struct ManagedServer {
    /// Human-readable server name (e.g., "filesystem").
    pub name: String,
    /// The child process handle.
    process: Child,
    /// JSON-RPC transport (stdin/stdout).
    pub transport: StdioTransport,
    /// Tool definitions received during initialization.
    pub tools: Vec<McpToolDefinition>,
    /// Number of times this server has been restarted.
    restart_count: u32,
    /// The original config used to spawn this server (retained for restart).
    #[allow(dead_code)]
    config: ServerConfig,
}

impl ManagedServer {
    /// How many times this server has been restarted.
    pub fn restart_count(&self) -> u32 {
        self.restart_count
    }

    /// Check if the server process is still running.
    pub async fn is_alive(&mut self) -> bool {
        match self.process.try_wait() {
            Ok(None) => true,  // Still running
            Ok(Some(_)) => false, // Exited
            Err(_) => false,   // Error checking — assume dead
        }
    }

    /// Attempt to gracefully shut down the server.
    pub async fn shutdown(&mut self) -> Result<(), McpError> {
        // Send shutdown notification (best-effort)
        let _ = self.transport.notify("shutdown", None).await;

        // Wait for graceful exit
        let result = tokio::time::timeout(SHUTDOWN_TIMEOUT, self.process.wait()).await;

        match result {
            Ok(Ok(_)) => Ok(()),
            _ => {
                // Force kill if graceful shutdown failed/timed out
                let _ = self.process.kill().await;
                Ok(())
            }
        }
    }
}

// ─── Spawning ────────────────────────────────────────────────────────────────

/// Spawn a single MCP server process and perform the initialization handshake.
///
/// Returns a `ManagedServer` with its transport and discovered tools.
pub async fn spawn_server(
    name: &str,
    config: &ServerConfig,
    working_dir: Option<&str>,
) -> Result<ManagedServer, McpError> {
    let mut cmd = Command::new(&config.command);
    cmd.args(&config.args);

    // Set environment variables
    for (key, value) in &config.env {
        cmd.env(key, value);
    }

    // Set working directory: per-server cwd overrides the global working_dir
    let effective_dir = config.cwd.as_deref().or(working_dir);
    if let Some(dir) = effective_dir {
        cmd.current_dir(dir);
    }

    // Windows: prevent console window from appearing for child processes
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    // Wire stdio for JSON-RPC
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped()); // Capture stderr for logging

    let mut child = cmd.spawn().map_err(|e| McpError::SpawnFailed {
        name: name.to_string(),
        reason: format!("{e}"),
    })?;

    let stdin = child.stdin.take().ok_or(McpError::SpawnFailed {
        name: name.to_string(),
        reason: "failed to capture stdin".into(),
    })?;

    let stdout = child.stdout.take().ok_or(McpError::SpawnFailed {
        name: name.to_string(),
        reason: "failed to capture stdout".into(),
    })?;

    // Extract stderr for diagnostic capture on failure
    let stderr_handle = child.stderr.take();

    let transport = StdioTransport::new(name, stdin, stdout);

    // Perform initialization handshake with timeout
    let tools = match tokio::time::timeout(INIT_TIMEOUT, initialize(&transport, name)).await {
        Ok(Ok(tools)) => tools,
        Ok(Err(e)) => {
            let stderr_ctx = read_stderr_on_failure(stderr_handle).await;
            if !stderr_ctx.is_empty() {
                tracing::warn!(
                    server = name,
                    stderr = %stderr_ctx,
                    "server stderr captured on failure"
                );
            }
            let reason = format!("{e}{}", format_stderr_suffix(&stderr_ctx));
            return Err(McpError::InitFailed {
                name: name.to_string(),
                reason,
            });
        }
        Err(_) => {
            let stderr_ctx = read_stderr_on_failure(stderr_handle).await;
            if !stderr_ctx.is_empty() {
                tracing::warn!(
                    server = name,
                    stderr = %stderr_ctx,
                    "server stderr captured on timeout"
                );
            }
            let _ = child.kill().await;
            let reason = format!(
                "initialization timed out after {}s{}",
                INIT_TIMEOUT.as_secs(),
                format_stderr_suffix(&stderr_ctx)
            );
            return Err(McpError::InitFailed {
                name: name.to_string(),
                reason,
            });
        }
    };

    Ok(ManagedServer {
        name: name.to_string(),
        process: child,
        transport,
        tools,
        restart_count: 0,
        config: config.clone(),
    })
}

/// Read any available stderr output from a failed server process.
///
/// Uses a short timeout to avoid blocking if stderr is empty or the process
/// is still writing. Truncates to 2000 chars to keep log messages readable.
async fn read_stderr_on_failure(
    stderr_handle: Option<tokio::process::ChildStderr>,
) -> String {
    use tokio::io::AsyncReadExt;

    let Some(mut stderr) = stderr_handle else {
        return String::new();
    };

    let mut buf = String::new();
    match tokio::time::timeout(
        Duration::from_millis(500),
        stderr.read_to_string(&mut buf),
    )
    .await
    {
        Ok(Ok(_)) => {
            if buf.len() > 2000 {
                buf.truncate(2000);
                buf.push_str("...(truncated)");
            }
            buf
        }
        _ => String::new(),
    }
}

/// Format a stderr suffix for error messages (empty string if no stderr).
fn format_stderr_suffix(stderr: &str) -> String {
    if stderr.is_empty() {
        String::new()
    } else {
        format!(" | stderr: {}", stderr.trim())
    }
}

/// Perform the MCP initialization handshake.
async fn initialize(
    transport: &StdioTransport,
    server_name: &str,
) -> Result<Vec<McpToolDefinition>, McpError> {
    let response = transport.request("initialize", None).await?;

    let result = super::transport::extract_result(response)?;

    let init_result: InitializeResult =
        serde_json::from_value(result).map_err(|e| McpError::InitFailed {
            name: server_name.to_string(),
            reason: format!("failed to parse initialize response: {e}"),
        })?;

    Ok(init_result.tools)
}

/// Restart a crashed server with exponential backoff.
///
/// Returns the new `ManagedServer` if successful, or an error if all
/// attempts are exhausted.
pub async fn restart_server(
    name: &str,
    config: &ServerConfig,
    working_dir: Option<&str>,
    current_restart_count: u32,
) -> Result<ManagedServer, McpError> {
    if current_restart_count >= MAX_RESTART_ATTEMPTS {
        return Err(McpError::RestartExhausted {
            name: name.to_string(),
            attempts: MAX_RESTART_ATTEMPTS,
        });
    }

    // Exponential backoff: 1s, 2s, 4s
    let delay = RESTART_BASE_DELAY * 2u32.pow(current_restart_count);
    sleep(delay).await;

    let mut server = spawn_server(name, config, working_dir).await?;
    server.restart_count = current_restart_count + 1;
    Ok(server)
}

// ─── Batch Operations ────────────────────────────────────────────────────────

/// Spawn all configured servers concurrently.
///
/// Returns a map of server name → `ManagedServer`. Servers that fail to
/// start are logged but not included (partial startup is acceptable).
pub async fn spawn_all_servers(
    configs: &HashMap<String, ServerConfig>,
    working_dir: Option<&str>,
) -> (HashMap<String, ManagedServer>, Vec<(String, McpError)>) {
    let mut servers = HashMap::new();
    let mut errors = Vec::new();

    // Spawn servers concurrently using join_all
    let mut handles: Vec<(String, _)> = Vec::new();
    for (name, config) in configs {
        let name = name.clone();
        let config = config.clone();
        let wd = working_dir.map(|s| s.to_string());
        handles.push((
            name.clone(),
            tokio::spawn(async move {
                spawn_server(&name, &config, wd.as_deref()).await
            }),
        ));
    }

    for (name, handle) in handles {
        match handle.await {
            Ok(Ok(server)) => {
                servers.insert(name, server);
            }
            Ok(Err(e)) => {
                errors.push((name, e));
            }
            Err(e) => {
                errors.push((
                    name.clone(),
                    McpError::SpawnFailed {
                        name,
                        reason: format!("join error: {e}"),
                    },
                ));
            }
        }
    }

    (servers, errors)
}

/// Shut down all managed servers gracefully.
pub async fn shutdown_all_servers(servers: &mut HashMap<String, ManagedServer>) {
    for (_, server) in servers.iter_mut() {
        let _ = server.shutdown().await;
    }
    servers.clear();
}
