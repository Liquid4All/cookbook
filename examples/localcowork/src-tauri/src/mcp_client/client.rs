//! MCP Client — high-level interface for tool execution.
//!
//! Orchestrates server lifecycle, tool registry, and tool call dispatch.
//! This is the primary API used by the ToolRouter (WS-2D).

use std::collections::HashMap;
use std::time::Instant;

use super::errors::McpError;
use super::lifecycle;
use super::registry::ToolRegistry;
use super::types::{McpServersConfig, ServerConfig, ToolCallResult};

// ─── Constants ───────────────────────────────────────────────────────────────

/// Default timeout for tool call execution (ms).
const DEFAULT_CALL_TIMEOUT_MS: u64 = 30_000;

// ─── McpClient ───────────────────────────────────────────────────────────────

/// High-level MCP client that manages multiple servers and routes tool calls.
pub struct McpClient {
    /// Running server processes.
    servers: HashMap<String, lifecycle::ManagedServer>,
    /// Server configurations (for restarts).
    configs: HashMap<String, ServerConfig>,
    /// Aggregated tool definitions from all servers.
    pub registry: ToolRegistry,
    /// Working directory for server processes.
    working_dir: Option<String>,
    /// Tool call timeout in milliseconds.
    call_timeout_ms: u64,
}

impl McpClient {
    /// Create a new MCP client from a servers configuration file.
    pub fn new(config: McpServersConfig, working_dir: Option<String>) -> Self {
        Self {
            servers: HashMap::new(),
            configs: config.servers,
            registry: ToolRegistry::new(),
            working_dir,
            call_timeout_ms: DEFAULT_CALL_TIMEOUT_MS,
        }
    }

    /// Set the tool call timeout in milliseconds.
    pub fn set_call_timeout(&mut self, timeout_ms: u64) {
        self.call_timeout_ms = timeout_ms;
    }

    // ─── Lifecycle ───────────────────────────────────────────────────────

    /// Start all configured servers and build the tool registry.
    ///
    /// Returns a list of servers that failed to start (partial startup is OK).
    pub async fn start_all(&mut self) -> Vec<(String, McpError)> {
        let (servers, errors) =
            lifecycle::spawn_all_servers(&self.configs, self.working_dir.as_deref()).await;

        // Build registry from all successfully started servers
        for (name, server) in &servers {
            self.registry
                .register_server_tools(name, server.tools.clone());
        }

        self.servers = servers;
        errors
    }

    /// Start a specific server by name.
    pub async fn start_server(&mut self, name: &str) -> Result<(), McpError> {
        let config = self.configs.get(name).ok_or(McpError::ConfigError {
            reason: format!("no configuration for server '{name}'"),
        })?;

        let server =
            lifecycle::spawn_server(name, config, self.working_dir.as_deref()).await?;

        self.registry
            .register_server_tools(name, server.tools.clone());
        self.servers.insert(name.to_string(), server);

        Ok(())
    }

    /// Shut down all servers gracefully.
    pub async fn shutdown_all(&mut self) {
        lifecycle::shutdown_all_servers(&mut self.servers).await;
        self.registry = ToolRegistry::new();
    }

    /// Shut down a specific server.
    pub async fn shutdown_server(&mut self, name: &str) {
        if let Some(mut server) = self.servers.remove(name) {
            let _ = server.shutdown().await;
        }
        self.registry.unregister_server(name);
    }

    // ─── Tool Execution ──────────────────────────────────────────────────

    /// Execute a tool call, routing to the appropriate server.
    ///
    /// Steps:
    /// 1. Validate the tool exists and arguments are structurally valid
    /// 2. Find the owning server
    /// 3. Send JSON-RPC `tools/call` request
    /// 4. Parse and return the result
    pub async fn call_tool(
        &mut self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult, McpError> {
        let start = Instant::now();

        // 1. Validate
        self.registry.validate_tool_call(tool_name, &arguments)?;

        // 2. Find server
        let server_name = self
            .registry
            .get_server_for_tool(tool_name)
            .ok_or(McpError::UnknownTool {
                name: tool_name.to_string(),
            })?
            .to_string();

        let server = self
            .servers
            .get(&server_name)
            .ok_or(McpError::ServerCrashed {
                name: server_name.clone(),
                reason: "server not running".into(),
            })?;

        // 3. Send request
        let params = serde_json::json!({
            "name": tool_name,
            "arguments": arguments,
        });

        let response = tokio::time::timeout(
            std::time::Duration::from_millis(self.call_timeout_ms),
            server.transport.request("tools/call", Some(params)),
        )
        .await
        .map_err(|_| McpError::Timeout {
            tool: tool_name.to_string(),
            timeout_ms: self.call_timeout_ms,
        })?
        .map_err(|e| {
            // Check if this is a transport error (server might have crashed)
            if matches!(e, McpError::TransportError { .. }) {
                McpError::ServerCrashed {
                    name: server_name.clone(),
                    reason: e.to_string(),
                }
            } else {
                e
            }
        })?;

        let elapsed = start.elapsed().as_millis() as u64;

        // 4. Parse response
        match super::transport::extract_result(response) {
            Ok(result) => Ok(ToolCallResult {
                tool_name: tool_name.to_string(),
                success: true,
                result: Some(result),
                error: None,
                execution_time_ms: elapsed,
            }),
            Err(McpError::ServerError { code, message, .. }) => Ok(ToolCallResult {
                tool_name: tool_name.to_string(),
                success: false,
                result: None,
                error: Some(format!("[{code}] {message}")),
                execution_time_ms: elapsed,
            }),
            Err(e) => Err(e),
        }
    }

    /// Restart a crashed server and re-register its tools.
    pub async fn restart_server(&mut self, name: &str) -> Result<(), McpError> {
        let config = self.configs.get(name).ok_or(McpError::ConfigError {
            reason: format!("no configuration for server '{name}'"),
        })?;

        let restart_count = self
            .servers
            .get(name)
            .map(|s| s.restart_count())
            .unwrap_or(0);

        // Remove the old server
        self.registry.unregister_server(name);
        if let Some(mut old) = self.servers.remove(name) {
            let _ = old.shutdown().await;
        }

        // Restart with backoff
        let server = lifecycle::restart_server(
            name,
            config,
            self.working_dir.as_deref(),
            restart_count,
        )
        .await?;

        self.registry
            .register_server_tools(name, server.tools.clone());
        self.servers.insert(name.to_string(), server);

        Ok(())
    }

    // ─── Status ──────────────────────────────────────────────────────────

    /// Get the number of running servers.
    pub fn running_server_count(&self) -> usize {
        self.servers.len()
    }

    /// Get the number of registered tools.
    pub fn tool_count(&self) -> usize {
        self.registry.len()
    }

    /// Check if a specific server is running.
    pub fn is_server_running(&self, name: &str) -> bool {
        self.servers.contains_key(name)
    }

    /// Get a list of running server names.
    pub fn running_servers(&self) -> Vec<String> {
        self.servers.keys().cloned().collect()
    }

    /// Get names of all configured servers (including those that failed to start).
    pub fn configured_servers(&self) -> Vec<String> {
        let mut names: Vec<String> = self.configs.keys().cloned().collect();
        names.sort();
        names
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_config() -> McpServersConfig {
        McpServersConfig {
            servers: HashMap::new(),
        }
    }

    #[test]
    fn test_new_client_empty() {
        let client = McpClient::new(empty_config(), None);
        assert_eq!(client.running_server_count(), 0);
        assert_eq!(client.tool_count(), 0);
        assert!(client.registry.is_empty());
    }

    #[test]
    fn test_set_call_timeout() {
        let mut client = McpClient::new(empty_config(), None);
        client.set_call_timeout(5000);
        assert_eq!(client.call_timeout_ms, 5000);
    }

    #[test]
    fn test_is_server_running() {
        let client = McpClient::new(empty_config(), None);
        assert!(!client.is_server_running("filesystem"));
    }

    #[test]
    fn test_running_servers_empty() {
        let client = McpClient::new(empty_config(), None);
        assert!(client.running_servers().is_empty());
    }

    #[test]
    fn test_configured_servers() {
        let mut servers = HashMap::new();
        servers.insert(
            "zeta".to_string(),
            ServerConfig {
                command: "npx".to_string(),
                args: vec![],
                env: HashMap::new(),
                cwd: None,
                venv: None,
            },
        );
        servers.insert(
            "alpha".to_string(),
            ServerConfig {
                command: "npx".to_string(),
                args: vec![],
                env: HashMap::new(),
                cwd: None,
                venv: None,
            },
        );
        let config = McpServersConfig { servers };
        let client = McpClient::new(config, None);

        let names = client.configured_servers();
        assert_eq!(names, vec!["alpha", "zeta"]); // sorted
    }
}
