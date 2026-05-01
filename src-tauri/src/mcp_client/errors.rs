//! MCP Client error types.

use thiserror::Error;

/// Errors that can occur during MCP client operations.
#[derive(Debug, Error)]
pub enum McpError {
    /// A server process failed to start.
    #[error("failed to spawn server '{name}': {reason}")]
    SpawnFailed {
        name: String,
        reason: String,
    },

    /// The initialization handshake failed.
    #[error("server '{name}' initialization failed: {reason}")]
    InitFailed {
        name: String,
        reason: String,
    },

    /// JSON-RPC communication error (malformed message, I/O error).
    #[error("transport error for server '{server}': {reason}")]
    TransportError {
        server: String,
        reason: String,
    },

    /// Server returned a JSON-RPC error response.
    #[error("server error [{code}]: {message}")]
    ServerError {
        code: i32,
        message: String,
        data: Option<serde_json::Value>,
    },

    /// Tool not found in the aggregated registry.
    #[error("unknown tool: '{name}'")]
    UnknownTool {
        name: String,
    },

    /// Tool call arguments failed schema validation.
    #[error("invalid arguments for '{tool}': {reason}")]
    InvalidArguments {
        tool: String,
        reason: String,
    },

    /// A tool call timed out.
    #[error("tool call '{tool}' timed out after {timeout_ms}ms")]
    Timeout {
        tool: String,
        timeout_ms: u64,
    },

    /// Server process crashed unexpectedly.
    #[error("server '{name}' crashed: {reason}")]
    ServerCrashed {
        name: String,
        reason: String,
    },

    /// Configuration error (missing servers, bad config file).
    #[error("config error: {reason}")]
    ConfigError {
        reason: String,
    },

    /// All restart attempts exhausted for a server.
    #[error("server '{name}' failed after {attempts} restart attempts")]
    RestartExhausted {
        name: String,
        attempts: u32,
    },
}
