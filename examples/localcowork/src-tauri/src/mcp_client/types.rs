//! Shared types for the MCP client.
//!
//! JSON-RPC 2.0 message types and MCP protocol structures.

use serde::{Deserialize, Serialize};

// ─── JSON-RPC 2.0 ───────────────────────────────────────────────────────────

/// JSON-RPC 2.0 request message.
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC request.
    pub fn new(id: u64, method: &str, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        }
    }
}

/// JSON-RPC 2.0 response message (success or error).
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcResponse {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub id: u64,
    pub result: Option<serde_json::Value>,
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

// ─── MCP Protocol Types ──────────────────────────────────────────────────────

/// MCP tool definition as returned by `initialize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(default, alias = "inputSchema")]
    pub params_schema: serde_json::Value,
    #[serde(default)]
    pub returns_schema: serde_json::Value,
    #[serde(default, alias = "confirmationRequired")]
    pub confirmation_required: bool,
    #[serde(default, alias = "undoSupported")]
    pub undo_supported: bool,
}

/// Server configuration from `mcp_servers.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
    /// Per-server working directory (overrides the global working_dir).
    #[serde(default)]
    pub cwd: Option<String>,
    /// Optional Python virtual environment path. When set, `command` is resolved
    /// to `{venv}/bin/{command}` and `VIRTUAL_ENV` + `PATH` are injected.
    #[serde(default)]
    pub venv: Option<String>,
}

/// Top-level MCP servers configuration file.
#[derive(Debug, Clone, Deserialize)]
pub struct McpServersConfig {
    pub servers: std::collections::HashMap<String, ServerConfig>,
}

/// Result of a tool call execution.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallResult {
    pub tool_name: String,
    pub success: bool,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub execution_time_ms: u64,
}

/// MCP initialize response payload.
#[derive(Debug, Clone, Deserialize)]
pub struct InitializeResult {
    #[serde(default)]
    pub capabilities: serde_json::Value,
    #[serde(default)]
    pub tools: Vec<McpToolDefinition>,
    #[serde(default, alias = "serverInfo")]
    pub server_info: Option<ServerInfo>,
}

/// Server info returned in the initialize response.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerInfo {
    pub name: Option<String>,
    pub version: Option<String>,
}

// ─── Standard MCP Error Codes ────────────────────────────────────────────────

/// Well-known JSON-RPC / MCP error codes.
pub mod error_codes {
    /// Invalid JSON was received.
    pub const PARSE_ERROR: i32 = -32700;
    /// The JSON sent is not a valid Request object.
    pub const INVALID_REQUEST: i32 = -32600;
    /// The method does not exist or is not available.
    pub const METHOD_NOT_FOUND: i32 = -32601;
    /// Invalid method parameters.
    pub const INVALID_PARAMS: i32 = -32602;
    /// Internal JSON-RPC error.
    pub const INTERNAL_ERROR: i32 = -32603;
    /// File not found (MCP extension).
    pub const FILE_NOT_FOUND: i32 = -32001;
    /// Permission denied (MCP extension).
    pub const PERMISSION_DENIED: i32 = -32002;
    /// Operation cancelled by user (MCP extension).
    pub const CANCELLED: i32 = -32003;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_rpc_request_serialization() {
        let req = JsonRpcRequest::new(1, "initialize", None);
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"method\":\"initialize\""));
        // params should be omitted when None
        assert!(!json.contains("params"));
    }

    #[test]
    fn test_json_rpc_request_with_params() {
        let params = serde_json::json!({"name": "test.tool", "arguments": {"path": "/tmp"}});
        let req = JsonRpcRequest::new(42, "tools/call", Some(params));
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"id\":42"));
        assert!(json.contains("tools/call"));
        assert!(json.contains("/tmp"));
    }

    #[test]
    fn test_json_rpc_response_deserialization() {
        let json = r#"{"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, 1);
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_json_rpc_error_response() {
        let json = r#"{
            "jsonrpc": "2.0",
            "id": 2,
            "result": null,
            "error": {"code": -32601, "message": "Method not found"}
        }"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.error.is_some());
        let err = resp.error.unwrap();
        assert_eq!(err.code, error_codes::METHOD_NOT_FOUND);
    }

    #[test]
    fn test_tool_definition_defaults() {
        let json = r#"{"name": "test.tool", "description": "A test tool"}"#;
        let tool: McpToolDefinition = serde_json::from_str(json).unwrap();
        assert!(!tool.confirmation_required);
        assert!(!tool.undo_supported);
    }
}
