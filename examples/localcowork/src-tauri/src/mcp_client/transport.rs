//! JSON-RPC over stdio transport.
//!
//! Handles low-level communication with MCP server child processes:
//! - Writing JSON-RPC requests to stdin
//! - Reading JSON-RPC responses from stdout
//! - Line-delimited JSON protocol (one JSON object per line)

use std::sync::atomic::{AtomicU64, Ordering};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout};
use tokio::sync::Mutex;

use super::errors::McpError;
use super::types::{JsonRpcRequest, JsonRpcResponse};

// ─── Request ID Generator ────────────────────────────────────────────────────

/// Global monotonic request ID counter.
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Generate a unique request ID.
pub fn next_request_id() -> u64 {
    NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed)
}

// ─── Transport ───────────────────────────────────────────────────────────────

/// Bi-directional JSON-RPC transport over a child process's stdio.
pub struct StdioTransport {
    server_name: String,
    writer: Mutex<ChildStdin>,
    reader: Mutex<BufReader<ChildStdout>>,
}

impl StdioTransport {
    /// Create a new transport from a child process's stdin/stdout.
    pub fn new(server_name: &str, stdin: ChildStdin, stdout: ChildStdout) -> Self {
        Self {
            server_name: server_name.to_string(),
            writer: Mutex::new(stdin),
            reader: Mutex::new(BufReader::new(stdout)),
        }
    }

    /// Send a JSON-RPC request and wait for the matching response.
    ///
    /// This is a simple request-response pattern: write one line of JSON,
    /// read lines until we get a response with a matching `id`.
    pub async fn request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<JsonRpcResponse, McpError> {
        let id = next_request_id();
        let req = JsonRpcRequest::new(id, method, params);

        // Serialize and send
        let mut json = serde_json::to_string(&req).map_err(|e| McpError::TransportError {
            server: self.server_name.clone(),
            reason: format!("failed to serialize request: {e}"),
        })?;
        json.push('\n');

        {
            let mut writer = self.writer.lock().await;
            writer
                .write_all(json.as_bytes())
                .await
                .map_err(|e| McpError::TransportError {
                    server: self.server_name.clone(),
                    reason: format!("failed to write to stdin: {e}"),
                })?;
            writer
                .flush()
                .await
                .map_err(|e| McpError::TransportError {
                    server: self.server_name.clone(),
                    reason: format!("failed to flush stdin: {e}"),
                })?;
        }

        // Read response lines until we find one with matching id
        let mut line_buf = String::new();
        let mut reader = self.reader.lock().await;

        loop {
            line_buf.clear();
            let bytes_read = reader
                .read_line(&mut line_buf)
                .await
                .map_err(|e| McpError::TransportError {
                    server: self.server_name.clone(),
                    reason: format!("failed to read from stdout: {e}"),
                })?;

            if bytes_read == 0 {
                return Err(McpError::TransportError {
                    server: self.server_name.clone(),
                    reason: "server stdout closed (process may have exited)".into(),
                });
            }

            let trimmed = line_buf.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Try to parse as JSON-RPC response
            match serde_json::from_str::<JsonRpcResponse>(trimmed) {
                Ok(resp) if resp.id == id => return Ok(resp),
                Ok(_) => {
                    // Response for a different request ID — skip
                    // This shouldn't happen in our single-threaded protocol,
                    // but handle gracefully.
                    continue;
                }
                Err(_) => {
                    // Not a JSON-RPC response — could be server log output.
                    // Skip and keep reading.
                    continue;
                }
            }
        }
    }

    /// Send a JSON-RPC notification (no response expected).
    pub async fn notify(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), McpError> {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });

        let mut json = serde_json::to_string(&notification).map_err(|e| {
            McpError::TransportError {
                server: self.server_name.clone(),
                reason: format!("failed to serialize notification: {e}"),
            }
        })?;
        json.push('\n');

        let mut writer = self.writer.lock().await;
        writer
            .write_all(json.as_bytes())
            .await
            .map_err(|e| McpError::TransportError {
                server: self.server_name.clone(),
                reason: format!("failed to write notification: {e}"),
            })?;
        writer
            .flush()
            .await
            .map_err(|e| McpError::TransportError {
                server: self.server_name.clone(),
                reason: format!("failed to flush notification: {e}"),
            })?;

        Ok(())
    }
}

// ─── Response Helpers ────────────────────────────────────────────────────────

/// Extract the result from a JSON-RPC response, converting errors to `McpError`.
pub fn extract_result(response: JsonRpcResponse) -> Result<serde_json::Value, McpError> {
    if let Some(err) = response.error {
        return Err(McpError::ServerError {
            code: err.code,
            message: err.message,
            data: err.data,
        });
    }

    response.result.ok_or(McpError::ServerError {
        code: -32603,
        message: "response missing both result and error".into(),
        data: None,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_request_id_is_monotonic() {
        let id1 = next_request_id();
        let id2 = next_request_id();
        assert!(id2 > id1);
    }

    #[test]
    fn test_extract_result_success() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: 1,
            result: Some(serde_json::json!({"text": "hello"})),
            error: None,
        };
        let result = extract_result(resp).unwrap();
        assert_eq!(result["text"], "hello");
    }

    #[test]
    fn test_extract_result_error() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: 1,
            result: None,
            error: Some(super::super::types::JsonRpcError {
                code: -32601,
                message: "Method not found".into(),
                data: None,
            }),
        };
        let err = extract_result(resp).unwrap_err();
        match err {
            McpError::ServerError { code, message, .. } => {
                assert_eq!(code, -32601);
                assert_eq!(message, "Method not found");
            }
            _ => panic!("expected ServerError"),
        }
    }

    #[test]
    fn test_extract_result_missing_both() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: 1,
            result: None,
            error: None,
        };
        let err = extract_result(resp).unwrap_err();
        assert!(matches!(err, McpError::ServerError { .. }));
    }
}
