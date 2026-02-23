//! ToolRouter — dispatches model tool calls to MCP servers.
//!
//! The ToolRouter is the bridge between the LLM's tool call decisions and the
//! MCP server ecosystem. It handles:
//! - Validation (tool exists, arguments match schema)
//! - Confirmation flow (read-only → auto, mutable → confirm, destructive → warn)
//! - Execution via McpClient
//! - Retry with exponential backoff for transient errors
//! - Audit logging of every tool execution
//! - Undo stack entries for mutable/destructive actions

use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::agent_core::tokens::truncate_utf8;
use crate::inference::types::ToolCall;
use crate::mcp_client::errors::McpError;
use crate::mcp_client::types::ToolCallResult;
use crate::mcp_client::McpClient;

use super::conversation::ConversationManager;
use super::permissions::{PermissionScope, PermissionStatus, PermissionStore};
use super::types::{
    AuditStatus, ConfirmationRequest, ConfirmationResponse, NewUndoEntry,
};

// ─── Constants ──────────────────────────────────────────────────────────────

/// Maximum retry attempts for transient tool execution errors.
const MAX_RETRIES: u32 = 2;

/// Base delay between retries (doubles each attempt).
const RETRY_BASE_DELAY: Duration = Duration::from_millis(500);

// ─── ToolRouter ─────────────────────────────────────────────────────────────

/// Dispatches tool calls from the model to MCP servers and manages
/// the human-in-the-loop confirmation flow with tiered permissions.
pub struct ToolRouter {
    /// Sender for confirmation requests (to the frontend).
    confirm_tx: mpsc::Sender<ConfirmationRequest>,
    /// Receiver for confirmation responses (from the frontend).
    confirm_rx: mpsc::Receiver<ConfirmationResponse>,
    /// Tiered permission grants (session + persistent).
    pub permissions: PermissionStore,
}

impl ToolRouter {
    /// Create a new ToolRouter with confirmation channels.
    ///
    /// The caller must wire the other end of these channels to the frontend
    /// (via Tauri IPC events).
    pub fn new(
        confirm_tx: mpsc::Sender<ConfirmationRequest>,
        confirm_rx: mpsc::Receiver<ConfirmationResponse>,
    ) -> Self {
        Self {
            confirm_tx,
            confirm_rx,
            permissions: PermissionStore::new(),
        }
    }

    /// Create a ToolRouter for testing (no confirmation flow — auto-confirms).
    #[cfg(test)]
    pub fn new_auto_confirm() -> (Self, mpsc::Sender<ConfirmationResponse>, mpsc::Receiver<ConfirmationRequest>) {
        let (req_tx, req_rx) = mpsc::channel(16);
        let (resp_tx, resp_rx) = mpsc::channel(16);
        (
            Self {
                confirm_tx: req_tx,
                confirm_rx: resp_rx,
                permissions: PermissionStore::new_in_memory(),
            },
            resp_tx,
            req_rx,
        )
    }

    // ─── Dispatch ───────────────────────────────────────────────────────

    /// Dispatch a batch of tool calls from the model.
    ///
    /// Processes tool calls sequentially (model expects ordered results).
    /// Returns a result for each tool call.
    pub async fn dispatch_tool_calls(
        &mut self,
        tool_calls: &[ToolCall],
        session_id: &str,
        mcp_client: &mut McpClient,
        conversation: &ConversationManager,
    ) -> Vec<ToolCallResult> {
        let mut results = Vec::new();

        for tc in tool_calls {
            let result = self
                .dispatch_single(tc, session_id, mcp_client, conversation)
                .await;
            results.push(result);
        }

        results
    }

    /// Dispatch a single tool call with full lifecycle:
    /// validate → confirm → execute → audit → undo.
    pub async fn dispatch_single(
        &mut self,
        tool_call: &ToolCall,
        session_id: &str,
        mcp_client: &mut McpClient,
        conversation: &ConversationManager,
    ) -> ToolCallResult {
        let start = Instant::now();

        // 1. Validate
        if let Err(e) = mcp_client.registry.validate_tool_call(
            &tool_call.name,
            &tool_call.arguments,
        ) {
            return self.log_and_return_error(
                &tool_call.name,
                &tool_call.arguments,
                session_id,
                conversation,
                AuditStatus::Error,
                false,
                start,
                &format!("validation failed: {e}"),
            );
        }

        // 2. Check confirmation requirements
        let needs_confirmation = mcp_client.registry.requires_confirmation(&tool_call.name);
        let supports_undo = mcp_client.registry.supports_undo(&tool_call.name);

        // 3. Permission check — skip confirmation if tool has an active grant
        if needs_confirmation
            && self.permissions.check(&tool_call.name) == PermissionStatus::Allowed
        {
            tracing::debug!(
                tool = %tool_call.name,
                "skipping confirmation — permission granted"
            );
            // Fall through to execution
        } else if needs_confirmation {
            // 4. Confirmation flow
            let preview = generate_preview(&tool_call.name, &tool_call.arguments);
            let is_destructive = is_destructive_action(&tool_call.name);

            let request = ConfirmationRequest {
                request_id: uuid::Uuid::new_v4().to_string(),
                tool_name: tool_call.name.clone(),
                arguments: tool_call.arguments.clone(),
                preview,
                confirmation_required: true,
                undo_supported: supports_undo,
                is_destructive,
            };

            // Send confirmation request to frontend
            if self.confirm_tx.send(request).await.is_err() {
                return self.log_and_return_error(
                    &tool_call.name,
                    &tool_call.arguments,
                    session_id,
                    conversation,
                    AuditStatus::Error,
                    false,
                    start,
                    "failed to send confirmation request to frontend",
                );
            }

            // Wait for user response
            match self.confirm_rx.recv().await {
                Some(ConfirmationResponse::Confirmed) => {
                    // Allow Once — proceed with execution, no grant stored
                }
                Some(ConfirmationResponse::ConfirmedForSession) => {
                    // Allow for Session — grant + proceed
                    self.permissions
                        .grant(&tool_call.name, PermissionScope::Session);
                }
                Some(ConfirmationResponse::ConfirmedAlways) => {
                    // Always Allow — persistent grant + proceed
                    self.permissions
                        .grant(&tool_call.name, PermissionScope::Always);
                }
                Some(ConfirmationResponse::EditedAndConfirmed { new_arguments }) => {
                    // Execute with modified arguments
                    return self
                        .execute_tool(
                            &tool_call.name,
                            &new_arguments,
                            &tool_call.id,
                            session_id,
                            mcp_client,
                            conversation,
                            supports_undo,
                            start,
                        )
                        .await;
                }
                Some(ConfirmationResponse::Rejected) => {
                    return self.log_and_return_error(
                        &tool_call.name,
                        &tool_call.arguments,
                        session_id,
                        conversation,
                        AuditStatus::RejectedByUser,
                        false,
                        start,
                        "user rejected the tool call",
                    );
                }
                None => {
                    return self.log_and_return_error(
                        &tool_call.name,
                        &tool_call.arguments,
                        session_id,
                        conversation,
                        AuditStatus::Error,
                        false,
                        start,
                        "confirmation channel closed",
                    );
                }
            }
        }

        // 4. Execute
        self.execute_tool(
            &tool_call.name,
            &tool_call.arguments,
            &tool_call.id,
            session_id,
            mcp_client,
            conversation,
            supports_undo,
            start,
        )
        .await
    }

    // ─── Execution ──────────────────────────────────────────────────────

    /// Execute a tool call with retry logic.
    #[allow(clippy::too_many_arguments)]
    async fn execute_tool(
        &self,
        tool_name: &str,
        arguments: &serde_json::Value,
        _tool_call_id: &str,
        session_id: &str,
        mcp_client: &mut McpClient,
        conversation: &ConversationManager,
        supports_undo: bool,
        start: Instant,
    ) -> ToolCallResult {
        let mut last_error: Option<String> = None;

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let delay = RETRY_BASE_DELAY * 2u32.pow(attempt - 1);
                tokio::time::sleep(delay).await;
            }

            match mcp_client.call_tool(tool_name, arguments.clone()).await {
                Ok(result) => {
                    let elapsed = start.elapsed().as_millis() as u64;

                    // Audit log
                    let _ = conversation.db().insert_audit_entry(
                        session_id,
                        tool_name,
                        arguments,
                        result.result.as_ref(),
                        if result.success {
                            AuditStatus::Success
                        } else {
                            AuditStatus::Error
                        },
                        true, // user confirmed (or auto-confirmed)
                        elapsed,
                    );

                    // Undo stack
                    if supports_undo && result.success {
                        let undo = NewUndoEntry {
                            tool_name: tool_name.to_string(),
                            action_type: infer_action_type(tool_name),
                            original_state: capture_original_state(tool_name, arguments),
                            new_state: capture_new_state(tool_name, &result),
                        };
                        let _ = conversation.push_undo(session_id, &undo);
                    }

                    return ToolCallResult {
                        tool_name: tool_name.to_string(),
                        success: result.success,
                        result: result.result,
                        error: result.error,
                        execution_time_ms: elapsed,
                    };
                }
                Err(e) => {
                    if is_retriable_mcp_error(&e) && attempt < MAX_RETRIES {
                        last_error = Some(e.to_string());
                        continue;
                    }

                    return self.log_and_return_error(
                        tool_name,
                        arguments,
                        session_id,
                        conversation,
                        AuditStatus::Error,
                        true,
                        start,
                        &e.to_string(),
                    );
                }
            }
        }

        // All retries exhausted
        self.log_and_return_error(
            tool_name,
            arguments,
            session_id,
            conversation,
            AuditStatus::Error,
            true,
            start,
            &last_error.unwrap_or_else(|| "all retries exhausted".to_string()),
        )
    }

    // ─── Helpers ────────────────────────────────────────────────────────

    /// Log an error result to the audit log and return a ToolCallResult.
    #[allow(clippy::too_many_arguments)]
    fn log_and_return_error(
        &self,
        tool_name: &str,
        arguments: &serde_json::Value,
        session_id: &str,
        conversation: &ConversationManager,
        status: AuditStatus,
        user_confirmed: bool,
        start: Instant,
        error_msg: &str,
    ) -> ToolCallResult {
        let elapsed = start.elapsed().as_millis() as u64;

        let _ = conversation.db().insert_audit_entry(
            session_id,
            tool_name,
            arguments,
            None,
            status,
            user_confirmed,
            elapsed,
        );

        ToolCallResult {
            tool_name: tool_name.to_string(),
            success: false,
            result: None,
            error: Some(error_msg.to_string()),
            execution_time_ms: elapsed,
        }
    }
}

// ─── Free Functions ─────────────────────────────────────────────────────────

/// Check if an MCP error is retriable (transient).
fn is_retriable_mcp_error(err: &McpError) -> bool {
    matches!(
        err,
        McpError::Timeout { .. } | McpError::ServerCrashed { .. } | McpError::TransportError { .. }
    )
}

/// Determine if a tool action is destructive (delete, overwrite).
pub fn is_destructive_action(tool_name: &str) -> bool {
    let name = tool_name.split('.').next_back().unwrap_or(tool_name);
    matches!(
        name,
        "delete_file" | "delete_collection" | "delete_task" | "send_draft"
    )
}

/// Infer the action type from a tool name (for undo entries).
fn infer_action_type(tool_name: &str) -> String {
    let name = tool_name.split('.').next_back().unwrap_or(tool_name);
    if name.starts_with("move") {
        "move".to_string()
    } else if name.starts_with("delete") {
        "delete".to_string()
    } else if name.starts_with("create") || name.starts_with("write") {
        "create".to_string()
    } else {
        "write".to_string()
    }
}

/// Generate a human-readable preview for a tool call.
pub fn generate_preview(tool_name: &str, arguments: &serde_json::Value) -> String {
    let name = tool_name.split('.').next_back().unwrap_or(tool_name);

    match name {
        "write_file" => {
            let path = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("<unknown>");
            format!("Write to file: {path}")
        }
        "move_file" => {
            let src = arguments
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("<unknown>");
            let dst = arguments
                .get("destination")
                .and_then(|v| v.as_str())
                .unwrap_or("<unknown>");
            format!("Move: {src} → {dst}")
        }
        "delete_file" => {
            let path = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("<unknown>");
            format!("Delete file: {path}")
        }
        "copy_file" => {
            let src = arguments
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("<unknown>");
            let dst = arguments
                .get("destination")
                .and_then(|v| v.as_str())
                .unwrap_or("<unknown>");
            format!("Copy: {src} → {dst}")
        }
        "create_pdf" | "create_docx" => {
            let path = arguments
                .get("output_path")
                .and_then(|v| v.as_str())
                .unwrap_or("<unknown>");
            format!("Create document: {path}")
        }
        "create_task" => {
            let title = arguments
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("<unknown>");
            format!("Create task: {title}")
        }
        "send_draft" => {
            let to = arguments
                .get("to")
                .and_then(|v| v.as_str())
                .unwrap_or("<unknown>");
            format!("Send email to: {to}")
        }
        _ => {
            // Generic preview
            let args_preview = serde_json::to_string(arguments)
                .unwrap_or_default();
            let truncated = if args_preview.len() > 100 {
                format!("{}...", truncate_utf8(&args_preview, 100))
            } else {
                args_preview
            };
            format!("Execute {tool_name}: {truncated}")
        }
    }
}

/// Capture the original state before a mutable action (for undo).
fn capture_original_state(tool_name: &str, arguments: &serde_json::Value) -> serde_json::Value {
    let name = tool_name.split('.').next_back().unwrap_or(tool_name);
    match name {
        "move_file" => serde_json::json!({
            "path": arguments.get("source"),
        }),
        "delete_file" => serde_json::json!({
            "path": arguments.get("path"),
        }),
        "write_file" => serde_json::json!({
            "path": arguments.get("path"),
            "existed_before": true,
        }),
        _ => serde_json::json!({
            "tool": tool_name,
            "arguments": arguments,
        }),
    }
}

/// Capture the new state after a mutable action (for undo).
fn capture_new_state(tool_name: &str, result: &ToolCallResult) -> serde_json::Value {
    serde_json::json!({
        "tool": tool_name,
        "success": result.success,
        "result": result.result,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_destructive_action() {
        assert!(is_destructive_action("filesystem.delete_file"));
        assert!(is_destructive_action("email.send_draft"));
        assert!(!is_destructive_action("filesystem.list_dir"));
        assert!(!is_destructive_action("filesystem.write_file"));
    }

    #[test]
    fn test_infer_action_type() {
        assert_eq!(infer_action_type("filesystem.move_file"), "move");
        assert_eq!(infer_action_type("filesystem.delete_file"), "delete");
        assert_eq!(infer_action_type("filesystem.create_pdf"), "create");
        assert_eq!(infer_action_type("filesystem.write_file"), "create");
        assert_eq!(infer_action_type("filesystem.copy_file"), "write");
    }

    #[test]
    fn test_generate_preview_write() {
        let args = serde_json::json!({"path": "/tmp/file.txt", "content": "hello"});
        let preview = generate_preview("filesystem.write_file", &args);
        assert!(preview.contains("/tmp/file.txt"));
    }

    #[test]
    fn test_generate_preview_move() {
        let args = serde_json::json!({"source": "/old/a.txt", "destination": "/new/a.txt"});
        let preview = generate_preview("filesystem.move_file", &args);
        assert!(preview.contains("/old/a.txt"));
        assert!(preview.contains("/new/a.txt"));
    }

    #[test]
    fn test_generate_preview_delete() {
        let args = serde_json::json!({"path": "/tmp/old.txt"});
        let preview = generate_preview("filesystem.delete_file", &args);
        assert!(preview.contains("Delete file"));
        assert!(preview.contains("/tmp/old.txt"));
    }

    #[test]
    fn test_generate_preview_generic() {
        let args = serde_json::json!({"query": "SELECT * FROM users"});
        let preview = generate_preview("data.query_sqlite", &args);
        assert!(preview.contains("data.query_sqlite"));
    }

    #[test]
    fn test_capture_original_state_move() {
        let args = serde_json::json!({"source": "/a.txt", "destination": "/b.txt"});
        let state = capture_original_state("filesystem.move_file", &args);
        assert!(state.get("path").is_some());
    }

    #[test]
    fn test_is_retriable_mcp_error() {
        assert!(is_retriable_mcp_error(&McpError::Timeout {
            tool: "t".into(),
            timeout_ms: 1000,
        }));
        assert!(is_retriable_mcp_error(&McpError::ServerCrashed {
            name: "s".into(),
            reason: "gone".into(),
        }));
        assert!(!is_retriable_mcp_error(&McpError::UnknownTool {
            name: "x".into(),
        }));
    }

    #[tokio::test]
    async fn test_dispatch_rejected_tool_call() {
        use crate::agent_core::database::AgentDatabase;
        use crate::agent_core::conversation::ConversationManager;
        use crate::mcp_client::types::{McpServersConfig, McpToolDefinition};
        use std::collections::HashMap;

        // Set up infrastructure
        let db = AgentDatabase::open(":memory:").unwrap();
        let conv = ConversationManager::new(db);
        conv.new_session("s1", "system").unwrap();

        let mcp_config = McpServersConfig {
            servers: HashMap::new(),
        };
        let mut mcp = McpClient::new(mcp_config, None);

        // Register a tool that requires confirmation
        mcp.registry.register_server_tools(
            "filesystem",
            vec![McpToolDefinition {
                name: "write_file".to_string(),
                description: "Write a file".to_string(),
                params_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]
                }),
                returns_schema: serde_json::json!({}),
                confirmation_required: true,
                undo_supported: true,
            }],
        );

        // Create router with auto-reject
        let (mut router, resp_tx, mut req_rx) = ToolRouter::new_auto_confirm();

        let tc = ToolCall {
            id: "call_1".to_string(),
            name: "filesystem.write_file".to_string(),
            arguments: serde_json::json!({"path": "/tmp/test.txt", "content": "hello"}),
        };

        // Spawn a task that rejects the confirmation
        tokio::spawn(async move {
            let _req = req_rx.recv().await.unwrap();
            resp_tx.send(ConfirmationResponse::Rejected).await.unwrap();
        });

        let result = router
            .dispatch_single(&tc, "s1", &mut mcp, &conv)
            .await;

        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("rejected"));

        // Check audit log
        let entries = conv.db().get_audit_entries("s1").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].result_status, AuditStatus::RejectedByUser);
    }

    #[tokio::test]
    async fn test_dispatch_validation_failure() {
        use crate::agent_core::database::AgentDatabase;
        use crate::agent_core::conversation::ConversationManager;
        use crate::mcp_client::types::McpServersConfig;
        use std::collections::HashMap;

        let db = AgentDatabase::open(":memory:").unwrap();
        let conv = ConversationManager::new(db);
        conv.new_session("s1", "system").unwrap();

        let mcp_config = McpServersConfig {
            servers: HashMap::new(),
        };
        let mut mcp = McpClient::new(mcp_config, None);

        let (mut router, _resp_tx, _req_rx) = ToolRouter::new_auto_confirm();

        // Tool doesn't exist — should fail validation
        let tc = ToolCall {
            id: "call_1".to_string(),
            name: "nonexistent.tool".to_string(),
            arguments: serde_json::json!({}),
        };

        let result = router.dispatch_single(&tc, "s1", &mut mcp, &conv).await;
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("validation failed"));
    }

    #[test]
    fn test_permission_check_skips_confirmation() {
        // Verify that a granted permission changes the check result
        let (mut router, _resp_tx, _req_rx) = ToolRouter::new_auto_confirm();

        // Initially needs confirmation
        assert_eq!(
            router.permissions.check("filesystem.write_file"),
            PermissionStatus::NeedsConfirmation
        );

        // Grant session permission
        router
            .permissions
            .grant("filesystem.write_file", PermissionScope::Session);

        // Now allowed
        assert_eq!(
            router.permissions.check("filesystem.write_file"),
            PermissionStatus::Allowed
        );

        // Clear session — back to needing confirmation
        router.permissions.clear_session();
        assert_eq!(
            router.permissions.check("filesystem.write_file"),
            PermissionStatus::NeedsConfirmation
        );
    }

    #[test]
    fn test_permission_always_grant_survives_session_clear() {
        let (mut router, _resp_tx, _req_rx) = ToolRouter::new_auto_confirm();

        router
            .permissions
            .grant("filesystem.write_file", PermissionScope::Always);
        router.permissions.clear_session();

        // Always grant should survive session clear
        assert_eq!(
            router.permissions.check("filesystem.write_file"),
            PermissionStatus::Allowed
        );
    }
}
