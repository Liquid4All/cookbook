//! Shared types for the agent core.
//!
//! Conversation messages, session metadata, undo entries, and confirmation
//! types used across the ConversationManager and ToolRouter.

use serde::{Deserialize, Serialize};

use crate::inference::types::{Role, ToolCall};

// ─── Conversation Messages ──────────────────────────────────────────────────

/// A single message stored in conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    /// Auto-incremented row ID (set by DB on insert).
    pub id: i64,
    /// Which session this message belongs to.
    pub session_id: String,
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// Message role: system, user, assistant, or tool.
    pub role: Role,
    /// Text content (user messages, assistant text, system prompt).
    pub content: Option<String>,
    /// Tool calls made by the assistant in this message.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For `tool` role: the ID of the tool call this result belongs to.
    pub tool_call_id: Option<String>,
    /// For `tool` role: the JSON result from executing the tool.
    pub tool_result: Option<serde_json::Value>,
    /// Estimated token count for this message.
    pub token_count: u32,
}

/// Builder for creating conversation messages without specifying DB fields.
#[derive(Debug, Clone)]
pub struct NewMessage {
    /// Message role.
    pub role: Role,
    /// Text content.
    pub content: Option<String>,
    /// Tool calls (assistant messages).
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID (tool result messages).
    pub tool_call_id: Option<String>,
    /// Tool result (tool result messages).
    pub tool_result: Option<serde_json::Value>,
}

// ─── Sessions ───────────────────────────────────────────────────────────────

/// Metadata for a conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier.
    pub id: String,
    /// ISO 8601 creation timestamp.
    pub created_at: String,
    /// ISO 8601 last activity timestamp.
    pub last_activity: String,
    /// Rolling session summary (populated after eviction).
    pub summary: Option<String>,
    /// File paths touched during this session.
    pub files_touched: Vec<String>,
    /// High-level decisions made during this session.
    pub decisions_made: Vec<String>,
}

/// Session summary for context window inclusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// The session this summary belongs to.
    pub session_id: String,
    /// Human-readable summary of past interactions.
    pub summary_text: String,
    /// Files that have been mentioned or modified.
    pub files_touched: Vec<String>,
    /// Decisions the user or assistant has made.
    pub decisions_made: Vec<String>,
}

// ─── Undo Stack ─────────────────────────────────────────────────────────────

/// An entry in the undo stack for a mutable/destructive tool action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoEntry {
    /// Auto-incremented row ID.
    pub id: i64,
    /// Session this entry belongs to.
    pub session_id: String,
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// The tool that was executed.
    pub tool_name: String,
    /// Category of the action: "move", "delete", "create", "write".
    pub action_type: String,
    /// Serialized original state before the action.
    pub original_state: serde_json::Value,
    /// Serialized new state after the action.
    pub new_state: serde_json::Value,
    /// Whether this entry has been undone.
    pub undone: bool,
}

/// Input for creating a new undo entry (no DB fields).
#[derive(Debug, Clone)]
pub struct NewUndoEntry {
    /// The tool that was executed.
    pub tool_name: String,
    /// Category of the action.
    pub action_type: String,
    /// Original state before the action.
    pub original_state: serde_json::Value,
    /// New state after the action.
    pub new_state: serde_json::Value,
}

// ─── Confirmation ───────────────────────────────────────────────────────────

/// Request sent to the frontend for user confirmation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfirmationRequest {
    /// Unique request ID for matching responses.
    pub request_id: String,
    /// The tool being called.
    pub tool_name: String,
    /// The arguments to the tool.
    pub arguments: serde_json::Value,
    /// Human-readable preview of what will happen.
    pub preview: String,
    /// Whether this tool requires confirmation.
    pub confirmation_required: bool,
    /// Whether the action supports undo.
    pub undo_supported: bool,
    /// Whether this is a destructive action (delete, overwrite).
    pub is_destructive: bool,
}

/// Response from the frontend after user decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ConfirmationResponse {
    /// User confirmed the action (Allow Once).
    Confirmed,
    /// User confirmed for the remainder of this session (Allow for Session).
    ConfirmedForSession,
    /// User confirmed permanently — never ask again (Always Allow).
    ConfirmedAlways,
    /// User rejected the action.
    Rejected,
    /// User edited the arguments before confirming.
    #[serde(rename = "edited")]
    EditedAndConfirmed {
        /// Modified arguments.
        new_arguments: serde_json::Value,
    },
}

// ─── Audit Log ──────────────────────────────────────────────────────────────

/// A single entry in the tool execution audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Auto-incremented row ID.
    pub id: i64,
    /// Session this entry belongs to.
    pub session_id: String,
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// The tool that was executed.
    pub tool_name: String,
    /// Arguments passed to the tool.
    pub arguments: serde_json::Value,
    /// Result returned by the tool (if successful).
    pub result: Option<serde_json::Value>,
    /// Execution status.
    pub result_status: AuditStatus,
    /// Whether the user confirmed this action.
    pub user_confirmed: bool,
    /// How long the execution took (ms).
    pub execution_time_ms: u64,
}

/// Status of a tool execution in the audit log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditStatus {
    /// Tool executed successfully.
    Success,
    /// Tool execution returned an error.
    Error,
    /// User rejected the tool call.
    RejectedByUser,
    /// Tool call was skipped (e.g., auto-confirm for read-only).
    Skipped,
}

impl AuditStatus {
    /// Convert to database string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            AuditStatus::Success => "success",
            AuditStatus::Error => "error",
            AuditStatus::RejectedByUser => "rejected_by_user",
            AuditStatus::Skipped => "skipped",
        }
    }

    /// Parse from database string representation.
    pub fn parse(s: &str) -> Self {
        match s {
            "success" => AuditStatus::Success,
            "error" => AuditStatus::Error,
            "rejected_by_user" => AuditStatus::RejectedByUser,
            "skipped" => AuditStatus::Skipped,
            _ => AuditStatus::Error,
        }
    }
}

// ─── Context Window Budget ──────────────────────────────────────────────────

/// Snapshot of the current context window token usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBudget {
    /// Total context window size in tokens.
    pub total: u32,
    /// Tokens used by the system prompt.
    pub system_prompt: u32,
    /// Tokens used by tool definitions.
    pub tool_definitions: u32,
    /// Tokens used by conversation history.
    pub conversation_history: u32,
    /// Tokens reserved for the model's output response.
    pub output_reservation: u32,
    /// Remaining tokens (safety buffer excluded).
    pub remaining: u32,
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_status_roundtrip() {
        for status in [
            AuditStatus::Success,
            AuditStatus::Error,
            AuditStatus::RejectedByUser,
            AuditStatus::Skipped,
        ] {
            assert_eq!(AuditStatus::parse(status.as_str()), status);
        }
    }

    #[test]
    fn test_audit_status_unknown_defaults_to_error() {
        assert_eq!(AuditStatus::parse("unknown"), AuditStatus::Error);
    }

    #[test]
    fn test_confirmation_response_serialization() {
        // Tagged enum with camelCase: {"type": "confirmed"}
        let confirmed = ConfirmationResponse::Confirmed;
        let json = serde_json::to_string(&confirmed).unwrap();
        assert_eq!(json, r#"{"type":"confirmed"}"#);

        let session = ConfirmationResponse::ConfirmedForSession;
        let json = serde_json::to_string(&session).unwrap();
        assert_eq!(json, r#"{"type":"confirmedForSession"}"#);

        let always = ConfirmationResponse::ConfirmedAlways;
        let json = serde_json::to_string(&always).unwrap();
        assert_eq!(json, r#"{"type":"confirmedAlways"}"#);

        let rejected = ConfirmationResponse::Rejected;
        let json = serde_json::to_string(&rejected).unwrap();
        assert_eq!(json, r#"{"type":"rejected"}"#);

        let edited = ConfirmationResponse::EditedAndConfirmed {
            new_arguments: serde_json::json!({"path": "/tmp/new"}),
        };
        let json = serde_json::to_string(&edited).unwrap();
        assert!(json.contains(r#""type":"edited""#));
        assert!(json.contains("/tmp/new"));
    }

    #[test]
    fn test_confirmation_response_deserialization() {
        // Frontend sends {"type": "confirmed"} etc.
        let confirmed: ConfirmationResponse =
            serde_json::from_str(r#"{"type":"confirmed"}"#).unwrap();
        assert!(matches!(confirmed, ConfirmationResponse::Confirmed));

        let session: ConfirmationResponse =
            serde_json::from_str(r#"{"type":"confirmedForSession"}"#).unwrap();
        assert!(matches!(session, ConfirmationResponse::ConfirmedForSession));

        let edited: ConfirmationResponse =
            serde_json::from_str(r#"{"type":"edited","new_arguments":{"path":"/tmp"}}"#).unwrap();
        assert!(matches!(edited, ConfirmationResponse::EditedAndConfirmed { .. }));
    }

    #[test]
    fn test_confirmation_request_serialization() {
        let req = ConfirmationRequest {
            request_id: "r1".to_string(),
            tool_name: "filesystem.write_file".to_string(),
            arguments: serde_json::json!({"path": "/tmp/test.txt"}),
            preview: "Write to file: /tmp/test.txt".to_string(),
            confirmation_required: true,
            undo_supported: true,
            is_destructive: false,
        };
        let json = serde_json::to_string(&req).unwrap();
        // camelCase: requestId, toolName, undoSupported, isDestructive
        assert!(json.contains("requestId"));
        assert!(json.contains("toolName"));
        assert!(json.contains("undoSupported"));
        assert!(json.contains("isDestructive"));
        assert!(json.contains("confirmationRequired"));
        // Should NOT contain snake_case
        assert!(!json.contains("request_id"));
        assert!(!json.contains("tool_name"));
    }

    #[test]
    fn test_new_message_builder() {
        let msg = NewMessage {
            role: Role::User,
            content: Some("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
            tool_result: None,
        };
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.as_deref(), Some("hello"));
    }
}
