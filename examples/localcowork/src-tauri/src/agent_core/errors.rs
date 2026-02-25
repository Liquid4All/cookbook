//! Agent Core error types.

use thiserror::Error;

/// Errors that can occur during agent core operations.
#[derive(Debug, Error)]
pub enum AgentError {
    /// Database operation failed.
    #[error("database error: {reason}")]
    DatabaseError { reason: String },

    /// Session not found.
    #[error("session not found: '{session_id}'")]
    SessionNotFound { session_id: String },

    /// Context window budget exceeded.
    #[error("context window budget exceeded: {used} / {limit} tokens")]
    ContextOverflow { used: u32, limit: u32 },

    /// Token counting failed.
    #[error("token estimation error: {reason}")]
    TokenEstimationError { reason: String },

    /// Tool execution error (wraps McpError).
    #[error("tool execution failed: {reason}")]
    ToolExecutionError { reason: String },

    /// Tool call rejected by user.
    #[error("tool call '{tool_name}' rejected by user")]
    ToolCallRejected { tool_name: String },

    /// Undo operation failed.
    #[error("undo failed for entry {undo_id}: {reason}")]
    UndoFailed { undo_id: i64, reason: String },

    /// No undo entries available.
    #[error("no undo entries in session '{session_id}'")]
    NoUndoEntries { session_id: String },

    /// Confirmation channel error.
    #[error("confirmation channel error: {reason}")]
    ConfirmationError { reason: String },

    /// Audit log error.
    #[error("audit log error: {reason}")]
    AuditError { reason: String },

    /// Serialization error.
    #[error("serialization error: {reason}")]
    SerializationError { reason: String },
}

impl From<rusqlite::Error> for AgentError {
    fn from(e: rusqlite::Error) -> Self {
        AgentError::DatabaseError {
            reason: e.to_string(),
        }
    }
}

impl From<serde_json::Error> for AgentError {
    fn from(e: serde_json::Error) -> Self {
        AgentError::SerializationError {
            reason: e.to_string(),
        }
    }
}
