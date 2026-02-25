//! Agent Core — orchestration layer for LocalCowork.
//!
//! Submodules:
//! - `conversation`: Conversation history and context management
//! - `tool_router`: Dispatches model tool calls to MCP servers
//! - `tokens`: Token estimation for context window budgets
//! - `database`: SQLite persistence for sessions, messages, undo stack, audit
//! - `permissions`: Tiered permission grants (once / session / always)
//! - `response_analysis`: Detect incomplete tasks, deflection (FM-3), completion
//! - `orchestrator`: Dual-model pipeline — planner (24B) + router (1.2B) (ADR-009)
//! - `plan_parser`: Bracket + JSON plan output parsers for the orchestrator
//! - `tool_prefilter`: RAG pre-filter for tool selection (ADR-010 / ADR-009)
//! - `types`: Shared types across the agent core
//! - `errors`: Agent-level error types

pub mod conversation;
pub mod database;
pub mod errors;
pub mod orchestrator;
pub mod plan_parser;
pub mod plan_templates;
pub mod permissions;
pub mod response_analysis;
pub mod tokens;
pub mod tool_prefilter;
pub mod tool_router;
pub mod types;

// Re-exports for convenience
pub use conversation::ConversationManager;
pub use database::AgentDatabase;
pub use errors::AgentError;
pub use permissions::PermissionStore;
pub use tool_router::ToolRouter;
pub use types::{
    AuditEntry, AuditStatus, ConfirmationRequest, ConfirmationResponse, ContextBudget,
    ConversationMessage, NewMessage, NewUndoEntry, Session, SessionSummary, UndoEntry,
};
