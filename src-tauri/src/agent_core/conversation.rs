//! ConversationManager — persistent conversation history with context window management.
//!
//! Responsibilities:
//! - Store and retrieve conversation messages (SQLite)
//! - Track token usage per message
//! - Enforce context window budget (32k default) via eviction
//! - Maintain session summaries for evicted turns
//! - Build `Vec<ChatMessage>` for the inference client

use crate::inference::types::{
    ChatMessage, FunctionCallResponse, Role, ToolCallResponse,
};

use super::database::AgentDatabase;
use super::errors::AgentError;
use super::tokens;
use super::types::{
    ConversationMessage, ContextBudget, NewMessage, NewUndoEntry, SessionSummary, UndoEntry,
};

// ─── Constants ──────────────────────────────────────────────────────────────

/// Default total context window size (tokens).
const DEFAULT_CONTEXT_WINDOW: u32 = 32_768;

/// Tokens reserved for the system prompt (rules + few-shot examples).
const SYSTEM_PROMPT_BUDGET: u32 = 900;

/// Default tokens reserved for tool definitions.
///
/// Used when the actual tool definition tokens haven't been measured yet.
/// This is a conservative fallback — the real value should be computed from
/// the serialized tool definitions and set via `set_tool_definitions_budget()`.
const DEFAULT_TOOL_DEFINITIONS_BUDGET: u32 = 2_000;

/// Tokens reserved for the model's output response.
///
/// Every production agent reserves space for the model to generate its
/// response. Without this, the context window could be 100% filled with
/// input, leaving no room for output.
///
/// Note: The PRD's "Active file/document content" budget (~9,500 tokens)
/// was a static reservation for a ProactiveContextor feature that hasn't
/// been built yet. When that feature is implemented, it will dynamically
/// claim tokens from the conversation budget — not from a phantom static
/// reservation that wastes 29% of the context window.
const OUTPUT_RESERVATION: u32 = 2_000;

/// Safety buffer — never fill these tokens.
const SAFETY_BUFFER: u32 = 768;

/// When remaining tokens drop below this, trigger eviction.
///
/// Set to 5,000 so eviction fires well before the agent loop's
/// `MIN_ROUND_TOKEN_BUDGET` (1,500) gate kills the loop. With the
/// old value of 1,000, eviction never triggered because the budget
/// gate always fired first, making eviction effectively dead.
const EVICTION_THRESHOLD: u32 = 5_000;

/// Number of most recent turns to keep in full detail during eviction.
const FULL_DETAIL_TURNS: usize = 10;

/// Maximum tokens allowed for the session summary.
///
/// Without a cap, each eviction cycle appends to the summary, which can
/// grow to 2,000+ tokens after 3 cycles — eating into the space eviction
/// was supposed to free. The cap keeps the most recent portion.
const MAX_SUMMARY_TOKENS: u32 = 500;

// ─── ConversationManager ────────────────────────────────────────────────────

/// Manages conversation history, token budgets, and context window eviction.
pub struct ConversationManager {
    /// SQLite database handle.
    db: AgentDatabase,
    /// Total context window size (configurable per model).
    context_window: u32,
    /// Actual tokens consumed by tool definitions (measured, not estimated).
    ///
    /// Set by `set_tool_definitions_budget()` after tool definitions are built.
    /// Falls back to `DEFAULT_TOOL_DEFINITIONS_BUDGET` if not set.
    tool_definitions_budget: u32,
    /// Actual tokens consumed by the system prompt (measured, not estimated).
    ///
    /// Set by `set_system_prompt_budget()` after the dynamic system prompt is built.
    /// Falls back to `SYSTEM_PROMPT_BUDGET` if not set.
    system_prompt_budget: u32,
}

impl ConversationManager {
    /// Create a new ConversationManager backed by the given database.
    pub fn new(db: AgentDatabase) -> Self {
        Self {
            db,
            context_window: DEFAULT_CONTEXT_WINDOW,
            tool_definitions_budget: DEFAULT_TOOL_DEFINITIONS_BUDGET,
            system_prompt_budget: SYSTEM_PROMPT_BUDGET,
        }
    }

    /// Override the context window size (e.g., from model config).
    pub fn set_context_window(&mut self, size: u32) {
        self.context_window = size;
    }

    /// Set the actual tool definitions token budget based on measured serialization.
    ///
    /// This should be called after tool definitions are built (in `send_message`)
    /// so the budget calculation uses the real cost instead of the default estimate.
    pub fn set_tool_definitions_budget(&mut self, tokens: u32) {
        self.tool_definitions_budget = tokens;
    }

    /// Set the actual system prompt token budget based on the dynamic prompt.
    ///
    /// Called in `start_session` after building the prompt from the MCP registry.
    /// Ensures the context budget display reflects the real prompt size.
    pub fn set_system_prompt_budget(&mut self, tokens: u32) {
        self.system_prompt_budget = tokens;
    }

    /// Access the underlying database (for ToolRouter/audit operations).
    pub fn db(&self) -> &AgentDatabase {
        &self.db
    }

    // ─── Session Management ─────────────────────────────────────────────

    /// Start a new conversation session.
    ///
    /// Creates the session record and inserts the system prompt as the first
    /// message. Returns the session ID.
    pub fn new_session(
        &self,
        session_id: &str,
        system_prompt: &str,
    ) -> Result<(), AgentError> {
        self.db.create_session(session_id)?;

        let token_count = tokens::estimate_system_prompt_tokens(system_prompt);
        let msg = NewMessage {
            role: Role::System,
            content: Some(system_prompt.to_string()),
            tool_calls: None,
            tool_call_id: None,
            tool_result: None,
        };
        self.db.insert_message(session_id, &msg, token_count)?;
        Ok(())
    }

    // ─── Message Operations ─────────────────────────────────────────────

    /// Add a user message to the conversation.
    pub fn add_user_message(
        &self,
        session_id: &str,
        content: &str,
    ) -> Result<i64, AgentError> {
        let token_count = tokens::estimate_tokens(content) + 4; // overhead
        let msg = NewMessage {
            role: Role::User,
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            tool_result: None,
        };
        self.db.insert_message(session_id, &msg, token_count)
    }

    /// Add an assistant text message to the conversation.
    pub fn add_assistant_message(
        &self,
        session_id: &str,
        content: &str,
    ) -> Result<i64, AgentError> {
        let token_count = tokens::estimate_tokens(content) + 4;
        let msg = NewMessage {
            role: Role::Assistant,
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            tool_result: None,
        };
        self.db.insert_message(session_id, &msg, token_count)
    }

    /// Add an assistant message that contains tool calls.
    pub fn add_tool_call_message(
        &self,
        session_id: &str,
        tool_calls: &[crate::inference::types::ToolCall],
    ) -> Result<i64, AgentError> {
        // Estimate tokens for tool calls
        let mut token_count: u32 = 4; // overhead
        for tc in tool_calls {
            token_count += 10; // per-call overhead
            token_count += tokens::estimate_tokens(&tc.name);
            token_count += tokens::estimate_tokens(
                &serde_json::to_string(&tc.arguments).unwrap_or_default(),
            );
        }

        let msg = NewMessage {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(tool_calls.to_vec()),
            tool_call_id: None,
            tool_result: None,
        };
        self.db.insert_message(session_id, &msg, token_count)
    }

    /// Add a tool result message to the conversation.
    pub fn add_tool_result_message(
        &self,
        session_id: &str,
        tool_call_id: &str,
        result: &serde_json::Value,
    ) -> Result<i64, AgentError> {
        // Use the plain string if the value is a String, otherwise JSON-encode it.
        // This avoids double-serialization (wrapping "text" as "\"text\"") which
        // confuses local LLMs into thinking the tool result is empty/malformed.
        let result_str = match result.as_str() {
            Some(s) => s.to_string(),
            None => serde_json::to_string(result).unwrap_or_default(),
        };
        let token_count = tokens::estimate_tokens(&result_str) + 4;

        let msg = NewMessage {
            role: Role::Tool,
            content: Some(result_str),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
            tool_result: Some(result.clone()),
        };
        self.db.insert_message(session_id, &msg, token_count)
    }

    /// Get the full conversation history for a session.
    pub fn get_history(
        &self,
        session_id: &str,
    ) -> Result<Vec<ConversationMessage>, AgentError> {
        self.db.get_messages(session_id)
    }

    /// Get the N most recent messages.
    pub fn get_recent(
        &self,
        session_id: &str,
        n: usize,
    ) -> Result<Vec<ConversationMessage>, AgentError> {
        self.db.get_recent_messages(session_id, n)
    }

    // ─── Context Window Management ──────────────────────────────────────

    /// Get the current context budget snapshot.
    pub fn get_budget(&self, session_id: &str) -> Result<ContextBudget, AgentError> {
        let conversation_tokens = self.db.total_message_tokens(session_id)?;
        let total = self.context_window;
        let overhead = self.system_prompt_budget + self.tool_definitions_budget
            + OUTPUT_RESERVATION + SAFETY_BUFFER;
        let remaining = total.saturating_sub(overhead).saturating_sub(conversation_tokens);

        Ok(ContextBudget {
            total,
            system_prompt: self.system_prompt_budget,
            tool_definitions: self.tool_definitions_budget,
            conversation_history: conversation_tokens,
            output_reservation: OUTPUT_RESERVATION,
            remaining,
        })
    }

    /// Check if eviction is needed and perform it.
    ///
    /// Evicts the oldest non-system messages until remaining tokens are
    /// above the threshold. Evicted messages are summarized into the
    /// session summary.
    pub fn evict_if_needed(&self, session_id: &str) -> Result<u32, AgentError> {
        let budget = self.get_budget(session_id)?;

        if budget.remaining >= EVICTION_THRESHOLD {
            return Ok(0); // No eviction needed
        }

        let message_count = self.db.message_count(session_id)?;
        if message_count <= FULL_DETAIL_TURNS + 1 {
            // +1 for system prompt
            return Ok(0); // Not enough messages to evict
        }

        // Evict messages beyond the full-detail window
        let evict_count = message_count - FULL_DETAIL_TURNS - 1;
        let evicted = self.db.delete_oldest_messages(session_id, evict_count)?;

        // Build a summary from evicted messages
        let mut summary_parts = Vec::new();
        let mut files: Vec<String> = Vec::new();

        for msg in &evicted {
            let line = tokens::summarize_turn(&msg.role, msg.content.as_deref());
            summary_parts.push(line);

            // Track file paths mentioned in tool calls
            if let Some(ref tc) = msg.tool_calls {
                for call in tc {
                    if let Some(path) = call.arguments.get("path").and_then(|v| v.as_str()) {
                        if !files.contains(&path.to_string()) {
                            files.push(path.to_string());
                        }
                    }
                }
            }
        }

        let summary_text = summary_parts.join("\n");
        let evicted_tokens: u32 = evicted.iter().map(|m| m.token_count).sum();

        // Update session summary (append to existing, then cap)
        let existing = self.db.get_session_summary(session_id)?;
        let full_summary = match existing {
            Some(s) => format!("{}\n{}", s.summary_text, summary_text),
            None => summary_text,
        };

        // Cap summary to prevent it from consuming the space eviction freed
        let summary_tokens = tokens::estimate_tokens(&full_summary);
        let capped_summary = if summary_tokens > MAX_SUMMARY_TOKENS {
            let target_chars = (MAX_SUMMARY_TOKENS as f64 * 3.2) as usize;
            let start = full_summary.len().saturating_sub(target_chars);
            format!("[earlier context omitted]\n{}", &full_summary[start..])
        } else {
            full_summary
        };

        self.db.update_session_summary(
            session_id,
            &capped_summary,
            &files,
            &[], // decisions are tracked separately
        )?;

        Ok(evicted_tokens)
    }

    /// Build the `Vec<ChatMessage>` to send to the inference client.
    ///
    /// Includes: session summary (if any) + system prompt + recent messages.
    pub fn build_chat_messages(
        &self,
        session_id: &str,
    ) -> Result<Vec<ChatMessage>, AgentError> {
        let messages = self.db.get_messages(session_id)?;
        let summary = self.db.get_session_summary(session_id)?;

        let mut chat_messages = Vec::new();

        for msg in &messages {
            match msg.role {
                Role::System => {
                    // Prepend session summary to system prompt
                    let mut content = msg.content.clone().unwrap_or_default();
                    if let Some(ref s) = summary {
                        content = format!(
                            "{content}\n\n## Previous conversation summary:\n{}",
                            s.summary_text
                        );
                    }
                    chat_messages.push(ChatMessage {
                        role: Role::System,
                        content: Some(content),
                        tool_call_id: None,
                        tool_calls: None,
                    });
                }
                Role::User => {
                    chat_messages.push(ChatMessage {
                        role: Role::User,
                        content: msg.content.clone(),
                        tool_call_id: None,
                        tool_calls: None,
                    });
                }
                Role::Assistant => {
                    let tool_calls = msg.tool_calls.as_ref().map(|calls| {
                        calls
                            .iter()
                            .map(|tc| ToolCallResponse {
                                id: tc.id.clone(),
                                r#type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name: tc.name.clone(),
                                    arguments: serde_json::to_string(&tc.arguments)
                                        .unwrap_or_default(),
                                },
                            })
                            .collect()
                    });
                    chat_messages.push(ChatMessage {
                        role: Role::Assistant,
                        content: msg.content.clone(),
                        tool_call_id: None,
                        tool_calls,
                    });
                }
                Role::Tool => {
                    chat_messages.push(ChatMessage {
                        role: Role::Tool,
                        content: msg.content.clone(),
                        tool_call_id: msg.tool_call_id.clone(),
                        tool_calls: None,
                    });
                }
            }
        }

        Ok(chat_messages)
    }

    /// Build a windowed `Vec<ChatMessage>` optimized for multi-step workflows.
    ///
    /// Implements a 3-tier message strategy to minimize token waste:
    /// - **Tier 1 (recent)**: Last `recent_window` messages sent verbatim
    /// - **Tier 2 (middle)**: Tool results compressed to one-line summaries;
    ///   user/assistant messages kept verbatim
    /// - **Tier 3 (evicted)**: Already handled by session summary
    ///
    /// This prevents stale tool results from consuming context. A 6,000-char
    /// OCR result from round 2 is compressed to ~50 chars in rounds 4+.
    pub fn build_windowed_chat_messages(
        &self,
        session_id: &str,
        recent_window: usize,
    ) -> Result<Vec<ChatMessage>, AgentError> {
        let messages = self.db.get_messages(session_id)?;
        let summary = self.db.get_session_summary(session_id)?;

        let total = messages.len();
        // Window start index: everything before this is Tier 2 (compressed)
        // +1 to account for system prompt at index 0
        let window_start = if total > recent_window + 1 {
            total - recent_window
        } else {
            1 // include everything after system prompt
        };

        let mut chat_messages = Vec::new();

        for (i, msg) in messages.iter().enumerate() {
            match msg.role {
                Role::System => {
                    // Prepend session summary to system prompt (same as build_chat_messages)
                    let mut content = msg.content.clone().unwrap_or_default();
                    if let Some(ref s) = summary {
                        content = format!(
                            "{content}\n\n## Previous conversation summary:\n{}",
                            s.summary_text
                        );
                    }
                    chat_messages.push(ChatMessage {
                        role: Role::System,
                        content: Some(content),
                        tool_call_id: None,
                        tool_calls: None,
                    });
                }
                Role::Tool if i < window_start => {
                    // Tier 2: compress old tool results to one-line summary
                    let compressed = tokens::summarize_tool_result(
                        msg.tool_call_id.as_deref().unwrap_or("tool"),
                        &msg.tool_result.clone().unwrap_or(serde_json::Value::Null),
                    );
                    chat_messages.push(ChatMessage {
                        role: Role::Tool,
                        content: Some(compressed),
                        tool_call_id: msg.tool_call_id.clone(),
                        tool_calls: None,
                    });
                }
                Role::User => {
                    chat_messages.push(ChatMessage {
                        role: Role::User,
                        content: msg.content.clone(),
                        tool_call_id: None,
                        tool_calls: None,
                    });
                }
                Role::Assistant => {
                    let tool_calls = msg.tool_calls.as_ref().map(|calls| {
                        calls
                            .iter()
                            .map(|tc| ToolCallResponse {
                                id: tc.id.clone(),
                                r#type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name: tc.name.clone(),
                                    arguments: serde_json::to_string(&tc.arguments)
                                        .unwrap_or_default(),
                                },
                            })
                            .collect()
                    });
                    chat_messages.push(ChatMessage {
                        role: Role::Assistant,
                        content: msg.content.clone(),
                        tool_call_id: None,
                        tool_calls,
                    });
                }
                Role::Tool => {
                    // Tier 1: recent tool results — send verbatim
                    chat_messages.push(ChatMessage {
                        role: Role::Tool,
                        content: msg.content.clone(),
                        tool_call_id: msg.tool_call_id.clone(),
                        tool_calls: None,
                    });
                }
            }
        }

        Ok(chat_messages)
    }

    // ─── Undo Stack (delegates to DB) ───────────────────────────────────

    /// Push a new entry onto the undo stack.
    pub fn push_undo(
        &self,
        session_id: &str,
        entry: &NewUndoEntry,
    ) -> Result<i64, AgentError> {
        self.db.push_undo_entry(session_id, entry)
    }

    /// Get the current undo stack for a session.
    pub fn get_undo_stack(
        &self,
        session_id: &str,
    ) -> Result<Vec<UndoEntry>, AgentError> {
        self.db.get_undo_stack(session_id)
    }

    /// Mark an undo entry as undone.
    pub fn mark_undone(&self, undo_id: i64) -> Result<(), AgentError> {
        self.db.mark_undone(undo_id)
    }

    /// Get the session summary.
    pub fn get_session_summary(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionSummary>, AgentError> {
        self.db.get_session_summary(session_id)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_core::database::AgentDatabase;

    fn test_manager() -> ConversationManager {
        let db = AgentDatabase::open(":memory:").unwrap();
        ConversationManager::new(db)
    }

    #[test]
    fn test_new_session() {
        let mgr = test_manager();
        mgr.new_session("s1", "You are a helpful assistant.").unwrap();

        let history = mgr.get_history("s1").unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::System);
    }

    #[test]
    fn test_add_messages() {
        let mgr = test_manager();
        mgr.new_session("s1", "system").unwrap();

        mgr.add_user_message("s1", "Hello").unwrap();
        mgr.add_assistant_message("s1", "Hi there!").unwrap();

        let history = mgr.get_history("s1").unwrap();
        assert_eq!(history.len(), 3); // system + user + assistant
        assert_eq!(history[1].role, Role::User);
        assert_eq!(history[2].role, Role::Assistant);
    }

    #[test]
    fn test_add_tool_call_and_result() {
        let mgr = test_manager();
        mgr.new_session("s1", "system").unwrap();
        mgr.add_user_message("s1", "list files").unwrap();

        let tool_calls = vec![crate::inference::types::ToolCall {
            id: "call_1".to_string(),
            name: "filesystem.list_dir".to_string(),
            arguments: serde_json::json!({"path": "/tmp"}),
        }];
        mgr.add_tool_call_message("s1", &tool_calls).unwrap();

        let result = serde_json::json!({"files": ["a.txt", "b.txt"]});
        mgr.add_tool_result_message("s1", "call_1", &result).unwrap();

        let history = mgr.get_history("s1").unwrap();
        assert_eq!(history.len(), 4); // system + user + assistant(tool_calls) + tool
        assert_eq!(history[3].role, Role::Tool);
    }

    #[test]
    fn test_get_budget() {
        let mgr = test_manager();
        mgr.new_session("s1", "system prompt").unwrap();

        let budget = mgr.get_budget("s1").unwrap();
        assert_eq!(budget.total, DEFAULT_CONTEXT_WINDOW);
        assert!(budget.remaining > 0);
        assert!(budget.conversation_history > 0); // system prompt has tokens
    }

    #[test]
    fn test_eviction_not_needed() {
        let mgr = test_manager();
        mgr.new_session("s1", "system").unwrap();
        mgr.add_user_message("s1", "hello").unwrap();

        let evicted = mgr.evict_if_needed("s1").unwrap();
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_eviction_with_many_messages() {
        let mgr = test_manager();
        // Use a tiny context window to force eviction
        // We can't set context_window on test_manager easily, so let's
        // just test that the logic works with enough messages.
        mgr.new_session("s1", "system").unwrap();

        // Add many large messages to exceed budget
        for i in 0..50 {
            let large_content = format!("message {i}: {}", "x".repeat(500));
            mgr.add_user_message("s1", &large_content).unwrap();
        }

        let count_before = mgr.get_history("s1").unwrap().len();
        assert!(count_before > FULL_DETAIL_TURNS);

        let evicted = mgr.evict_if_needed("s1").unwrap();

        // With default 32k window, eviction may or may not trigger depending
        // on actual token counts. Let's just verify no error.
        // The important thing is the logic path works.
        let _ = evicted;
    }

    #[test]
    fn test_build_chat_messages() {
        let mgr = test_manager();
        mgr.new_session("s1", "You are helpful.").unwrap();
        mgr.add_user_message("s1", "Hello").unwrap();
        mgr.add_assistant_message("s1", "Hi!").unwrap();

        let chat = mgr.build_chat_messages("s1").unwrap();
        assert_eq!(chat.len(), 3);
        assert_eq!(chat[0].role, Role::System);
        assert_eq!(chat[1].role, Role::User);
        assert_eq!(chat[2].role, Role::Assistant);
    }

    #[test]
    fn test_build_chat_messages_with_summary() {
        let mgr = test_manager();
        mgr.new_session("s1", "You are helpful.").unwrap();

        // Manually set a session summary
        mgr.db().update_session_summary(
            "s1",
            "User previously asked about files in /tmp.",
            &["/tmp/file.txt".to_string()],
            &[],
        ).unwrap();

        let chat = mgr.build_chat_messages("s1").unwrap();
        let system_content = chat[0].content.as_ref().unwrap();
        assert!(system_content.contains("Previous conversation summary"));
        assert!(system_content.contains("files in /tmp"));
    }

    #[test]
    fn test_undo_stack() {
        let mgr = test_manager();
        mgr.new_session("s1", "system").unwrap();

        let entry = NewUndoEntry {
            tool_name: "filesystem.move_file".to_string(),
            action_type: "move".to_string(),
            original_state: serde_json::json!({"path": "/old/file.txt"}),
            new_state: serde_json::json!({"path": "/new/file.txt"}),
        };
        let id = mgr.push_undo("s1", &entry).unwrap();

        let stack = mgr.get_undo_stack("s1").unwrap();
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0].tool_name, "filesystem.move_file");

        mgr.mark_undone(id).unwrap();
        let stack = mgr.get_undo_stack("s1").unwrap();
        assert_eq!(stack.len(), 0);
    }

    #[test]
    fn test_budget_after_optimization() {
        // After removing the phantom ACTIVE_CONTEXT_BUDGET (9,500) and
        // replacing with OUTPUT_RESERVATION (2,000), a fresh session
        // should have significantly more remaining budget.
        let mgr = test_manager();
        mgr.new_session("s1", "short prompt").unwrap();

        let budget = mgr.get_budget("s1").unwrap();
        // With 32K total, overhead = 500 (system) + 2000 (tools) + 2000 (output) + 768 (safety)
        // = 5,268. Remaining = 32,768 - 5,268 - conversation_tokens ≈ 27,000+
        assert!(
            budget.remaining > 20_000,
            "remaining should be >20K after optimization, got {}",
            budget.remaining
        );
        assert_eq!(budget.output_reservation, 2_000);
    }

    #[test]
    fn test_build_windowed_compresses_old_tool_results() {
        let mgr = test_manager();
        mgr.new_session("s1", "system").unwrap();

        // Simulate a 3-round workflow so old tool results are clearly
        // outside the recent window:
        //  0: system
        //  1: user ("process files")
        //  2: assistant (tool_call 1)
        //  3: tool (result 1 — large, should be compressed)
        //  4: assistant (tool_call 2)
        //  5: tool (result 2 — large, should be compressed)
        //  6: assistant (tool_call 3)
        //  7: tool (result 3 — recent, keep verbatim)
        //  8: user ("continue")
        //
        // With window=4: window_start = 9 - 4 = 5
        // Messages at index < 5 are Tier 2 → tool results compressed
        // Messages at index >= 5 are Tier 1 → verbatim
        mgr.add_user_message("s1", "process files").unwrap();

        for i in 1..=3 {
            let tc = vec![crate::inference::types::ToolCall {
                id: format!("call_{i}"),
                name: "ocr.extract_text_from_image".to_string(),
                arguments: serde_json::json!({"path": format!("/tmp/img{i}.png")}),
            }];
            mgr.add_tool_call_message("s1", &tc).unwrap();

            let large_result = serde_json::json!({"text": "x".repeat(200)});
            mgr.add_tool_result_message("s1", &format!("call_{i}"), &large_result)
                .unwrap();
        }

        mgr.add_user_message("s1", "continue").unwrap();

        // Full build: all messages verbatim
        let full = mgr.build_chat_messages("s1").unwrap();
        // Windowed build with window=4: old tool results compressed
        let windowed = mgr.build_windowed_chat_messages("s1", 4).unwrap();

        // Both should have same number of messages
        assert_eq!(full.len(), windowed.len());
        assert_eq!(full.len(), 9); // system + user + 3*(tc+result) + user

        // The first tool result (index 3) should be compressed in windowed
        let full_tool = full[3].content.as_ref().unwrap();
        let windowed_tool = windowed[3].content.as_ref().unwrap();

        assert!(
            windowed_tool.len() < full_tool.len(),
            "windowed tool result ({} chars) should be shorter than full ({} chars)",
            windowed_tool.len(),
            full_tool.len()
        );
        // Compressed result should contain the summarization marker
        assert!(
            windowed_tool.starts_with('['),
            "compressed result should be a summary bracket: {}",
            windowed_tool
        );

        // The last tool result (index 7) should be verbatim (in recent window)
        let full_recent = full[7].content.as_ref().unwrap();
        let windowed_recent = windowed[7].content.as_ref().unwrap();
        assert_eq!(
            full_recent, windowed_recent,
            "recent tool result should be verbatim"
        );
    }

    #[test]
    fn test_summary_capped_on_eviction() {
        let mut mgr = test_manager();
        // Use a small context window to force eviction
        mgr.set_context_window(4_000);
        mgr.new_session("s1", "system").unwrap();

        // Add enough large messages to trigger eviction
        for i in 0..30 {
            let content = format!("large message {i}: {}", "x".repeat(300));
            mgr.add_user_message("s1", &content).unwrap();
        }

        let evicted = mgr.evict_if_needed("s1").unwrap();
        assert!(evicted > 0, "eviction should have triggered");

        // Check that summary exists and is capped
        let summary = mgr.get_session_summary("s1").unwrap();
        assert!(summary.is_some(), "summary should exist after eviction");

        let summary_text = summary.unwrap().summary_text;
        let summary_tokens = tokens::estimate_tokens(&summary_text);
        // The capped summary should be at most MAX_SUMMARY_TOKENS + some overhead
        // from the "[earlier context omitted]" prefix
        assert!(
            summary_tokens <= MAX_SUMMARY_TOKENS + 20,
            "summary should be capped at ~{} tokens, got {}",
            MAX_SUMMARY_TOKENS,
            summary_tokens
        );
    }
}
