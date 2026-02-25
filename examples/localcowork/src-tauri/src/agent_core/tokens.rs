//! Token estimation for context window management.
//!
//! Uses character-based heuristics calibrated for LLM tokenizers:
//! - English prose: ~3.2 chars/token (conservative — overestimate is safer)
//! - JSON/structured content: ~2.8 chars/token (denser due to punctuation, short keys)
//!
//! A more accurate tokenizer (tiktoken-rs) can replace this when the model
//! is finalized.

use crate::inference::types::{ChatMessage, Role};

// ─── Constants ──────────────────────────────────────────────────────────────

/// Average characters per token for English prose.
///
/// Calibrated conservatively — most LLM tokenizers produce ~3.5-4.0 chars/token
/// for English text. We use 3.2 to err on the side of overestimation, which is
/// safer than underestimating and overflowing the context window.
const CHARS_PER_TOKEN: f64 = 3.2;

/// Average characters per token for JSON/structured content.
///
/// JSON tokenizes more densely than prose due to punctuation, short keys,
/// braces, and colons. Tool call arguments, tool results, and schema
/// definitions all fall into this category.
const JSON_CHARS_PER_TOKEN: f64 = 2.8;

/// Per-message overhead (role label, formatting tokens).
const MESSAGE_OVERHEAD_TOKENS: u32 = 4;

/// Overhead for tool call JSON structure (per call).
const TOOL_CALL_OVERHEAD_TOKENS: u32 = 10;

// ─── UTF-8 Safe Truncation ──────────────────────────────────────────────────

/// Truncate a string to at most `max_bytes` bytes on a valid UTF-8 char boundary.
///
/// Returns a `&str` that is always valid UTF-8 and at most `max_bytes` long.
/// If the byte at `max_bytes` is inside a multi-byte character, the slice is
/// shortened to the preceding character boundary.
pub(crate) fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    // Walk backward to find a valid char boundary
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Estimate the token count for a string of natural language text.
pub fn estimate_tokens(text: &str) -> u32 {
    let chars = text.len() as f64;
    (chars / CHARS_PER_TOKEN).ceil() as u32
}

/// Estimate the token count for JSON/structured content.
///
/// JSON tokenizes more densely than prose, so this uses a tighter ratio.
/// Use this for tool call arguments, tool results, and schema definitions.
pub fn estimate_json_tokens(json_text: &str) -> u32 {
    let chars = json_text.len() as f64;
    (chars / JSON_CHARS_PER_TOKEN).ceil() as u32
}

/// Estimate the token count for a `ChatMessage`.
///
/// Accounts for content, tool calls, and per-message overhead.
/// Uses the JSON-specific estimator for tool call arguments (which are
/// always JSON) and the prose estimator for natural language content.
pub fn estimate_message_tokens(message: &ChatMessage) -> u32 {
    let mut total = MESSAGE_OVERHEAD_TOKENS;

    // Content tokens — use prose estimator for user/assistant text,
    // JSON estimator for tool results (role == Tool)
    if let Some(ref content) = message.content {
        total += match message.role {
            Role::Tool => estimate_json_tokens(content),
            _ => estimate_tokens(content),
        };
    }

    // Tool call tokens — arguments are always JSON
    if let Some(ref calls) = message.tool_calls {
        for call in calls {
            total += TOOL_CALL_OVERHEAD_TOKENS;
            total += estimate_tokens(&call.function.name);
            total += estimate_json_tokens(&call.function.arguments);
        }
    }

    // Tool call ID (for tool-role messages)
    if let Some(ref id) = message.tool_call_id {
        total += estimate_tokens(id);
    }

    total
}

/// Estimate the token count for a system prompt string.
pub fn estimate_system_prompt_tokens(prompt: &str) -> u32 {
    MESSAGE_OVERHEAD_TOKENS + estimate_tokens(prompt)
}

/// Estimate token count for tool definitions in OpenAI format.
///
/// Tool definitions are JSON, so we use the JSON-specific estimator.
pub fn estimate_tool_definitions_tokens(tools: &[serde_json::Value]) -> u32 {
    let json = serde_json::to_string(tools).unwrap_or_default();
    estimate_json_tokens(&json)
}

/// Estimate token count for a raw string that will be included as content.
pub fn estimate_content_tokens(content: &str) -> u32 {
    estimate_tokens(content)
}

/// Summarize a tool result into a one-line string.
///
/// Used when evicting old conversation turns to reduce token usage.
pub fn summarize_tool_result(tool_name: &str, result: &serde_json::Value) -> String {
    let result_str = serde_json::to_string(result).unwrap_or_default();
    let token_count = estimate_tokens(&result_str);

    if token_count <= 50 {
        // Short enough to keep as-is
        format!("[{tool_name} returned: {result_str}]")
    } else {
        // Summarize to one line
        let preview = truncate_utf8(&result_str, 100);
        format!(
            "[{tool_name} returned ~{token_count} tokens: {preview}...]"
        )
    }
}

/// Build a one-line summary of a conversation turn for eviction.
///
/// Captures the user's request and the assistant's response type.
pub fn summarize_turn(role: &Role, content: Option<&str>) -> String {
    match role {
        Role::User => {
            let text = content.unwrap_or("[empty]");
            let preview = truncate_utf8(text, 80);
            format!("User: {preview}")
        }
        Role::Assistant => {
            let text = content.unwrap_or("[tool calls]");
            let preview = truncate_utf8(text, 80);
            format!("Assistant: {preview}")
        }
        Role::Tool => {
            let text = content.unwrap_or("[result]");
            let preview = truncate_utf8(text, 60);
            format!("Tool result: {preview}")
        }
        Role::System => "System prompt".to_string(),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::{FunctionCallResponse, ToolCallResponse};

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_short() {
        // "hello" = 5 chars → ceil(5/3.2) = 2
        assert_eq!(estimate_tokens("hello"), 2);
    }

    #[test]
    fn test_estimate_tokens_longer() {
        // 100 chars → ceil(100/3.2) = 32
        let text = "a".repeat(100);
        assert_eq!(estimate_tokens(&text), 32);
    }

    #[test]
    fn test_estimate_json_tokens() {
        // 16 chars → ceil(16/2.8) = 6
        let json = r#"{"path": "/tmp"}"#;
        assert_eq!(estimate_json_tokens(json), 6);
    }

    #[test]
    fn test_estimate_message_tokens_content_only() {
        let msg = ChatMessage {
            role: Role::User,
            content: Some("Hello, world!".to_string()), // 13 chars → ceil(13/3.2) = 5
            tool_call_id: None,
            tool_calls: None,
        };
        let tokens = estimate_message_tokens(&msg);
        // 4 overhead + 5 content = 9
        assert_eq!(tokens, 9);
    }

    #[test]
    fn test_estimate_message_tokens_with_tool_calls() {
        let msg = ChatMessage {
            role: Role::Assistant,
            content: None,
            tool_call_id: None,
            tool_calls: Some(vec![ToolCallResponse {
                id: "call_1".to_string(),
                r#type: "function".to_string(),
                function: FunctionCallResponse {
                    name: "filesystem.list_dir".to_string(),
                    arguments: r#"{"path": "/tmp"}"#.to_string(),
                },
            }]),
        };
        let tokens = estimate_message_tokens(&msg);
        // 4 overhead + 10 tool_call_overhead + name_tokens + args_tokens > 4
        assert!(tokens > 4);
    }

    #[test]
    fn test_summarize_tool_result_short() {
        let result = serde_json::json!({"files": ["a.txt"]});
        let summary = summarize_tool_result("filesystem.list_dir", &result);
        assert!(summary.starts_with("[filesystem.list_dir returned:"));
    }

    #[test]
    fn test_summarize_tool_result_long() {
        let long_data: Vec<String> = (0..200).map(|i| format!("file_{i}.txt")).collect();
        let result = serde_json::json!({"files": long_data});
        let summary = summarize_tool_result("filesystem.list_dir", &result);
        assert!(summary.contains("tokens:"));
        assert!(summary.ends_with("...]"));
    }

    #[test]
    fn test_summarize_turn_user() {
        let summary = summarize_turn(&Role::User, Some("List all files in /tmp"));
        assert!(summary.starts_with("User: List all"));
    }

    #[test]
    fn test_summarize_turn_assistant_none() {
        let summary = summarize_turn(&Role::Assistant, None);
        assert_eq!(summary, "Assistant: [tool calls]");
    }

    #[test]
    fn test_truncate_utf8_ascii() {
        assert_eq!(truncate_utf8("hello world", 5), "hello");
    }

    #[test]
    fn test_truncate_utf8_within_multibyte() {
        // '═' is U+2550, encoded as 3 bytes: 0xE2, 0x95, 0x90
        let text = "═══"; // 9 bytes total
        // Cutting at byte 4 lands inside the second '═' (bytes 3..6)
        assert_eq!(truncate_utf8(text, 4), "═");
        // Cutting at byte 6 is exactly at a boundary
        assert_eq!(truncate_utf8(text, 6), "══");
    }

    #[test]
    fn test_truncate_utf8_no_truncation_needed() {
        assert_eq!(truncate_utf8("short", 100), "short");
    }

    #[test]
    fn test_summarize_tool_result_unicode_no_panic() {
        // Reproduces the crash: audit report with box-drawing chars
        let report = format!(
            "{}{}",
            "═".repeat(50),
            "AUDIT REPORT — Session test-session"
        );
        let result = serde_json::json!({"report": report});
        // This must NOT panic
        let summary = summarize_tool_result("audit.generate_audit_report", &result);
        assert!(summary.contains("audit.generate_audit_report"));
    }
}
