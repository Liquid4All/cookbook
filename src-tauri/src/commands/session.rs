//! Tauri IPC commands for session management.
//!
//! These commands let the frontend list, load, and delete conversation
//! sessions, as well as query the current context window budget.

use serde::Serialize;
use std::sync::Mutex;

use crate::agent_core::ConversationManager;
use crate::agent_core::tokens::truncate_utf8;

// ─── Response Types ─────────────────────────────────────────────────────────

/// Summary of a session for the sidebar list.
#[derive(Debug, Serialize)]
pub struct SessionListItem {
    pub id: String,
    pub created_at: String,
    pub last_activity: String,
    pub message_count: usize,
    pub preview: Option<String>,
}

/// Context budget snapshot sent to the frontend.
#[derive(Debug, Serialize)]
pub struct ContextBudgetResponse {
    pub total: u32,
    pub system_prompt: u32,
    pub tool_definitions: u32,
    pub conversation_history: u32,
    pub output_reservation: u32,
    pub remaining: u32,
}

// ─── Commands ───────────────────────────────────────────────────────────────

/// List sessions that have actual user content (not just system prompt).
///
/// Excludes empty sessions to keep the sidebar clean. Sessions are
/// sorted by most recent activity first.
#[tauri::command]
pub fn list_sessions(
    state: tauri::State<'_, Mutex<ConversationManager>>,
) -> Result<Vec<SessionListItem>, String> {
    let mgr = state.lock().map_err(|e| format!("Lock error: {e}"))?;
    let sessions = mgr.db().list_sessions().map_err(|e| format!("{e}"))?;

    let mut items = Vec::new();
    for session in sessions {
        let count = mgr.db().message_count(&session.id).unwrap_or(0);

        // Skip sessions with only a system prompt (no user interaction)
        if count <= 1 {
            continue;
        }

        // Get first user message as preview
        let preview = mgr
            .get_history(&session.id)
            .ok()
            .and_then(|msgs| {
                msgs.iter()
                    .find(|m| m.role == crate::inference::types::Role::User)
                    .and_then(|m| m.content.clone())
            })
            .map(|s| {
                if s.len() > 80 {
                    format!("{}…", truncate_utf8(&s, 77))
                } else {
                    s
                }
            });

        items.push(SessionListItem {
            id: session.id,
            created_at: session.created_at,
            last_activity: session.last_activity,
            message_count: count,
            preview,
        });
    }

    Ok(items)
}

/// Load a session's conversation history for display.
///
/// Returns messages with full metadata including toolCallId and toolCalls
/// so the frontend can properly render ToolTrace components.
#[tauri::command]
pub fn load_session(
    session_id: String,
    state: tauri::State<'_, Mutex<ConversationManager>>,
) -> Result<Vec<serde_json::Value>, String> {
    let mgr = state.lock().map_err(|e| format!("Lock error: {e}"))?;
    let history = mgr.get_history(&session_id).map_err(|e| format!("{e}"))?;

    let messages: Vec<serde_json::Value> = history
        .iter()
        .filter(|m| m.role != crate::inference::types::Role::System)
        .map(|m| {
            let mut msg = serde_json::json!({
                "id": m.id,
                "sessionId": m.session_id,
                "timestamp": m.timestamp,
                "role": format!("{:?}", m.role).to_lowercase(),
                "content": m.content,
                "tokenCount": m.token_count,
            });

            // Include tool_call_id and toolResult for tool result messages
            // so ToolTrace can correlate them and show results.
            if let Some(ref tc_id) = m.tool_call_id {
                let obj = msg.as_object_mut().unwrap();
                obj.insert(
                    "toolCallId".to_string(),
                    serde_json::Value::String(tc_id.clone()),
                );
                // Include toolResult so ToolTrace can show result status
                obj.insert(
                    "toolResult".to_string(),
                    serde_json::json!({
                        "success": true,
                        "result": m.content,
                        "toolCallId": tc_id,
                    }),
                );
            }

            // Include tool_calls for assistant messages
            if let Some(ref calls) = m.tool_calls {
                let tc_json: Vec<serde_json::Value> = calls
                    .iter()
                    .map(|tc| {
                        serde_json::json!({
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        })
                    })
                    .collect();
                msg.as_object_mut()
                    .unwrap()
                    .insert("toolCalls".to_string(), serde_json::Value::Array(tc_json));
            }

            msg
        })
        .collect();

    Ok(messages)
}

/// Delete a session and all its data.
#[tauri::command]
pub fn delete_session(
    session_id: String,
    state: tauri::State<'_, Mutex<ConversationManager>>,
) -> Result<(), String> {
    let mgr = state.lock().map_err(|e| format!("Lock error: {e}"))?;
    mgr.db()
        .delete_session(&session_id)
        .map_err(|e| format!("{e}"))
}

/// Get the current context window budget for a session.
#[tauri::command]
pub fn get_context_budget(
    session_id: String,
    state: tauri::State<'_, Mutex<ConversationManager>>,
) -> Result<ContextBudgetResponse, String> {
    let mgr = state.lock().map_err(|e| format!("Lock error: {e}"))?;
    let budget = mgr.get_budget(&session_id).map_err(|e| format!("{e}"))?;

    Ok(ContextBudgetResponse {
        total: budget.total,
        system_prompt: budget.system_prompt,
        tool_definitions: budget.tool_definitions,
        conversation_history: budget.conversation_history,
        output_reservation: budget.output_reservation,
        remaining: budget.remaining,
    })
}

/// Clean up orphan empty sessions (only system prompt, no user messages).
///
/// Called on app startup to remove sessions from previous launches that
/// were created but never used.
#[tauri::command]
pub fn cleanup_empty_sessions(
    state: tauri::State<'_, Mutex<ConversationManager>>,
) -> Result<u32, String> {
    let mgr = state.lock().map_err(|e| format!("Lock error: {e}"))?;
    let sessions = mgr.db().list_sessions().map_err(|e| format!("{e}"))?;

    let mut cleaned = 0u32;
    for session in &sessions {
        if let Ok(count) = mgr.db().message_count(&session.id) {
            if count <= 1 && mgr.db().delete_session(&session.id).is_ok() {
                cleaned += 1;
            }
        }
    }

    if cleaned > 0 {
        tracing::info!(cleaned = cleaned, "cleaned up empty sessions");
    }
    Ok(cleaned)
}
