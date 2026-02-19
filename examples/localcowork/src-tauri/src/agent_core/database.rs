//! SQLite database for conversation history, sessions, and undo stack.
//!
//! Uses `rusqlite` in synchronous mode (Tauri commands run on a thread pool).
//! WAL mode is enabled for concurrent reads during streaming.

use rusqlite::{params, Connection, OptionalExtension};

use super::errors::AgentError;
use super::types::{
    AuditEntry, AuditStatus, ConversationMessage, NewMessage, NewUndoEntry, Session,
    SessionSummary, UndoEntry,
};
use crate::inference::types::{Role, ToolCall};

// ─── Database ───────────────────────────────────────────────────────────────

/// SQLite database handle for the agent core.
pub struct AgentDatabase {
    conn: Connection,
}

impl AgentDatabase {
    /// Open (or create) the agent database at the given path.
    ///
    /// Pass `":memory:"` for an in-memory database (tests).
    pub fn open(path: &str) -> Result<Self, AgentError> {
        let conn = Connection::open(path)?;

        // Enable WAL mode for concurrent reads
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;

        let db = Self { conn };
        db.create_tables()?;
        Ok(db)
    }

    /// Create all required tables if they don't exist.
    fn create_tables(&self) -> Result<(), AgentError> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_activity TEXT NOT NULL DEFAULT (datetime('now')),
                summary TEXT,
                files_touched TEXT DEFAULT '[]',
                decisions_made TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                tool_result TEXT,
                token_count INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON conversation_messages(session_id, id);

            CREATE TABLE IF NOT EXISTS undo_stack (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                tool_name TEXT NOT NULL,
                action_type TEXT NOT NULL,
                original_state TEXT NOT NULL,
                new_state TEXT NOT NULL,
                undone INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_undo_session
                ON undo_stack(session_id, undone);

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                tool_name TEXT NOT NULL,
                arguments TEXT,
                result TEXT,
                result_status TEXT NOT NULL,
                user_confirmed INTEGER NOT NULL DEFAULT 0,
                execution_time_ms INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_audit_session
                ON audit_log(session_id);
            ",
        )?;
        Ok(())
    }

    // ─── Sessions ───────────────────────────────────────────────────────

    /// Create a new session with the given ID.
    pub fn create_session(&self, session_id: &str) -> Result<(), AgentError> {
        self.conn.execute(
            "INSERT INTO sessions (id) VALUES (?1)",
            params![session_id],
        )?;
        Ok(())
    }

    /// Get a session by ID.
    pub fn get_session(&self, session_id: &str) -> Result<Option<Session>, AgentError> {
        let result = self
            .conn
            .query_row(
                "SELECT id, created_at, last_activity, summary, files_touched, decisions_made
                 FROM sessions WHERE id = ?1",
                params![session_id],
                |row| {
                    Ok(Session {
                        id: row.get(0)?,
                        created_at: row.get(1)?,
                        last_activity: row.get(2)?,
                        summary: row.get(3)?,
                        files_touched: parse_json_array(row.get::<_, String>(4)?),
                        decisions_made: parse_json_array(row.get::<_, String>(5)?),
                    })
                },
            )
            .optional()?;
        Ok(result)
    }

    /// Update the session's last activity timestamp.
    pub fn touch_session(&self, session_id: &str) -> Result<(), AgentError> {
        self.conn.execute(
            "UPDATE sessions SET last_activity = datetime('now') WHERE id = ?1",
            params![session_id],
        )?;
        Ok(())
    }

    /// Update the session summary.
    pub fn update_session_summary(
        &self,
        session_id: &str,
        summary: &str,
        files: &[String],
        decisions: &[String],
    ) -> Result<(), AgentError> {
        let files_json = serde_json::to_string(files)?;
        let decisions_json = serde_json::to_string(decisions)?;
        self.conn.execute(
            "UPDATE sessions SET summary = ?2, files_touched = ?3, decisions_made = ?4
             WHERE id = ?1",
            params![session_id, summary, files_json, decisions_json],
        )?;
        Ok(())
    }

    /// Get the session summary for context window inclusion.
    pub fn get_session_summary(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionSummary>, AgentError> {
        let session = self.get_session(session_id)?;
        match session {
            Some(s) if s.summary.is_some() => Ok(Some(SessionSummary {
                session_id: s.id,
                summary_text: s.summary.unwrap_or_default(),
                files_touched: s.files_touched,
                decisions_made: s.decisions_made,
            })),
            _ => Ok(None),
        }
    }

    /// List all sessions, ordered by most recent activity first.
    pub fn list_sessions(&self) -> Result<Vec<Session>, AgentError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, created_at, last_activity, summary, files_touched, decisions_made
             FROM sessions
             ORDER BY last_activity DESC",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(Session {
                id: row.get(0)?,
                created_at: row.get(1)?,
                last_activity: row.get(2)?,
                summary: row.get(3)?,
                files_touched: parse_json_array(row.get::<_, String>(4)?),
                decisions_made: parse_json_array(row.get::<_, String>(5)?),
            })
        })?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row?);
        }
        Ok(sessions)
    }

    /// Delete a session and all its associated messages, undo entries, and audit log.
    pub fn delete_session(&self, session_id: &str) -> Result<(), AgentError> {
        self.conn.execute(
            "DELETE FROM audit_log WHERE session_id = ?1",
            params![session_id],
        )?;
        self.conn.execute(
            "DELETE FROM undo_stack WHERE session_id = ?1",
            params![session_id],
        )?;
        self.conn.execute(
            "DELETE FROM conversation_messages WHERE session_id = ?1",
            params![session_id],
        )?;
        self.conn.execute(
            "DELETE FROM sessions WHERE id = ?1",
            params![session_id],
        )?;
        Ok(())
    }

    // ─── Messages ───────────────────────────────────────────────────────

    /// Insert a new message into the conversation history.
    pub fn insert_message(
        &self,
        session_id: &str,
        msg: &NewMessage,
        token_count: u32,
    ) -> Result<i64, AgentError> {
        let role_str = role_to_str(&msg.role);
        let tool_calls_json = msg
            .tool_calls
            .as_ref()
            .map(|tc| serde_json::to_string(tc).unwrap_or_default());
        let tool_result_json = msg
            .tool_result
            .as_ref()
            .map(|r| serde_json::to_string(r).unwrap_or_default());

        self.conn.execute(
            "INSERT INTO conversation_messages
             (session_id, role, content, tool_calls, tool_call_id, tool_result, token_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id,
                role_str,
                msg.content,
                tool_calls_json,
                msg.tool_call_id,
                tool_result_json,
                token_count,
            ],
        )?;

        self.touch_session(session_id)?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get all messages for a session, ordered by ID (chronological).
    pub fn get_messages(
        &self,
        session_id: &str,
    ) -> Result<Vec<ConversationMessage>, AgentError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, timestamp, role, content,
                    tool_calls, tool_call_id, tool_result, token_count
             FROM conversation_messages
             WHERE session_id = ?1
             ORDER BY id ASC",
        )?;

        let rows = stmt.query_map(params![session_id], |row| {
            Ok(row_to_message(row))
        })?;

        let mut messages = Vec::new();
        for row in rows {
            messages.push(row?);
        }
        Ok(messages)
    }

    /// Get the N most recent messages for a session.
    pub fn get_recent_messages(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<ConversationMessage>, AgentError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, timestamp, role, content,
                    tool_calls, tool_call_id, tool_result, token_count
             FROM conversation_messages
             WHERE session_id = ?1
             ORDER BY id DESC
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![session_id, limit as i64], |row| {
            Ok(row_to_message(row))
        })?;

        let mut messages = Vec::new();
        for row in rows {
            messages.push(row?);
        }
        // Reverse so oldest is first
        messages.reverse();
        Ok(messages)
    }

    /// Get the total token count for all messages in a session.
    pub fn total_message_tokens(&self, session_id: &str) -> Result<u32, AgentError> {
        let total: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(token_count), 0)
             FROM conversation_messages WHERE session_id = ?1",
            params![session_id],
            |row| row.get(0),
        )?;
        Ok(total as u32)
    }

    /// Count messages in a session.
    pub fn message_count(&self, session_id: &str) -> Result<usize, AgentError> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM conversation_messages WHERE session_id = ?1",
            params![session_id],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Delete the oldest N messages from a session (for eviction).
    pub fn delete_oldest_messages(
        &self,
        session_id: &str,
        count: usize,
    ) -> Result<Vec<ConversationMessage>, AgentError> {
        // First, fetch the messages we're about to delete
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, timestamp, role, content,
                    tool_calls, tool_call_id, tool_result, token_count
             FROM conversation_messages
             WHERE session_id = ?1 AND role != 'system'
             ORDER BY id ASC
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![session_id, count as i64], |row| {
            Ok(row_to_message(row))
        })?;

        let mut evicted = Vec::new();
        for row in rows {
            evicted.push(row?);
        }

        // Delete them
        if !evicted.is_empty() {
            let ids: Vec<i64> = evicted.iter().map(|m| m.id).collect();
            let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
            let sql = format!(
                "DELETE FROM conversation_messages WHERE id IN ({})",
                placeholders.join(",")
            );
            let params: Vec<Box<dyn rusqlite::types::ToSql>> =
                ids.iter().map(|id| Box::new(*id) as Box<dyn rusqlite::types::ToSql>).collect();
            self.conn.execute(
                &sql,
                rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
            )?;
        }

        Ok(evicted)
    }

    // ─── Undo Stack ─────────────────────────────────────────────────────

    /// Push a new undo entry.
    pub fn push_undo_entry(
        &self,
        session_id: &str,
        entry: &NewUndoEntry,
    ) -> Result<i64, AgentError> {
        let original = serde_json::to_string(&entry.original_state)?;
        let new_state = serde_json::to_string(&entry.new_state)?;

        self.conn.execute(
            "INSERT INTO undo_stack
             (session_id, tool_name, action_type, original_state, new_state)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                session_id,
                entry.tool_name,
                entry.action_type,
                original,
                new_state,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get all non-undone entries in the undo stack for a session.
    pub fn get_undo_stack(&self, session_id: &str) -> Result<Vec<UndoEntry>, AgentError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, timestamp, tool_name, action_type,
                    original_state, new_state, undone
             FROM undo_stack
             WHERE session_id = ?1 AND undone = 0
             ORDER BY id DESC",
        )?;

        let rows = stmt.query_map(params![session_id], |row| {
            Ok(UndoEntry {
                id: row.get(0)?,
                session_id: row.get(1)?,
                timestamp: row.get(2)?,
                tool_name: row.get(3)?,
                action_type: row.get(4)?,
                original_state: parse_json_value(row.get::<_, String>(5)?),
                new_state: parse_json_value(row.get::<_, String>(6)?),
                undone: row.get::<_, i32>(7)? != 0,
            })
        })?;

        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
        }
        Ok(entries)
    }

    /// Mark an undo entry as undone.
    pub fn mark_undone(&self, undo_id: i64) -> Result<(), AgentError> {
        let updated = self.conn.execute(
            "UPDATE undo_stack SET undone = 1 WHERE id = ?1",
            params![undo_id],
        )?;
        if updated == 0 {
            return Err(AgentError::UndoFailed {
                undo_id,
                reason: "undo entry not found".to_string(),
            });
        }
        Ok(())
    }

    // ─── Audit Log ──────────────────────────────────────────────────────

    /// Insert an audit log entry.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_audit_entry(
        &self,
        session_id: &str,
        tool_name: &str,
        arguments: &serde_json::Value,
        result: Option<&serde_json::Value>,
        status: AuditStatus,
        user_confirmed: bool,
        execution_time_ms: u64,
    ) -> Result<i64, AgentError> {
        let args_json = serde_json::to_string(arguments)?;
        let result_json = result.map(|r| serde_json::to_string(r).unwrap_or_default());

        self.conn.execute(
            "INSERT INTO audit_log
             (session_id, tool_name, arguments, result, result_status,
              user_confirmed, execution_time_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id,
                tool_name,
                args_json,
                result_json,
                status.as_str(),
                user_confirmed as i32,
                execution_time_ms as i64,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get audit log entries for a session.
    pub fn get_audit_entries(
        &self,
        session_id: &str,
    ) -> Result<Vec<AuditEntry>, AgentError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, timestamp, tool_name, arguments,
                    result, result_status, user_confirmed, execution_time_ms
             FROM audit_log
             WHERE session_id = ?1
             ORDER BY id ASC",
        )?;

        let rows = stmt.query_map(params![session_id], |row| {
            Ok(AuditEntry {
                id: row.get(0)?,
                session_id: row.get(1)?,
                timestamp: row.get(2)?,
                tool_name: row.get(3)?,
                arguments: parse_json_value(row.get::<_, String>(4)?),
                result: row
                    .get::<_, Option<String>>(5)?
                    .map(parse_json_value),
                result_status: AuditStatus::parse(
                    &row.get::<_, String>(6)?,
                ),
                user_confirmed: row.get::<_, i32>(7)? != 0,
                execution_time_ms: row.get::<_, i64>(8)? as u64,
            })
        })?;

        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
        }
        Ok(entries)
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Convert a rusqlite row to a ConversationMessage.
fn row_to_message(row: &rusqlite::Row<'_>) -> ConversationMessage {
    ConversationMessage {
        id: row.get(0).unwrap_or(0),
        session_id: row.get(1).unwrap_or_default(),
        timestamp: row.get(2).unwrap_or_default(),
        role: str_to_role(&row.get::<_, String>(3).unwrap_or_default()),
        content: row.get(4).unwrap_or(None),
        tool_calls: row
            .get::<_, Option<String>>(5)
            .unwrap_or(None)
            .and_then(|s| serde_json::from_str::<Vec<ToolCall>>(&s).ok()),
        tool_call_id: row.get(6).unwrap_or(None),
        tool_result: row
            .get::<_, Option<String>>(7)
            .unwrap_or(None)
            .map(parse_json_value),
        token_count: row.get::<_, i32>(8).unwrap_or(0) as u32,
    }
}

/// Parse a JSON string into a Vec<String>, defaulting to empty.
fn parse_json_array(json: String) -> Vec<String> {
    serde_json::from_str(&json).unwrap_or_default()
}

/// Parse a JSON string into a serde_json::Value, defaulting to null.
fn parse_json_value(json: String) -> serde_json::Value {
    serde_json::from_str(&json).unwrap_or(serde_json::Value::Null)
}

/// Convert a Role to its string representation.
fn role_to_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Parse a string into a Role.
fn str_to_role(s: &str) -> Role {
    match s {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::Role;

    fn test_db() -> AgentDatabase {
        AgentDatabase::open(":memory:").unwrap()
    }

    #[test]
    fn test_create_and_get_session() {
        let db = test_db();
        db.create_session("test-session-1").unwrap();

        let session = db.get_session("test-session-1").unwrap();
        assert!(session.is_some());
        let s = session.unwrap();
        assert_eq!(s.id, "test-session-1");
        assert!(s.summary.is_none());
    }

    #[test]
    fn test_session_not_found() {
        let db = test_db();
        let session = db.get_session("nonexistent").unwrap();
        assert!(session.is_none());
    }

    #[test]
    fn test_insert_and_get_messages() {
        let db = test_db();
        db.create_session("s1").unwrap();

        let msg1 = NewMessage {
            role: Role::User,
            content: Some("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
            tool_result: None,
        };
        let msg2 = NewMessage {
            role: Role::Assistant,
            content: Some("hi there".to_string()),
            tool_calls: None,
            tool_call_id: None,
            tool_result: None,
        };

        db.insert_message("s1", &msg1, 10).unwrap();
        db.insert_message("s1", &msg2, 15).unwrap();

        let messages = db.get_messages("s1").unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(messages[0].token_count, 10);
    }

    #[test]
    fn test_get_recent_messages() {
        let db = test_db();
        db.create_session("s1").unwrap();

        for i in 0..10 {
            let msg = NewMessage {
                role: Role::User,
                content: Some(format!("message {i}")),
                tool_calls: None,
                tool_call_id: None,
                tool_result: None,
            };
            db.insert_message("s1", &msg, 5).unwrap();
        }

        let recent = db.get_recent_messages("s1", 3).unwrap();
        assert_eq!(recent.len(), 3);
        assert!(recent[0].content.as_ref().unwrap().contains("message 7"));
        assert!(recent[2].content.as_ref().unwrap().contains("message 9"));
    }

    #[test]
    fn test_total_message_tokens() {
        let db = test_db();
        db.create_session("s1").unwrap();

        let msg = NewMessage {
            role: Role::User,
            content: Some("test".to_string()),
            tool_calls: None,
            tool_call_id: None,
            tool_result: None,
        };
        db.insert_message("s1", &msg, 10).unwrap();
        db.insert_message("s1", &msg, 20).unwrap();

        assert_eq!(db.total_message_tokens("s1").unwrap(), 30);
    }

    #[test]
    fn test_delete_oldest_messages() {
        let db = test_db();
        db.create_session("s1").unwrap();

        // Insert system + 5 user messages
        let sys = NewMessage {
            role: Role::System,
            content: Some("system prompt".to_string()),
            tool_calls: None,
            tool_call_id: None,
            tool_result: None,
        };
        db.insert_message("s1", &sys, 50).unwrap();

        for i in 0..5 {
            let msg = NewMessage {
                role: Role::User,
                content: Some(format!("msg {i}")),
                tool_calls: None,
                tool_call_id: None,
                tool_result: None,
            };
            db.insert_message("s1", &msg, 10).unwrap();
        }

        // Delete 2 oldest non-system messages
        let evicted = db.delete_oldest_messages("s1", 2).unwrap();
        assert_eq!(evicted.len(), 2);
        assert!(evicted[0].content.as_ref().unwrap().contains("msg 0"));
        assert!(evicted[1].content.as_ref().unwrap().contains("msg 1"));

        // 4 remain (1 system + 3 user)
        assert_eq!(db.message_count("s1").unwrap(), 4);
    }

    #[test]
    fn test_undo_stack() {
        let db = test_db();
        db.create_session("s1").unwrap();

        let entry = NewUndoEntry {
            tool_name: "filesystem.move_file".to_string(),
            action_type: "move".to_string(),
            original_state: serde_json::json!({"path": "/old"}),
            new_state: serde_json::json!({"path": "/new"}),
        };
        let id = db.push_undo_entry("s1", &entry).unwrap();

        let stack = db.get_undo_stack("s1").unwrap();
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0].tool_name, "filesystem.move_file");
        assert!(!stack[0].undone);

        // Mark as undone
        db.mark_undone(id).unwrap();
        let stack = db.get_undo_stack("s1").unwrap();
        assert_eq!(stack.len(), 0); // Filtered out
    }

    #[test]
    fn test_audit_log() {
        let db = test_db();
        db.create_session("s1").unwrap();

        let args = serde_json::json!({"path": "/tmp"});
        let result = serde_json::json!({"files": ["a.txt"]});

        db.insert_audit_entry(
            "s1",
            "filesystem.list_dir",
            &args,
            Some(&result),
            AuditStatus::Success,
            false,
            42,
        )
        .unwrap();

        let entries = db.get_audit_entries("s1").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].tool_name, "filesystem.list_dir");
        assert_eq!(entries[0].result_status, AuditStatus::Success);
        assert!(!entries[0].user_confirmed);
        assert_eq!(entries[0].execution_time_ms, 42);
    }

    #[test]
    fn test_update_session_summary() {
        let db = test_db();
        db.create_session("s1").unwrap();

        db.update_session_summary(
            "s1",
            "User asked to organize files in /tmp",
            &["file.txt".to_string()],
            &["moved files to /archive".to_string()],
        )
        .unwrap();

        let summary = db.get_session_summary("s1").unwrap();
        assert!(summary.is_some());
        let s = summary.unwrap();
        assert!(s.summary_text.contains("organize files"));
        assert_eq!(s.files_touched, vec!["file.txt"]);
    }
}
