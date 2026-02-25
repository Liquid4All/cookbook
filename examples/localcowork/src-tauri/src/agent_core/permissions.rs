//! Permission Store — tiered permission grants for tool execution.
//!
//! Supports three tiers:
//! - **Allow Once** (default Confirmed) — no grant stored, ask every time.
//! - **Allow for Session** — grant lives until `clear_session()` is called.
//! - **Always Allow** — persisted to platform data dir / `permissions.json`.
//!
//! The ToolRouter checks `PermissionStore::check()` before entering the
//! confirmation flow. If the tool has an active grant, confirmation is skipped.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// ─── Types ──────────────────────────────────────────────────────────────────

/// Scope of a permission grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionScope {
    /// Valid until the session ends.
    Session,
    /// Persisted across restarts.
    Always,
}

/// A single permission grant for a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionGrant {
    /// The fully-qualified tool name (e.g. "filesystem.write_file").
    pub tool_name: String,
    /// Whether this is a session or persistent grant.
    pub scope: PermissionScope,
    /// ISO 8601 timestamp when the grant was created.
    pub granted_at: String,
}

/// Result of checking a tool's permission status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionStatus {
    /// Tool has an active grant — skip confirmation.
    Allowed,
    /// No grant — proceed with normal confirmation flow.
    NeedsConfirmation,
}

// ─── Persistent Format ──────────────────────────────────────────────────────

/// On-disk format for `permissions.json`.
#[derive(Debug, Default, Serialize, Deserialize)]
struct PersistedGrants {
    /// Version for forward compatibility.
    version: u32,
    /// Tool name → grant mapping.
    grants: HashMap<String, PermissionGrant>,
}

// ─── PermissionStore ────────────────────────────────────────────────────────

/// Manages session and persistent permission grants.
pub struct PermissionStore {
    /// Grants that expire when the session ends.
    session_grants: HashMap<String, PermissionGrant>,
    /// Grants persisted to disk (loaded on startup, saved on mutation).
    persistent_grants: HashMap<String, PermissionGrant>,
    /// Path to the permissions JSON file.
    persist_path: PathBuf,
}

impl Default for PermissionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl PermissionStore {
    /// Create a new PermissionStore and load any persisted grants.
    pub fn new() -> Self {
        let persist_path = Self::default_persist_path();
        let mut store = Self {
            session_grants: HashMap::new(),
            persistent_grants: HashMap::new(),
            persist_path,
        };
        store.load_from_disk();
        store
    }

    /// Create a PermissionStore for testing (in-memory only, no disk I/O).
    #[cfg(test)]
    pub fn new_in_memory() -> Self {
        Self {
            session_grants: HashMap::new(),
            persistent_grants: HashMap::new(),
            persist_path: PathBuf::from("/dev/null"),
        }
    }

    /// Check if a tool has an active permission grant.
    pub fn check(&self, tool_name: &str) -> PermissionStatus {
        if self.persistent_grants.contains_key(tool_name)
            || self.session_grants.contains_key(tool_name)
        {
            PermissionStatus::Allowed
        } else {
            PermissionStatus::NeedsConfirmation
        }
    }

    /// Grant a permission for a tool.
    pub fn grant(&mut self, tool_name: &str, scope: PermissionScope) {
        let grant = PermissionGrant {
            tool_name: tool_name.to_string(),
            scope,
            granted_at: chrono::Utc::now().to_rfc3339(),
        };

        match scope {
            PermissionScope::Session => {
                self.session_grants.insert(tool_name.to_string(), grant);
            }
            PermissionScope::Always => {
                self.persistent_grants
                    .insert(tool_name.to_string(), grant);
                self.save_to_disk();
            }
        }

        tracing::info!(
            tool = tool_name,
            scope = ?scope,
            "permission granted"
        );
    }

    /// Revoke a persistent permission grant.
    pub fn revoke(&mut self, tool_name: &str) -> bool {
        let removed_persistent = self.persistent_grants.remove(tool_name).is_some();
        let removed_session = self.session_grants.remove(tool_name).is_some();

        if removed_persistent {
            self.save_to_disk();
        }

        let removed = removed_persistent || removed_session;
        if removed {
            tracing::info!(tool = tool_name, "permission revoked");
        }
        removed
    }

    /// List all persistent grants (for the Settings UI).
    pub fn list_persistent(&self) -> Vec<&PermissionGrant> {
        let mut grants: Vec<&PermissionGrant> = self.persistent_grants.values().collect();
        grants.sort_by(|a, b| a.tool_name.cmp(&b.tool_name));
        grants
    }

    /// Clear all session grants (called when a session ends).
    pub fn clear_session(&mut self) {
        let count = self.session_grants.len();
        self.session_grants.clear();
        if count > 0 {
            tracing::info!(cleared = count, "session permissions cleared");
        }
    }

    // ─── Persistence ────────────────────────────────────────────────────

    /// Default path: platform-standard data directory / `permissions.json`.
    fn default_persist_path() -> PathBuf {
        crate::data_dir().join("permissions.json")
    }

    /// Load persistent grants from disk.
    fn load_from_disk(&mut self) {
        if !self.persist_path.exists() {
            return;
        }

        match std::fs::read_to_string(&self.persist_path) {
            Ok(content) => match serde_json::from_str::<PersistedGrants>(&content) {
                Ok(persisted) => {
                    tracing::info!(
                        count = persisted.grants.len(),
                        path = %self.persist_path.display(),
                        "loaded persistent permissions"
                    );
                    self.persistent_grants = persisted.grants;
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        path = %self.persist_path.display(),
                        "failed to parse permissions file, starting fresh"
                    );
                }
            },
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %self.persist_path.display(),
                    "failed to read permissions file"
                );
            }
        }
    }

    /// Save persistent grants to disk (atomic write).
    fn save_to_disk(&self) {
        let persisted = PersistedGrants {
            version: 1,
            grants: self.persistent_grants.clone(),
        };

        let content = match serde_json::to_string_pretty(&persisted) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(error = %e, "failed to serialize permissions");
                return;
            }
        };

        // Ensure parent directory exists
        if let Some(parent) = self.persist_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        // Write to temp file, then rename for atomicity
        let tmp_path = self.persist_path.with_extension("json.tmp");
        if let Err(e) = std::fs::write(&tmp_path, &content) {
            tracing::error!(error = %e, "failed to write permissions temp file");
            return;
        }
        if let Err(e) = std::fs::rename(&tmp_path, &self.persist_path) {
            tracing::error!(error = %e, "failed to rename permissions file");
            return;
        }

        tracing::debug!(
            count = self.persistent_grants.len(),
            "saved persistent permissions"
        );
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_returns_needs_confirmation_by_default() {
        let store = PermissionStore::new_in_memory();
        assert_eq!(
            store.check("filesystem.write_file"),
            PermissionStatus::NeedsConfirmation
        );
    }

    #[test]
    fn test_session_grant_allows_tool() {
        let mut store = PermissionStore::new_in_memory();
        store.grant("filesystem.write_file", PermissionScope::Session);
        assert_eq!(
            store.check("filesystem.write_file"),
            PermissionStatus::Allowed
        );
    }

    #[test]
    fn test_always_grant_allows_tool() {
        let mut store = PermissionStore::new_in_memory();
        store.grant("filesystem.write_file", PermissionScope::Always);
        assert_eq!(
            store.check("filesystem.write_file"),
            PermissionStatus::Allowed
        );
    }

    #[test]
    fn test_clear_session_removes_session_grants_only() {
        let mut store = PermissionStore::new_in_memory();
        store.grant("tool_a", PermissionScope::Session);
        store.grant("tool_b", PermissionScope::Always);

        store.clear_session();

        assert_eq!(store.check("tool_a"), PermissionStatus::NeedsConfirmation);
        assert_eq!(store.check("tool_b"), PermissionStatus::Allowed);
    }

    #[test]
    fn test_revoke_removes_grant() {
        let mut store = PermissionStore::new_in_memory();
        store.grant("tool_a", PermissionScope::Always);
        assert_eq!(store.check("tool_a"), PermissionStatus::Allowed);

        let removed = store.revoke("tool_a");
        assert!(removed);
        assert_eq!(store.check("tool_a"), PermissionStatus::NeedsConfirmation);
    }

    #[test]
    fn test_revoke_nonexistent_returns_false() {
        let mut store = PermissionStore::new_in_memory();
        assert!(!store.revoke("nonexistent"));
    }

    #[test]
    fn test_list_persistent_returns_sorted() {
        let mut store = PermissionStore::new_in_memory();
        store.grant("zzz.tool", PermissionScope::Always);
        store.grant("aaa.tool", PermissionScope::Always);
        store.grant("mmm.tool", PermissionScope::Session); // should not appear

        let grants = store.list_persistent();
        assert_eq!(grants.len(), 2);
        assert_eq!(grants[0].tool_name, "aaa.tool");
        assert_eq!(grants[1].tool_name, "zzz.tool");
    }

    #[test]
    fn test_permission_scope_serialization() {
        let session = PermissionScope::Session;
        let json = serde_json::to_string(&session).unwrap();
        assert_eq!(json, "\"session\"");

        let always = PermissionScope::Always;
        let json = serde_json::to_string(&always).unwrap();
        assert_eq!(json, "\"always\"");
    }

    #[test]
    fn test_grant_overwrites_existing() {
        let mut store = PermissionStore::new_in_memory();
        store.grant("tool_a", PermissionScope::Session);
        store.grant("tool_a", PermissionScope::Always);

        // Should be in persistent, not session
        assert_eq!(store.list_persistent().len(), 1);
        assert_eq!(store.check("tool_a"), PermissionStatus::Allowed);

        // Clear session — should still be allowed via persistent
        store.clear_session();
        assert_eq!(store.check("tool_a"), PermissionStatus::Allowed);
    }
}
