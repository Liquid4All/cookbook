/**
 * SessionSidebar — collapsible panel for session management.
 *
 * Shows a list of past sessions with previews, lets the user
 * create new sessions, switch between them, and delete old ones.
 */

import { useEffect, useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";

import type { ChatMessage } from "../../types";

// ─── Types ─────────────────────────────────────────────────────────────────

interface SessionListItem {
  readonly id: string;
  readonly created_at: string;
  readonly last_activity: string;
  readonly message_count: number;
  readonly preview: string | null;
}

interface SessionSidebarProps {
  readonly activeSessionId: string | null;
  readonly onNewSession: () => Promise<void>;
  readonly onSwitchSession: (
    sessionId: string,
    messages: readonly ChatMessage[],
  ) => void;
  readonly isOpen: boolean;
  readonly onClose: () => void;
}

// ─── Helpers ───────────────────────────────────────────────────────────────

function formatRelativeTime(isoDate: string): string {
  const date = new Date(isoDate);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60_000);

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;

  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString();
}

// ─── Component ─────────────────────────────────────────────────────────────

export function SessionSidebar({
  activeSessionId,
  onNewSession,
  onSwitchSession,
  isOpen,
  onClose,
}: SessionSidebarProps): React.JSX.Element | null {
  const [sessions, setSessions] = useState<readonly SessionListItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSessions = useCallback(async (): Promise<void> => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await invoke<SessionListItem[]>("list_sessions");
      setSessions(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Fetch sessions when sidebar opens
  useEffect(() => {
    if (isOpen) {
      void fetchSessions();
    }
  }, [isOpen, fetchSessions]);

  const handleSwitch = useCallback(
    async (sessionId: string): Promise<void> => {
      if (sessionId === activeSessionId) return;
      try {
        const messages = await invoke<ChatMessage[]>("load_session", {
          sessionId,
        });
        onSwitchSession(sessionId, messages);
      } catch (e) {
        setError(`Failed to load session: ${String(e)}`);
      }
    },
    [activeSessionId, onSwitchSession],
  );

  const handleDelete = useCallback(
    async (sessionId: string): Promise<void> => {
      try {
        await invoke("delete_session", { sessionId });
        setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      } catch (e) {
        setError(`Failed to delete: ${String(e)}`);
      }
    },
    [],
  );

  const handleNew = useCallback(async (): Promise<void> => {
    await onNewSession();
    void fetchSessions();
  }, [onNewSession, fetchSessions]);

  if (!isOpen) return null;

  return (
    <div className="session-sidebar">
      {/* Header */}
      <div className="session-sidebar-header">
        <h3 className="session-sidebar-title">Sessions</h3>
        <button
          className="session-sidebar-close"
          onClick={onClose}
          title="Close sidebar"
        >
          &times;
        </button>
      </div>

      {/* New Session Button */}
      <button className="session-new-btn" onClick={() => void handleNew()}>
        + New Session
      </button>

      {/* Error */}
      {error && (
        <div className="session-sidebar-error">
          <span>{error}</span>
          <button onClick={() => setError(null)}>&times;</button>
        </div>
      )}

      {/* Session List */}
      <div className="session-list">
        {isLoading && (
          <div className="session-list-loading">Loading sessions…</div>
        )}

        {!isLoading && sessions.length === 0 && (
          <div className="session-list-empty">No sessions yet</div>
        )}

        {sessions.map((session) => (
          <div
            key={session.id}
            className={`session-card ${
              session.id === activeSessionId ? "session-card-active" : ""
            }`}
            onClick={() => void handleSwitch(session.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleSwitch(session.id);
            }}
          >
            <div className="session-card-header">
              <span className="session-card-time">
                {formatRelativeTime(session.last_activity)}
              </span>
              <span className="session-card-count">
                {session.message_count} msgs
              </span>
            </div>
            <div className="session-card-preview">
              {session.preview ?? "Empty session"}
            </div>
            {session.id !== activeSessionId && (
              <button
                className="session-card-delete"
                onClick={(e) => {
                  e.stopPropagation();
                  void handleDelete(session.id);
                }}
                title="Delete session"
              >
                🗑
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
