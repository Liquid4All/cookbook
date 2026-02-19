/**
 * ChatPanel — the main chat interface.
 *
 * Composes SessionSidebar, ContextIndicator, MessageList, MessageInput,
 * and ConfirmationDialog into a single panel. Manages session lifecycle
 * and event listeners.
 */

import { useCallback, useEffect } from "react";

import { useChatStore } from "../../stores/chatStore";
import { ConfirmationDialog } from "./ConfirmationDialog";
import { ContextIndicator } from "./ContextIndicator";
import { MessageInput } from "./MessageInput";
import { MessageList } from "./MessageList";
import { SessionSidebar } from "./SessionSidebar";

export function ChatPanel(): React.JSX.Element {
  const {
    messages,
    isGenerating,
    streamingContent,
    pendingConfirmation,
    contextBudget,
    error,
    isInitializing,
    sessionId,
    isSidebarOpen,
    startSession,
    sendMessage,
    respondToConfirmation,
    switchSession,
    toggleSidebar,
    clearError,
    setupListeners,
  } = useChatStore();

  // Start session and set up event listeners on mount.
  // setupListeners() is guarded against React.StrictMode double-mount —
  // it returns a cleanup function that only tears down on the real unmount.
  useEffect(() => {
    let cleanup: (() => void) | null = null;

    const init = async (): Promise<void> => {
      cleanup = await setupListeners();
      // On mount: resume the most recent session (forceNew = false)
      await startSession(false);
    };

    void init();

    return () => {
      cleanup?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // "New Session" always creates a fresh session (forceNew = true)
  const handleNewSession = useCallback(async (): Promise<void> => {
    await startSession(true);
  }, [startSession]);

  if (isInitializing) {
    return (
      <div className="chat-panel">
        <div className="chat-loading">
          <p>Starting session...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-layout">
      {/* Session Sidebar */}
      <SessionSidebar
        activeSessionId={sessionId}
        onNewSession={handleNewSession}
        onSwitchSession={switchSession}
        isOpen={isSidebarOpen}
        onClose={toggleSidebar}
      />

      {/* Main Chat Panel */}
      <div className="chat-panel">
        {/* Chat Header with controls */}
        <div className="chat-header">
          <button
            className="chat-sidebar-toggle"
            onClick={toggleSidebar}
            title="Session history"
          >
            ☰
          </button>
          <button
            className="chat-new-session-btn"
            onClick={() => void handleNewSession()}
            title="New session"
          >
            + New
          </button>
          <div className="chat-header-spacer" />
          <ContextIndicator budget={contextBudget} />
        </div>

        {/* Error banner */}
        {error && (
          <div className="chat-error">
            <span className="error-text">{error}</span>
            <button className="error-dismiss" onClick={clearError}>
              &times;
            </button>
          </div>
        )}

        {/* Message list */}
        <MessageList
          messages={messages}
          isGenerating={isGenerating}
          streamingContent={streamingContent}
        />

        {/* Input */}
        <MessageInput
          onSend={sendMessage}
          disabled={isGenerating || !sessionId}
        />

        {/* Confirmation dialog */}
        {pendingConfirmation && (
          <ConfirmationDialog
            request={pendingConfirmation}
            onRespond={respondToConfirmation}
          />
        )}
      </div>
    </div>
  );
}
