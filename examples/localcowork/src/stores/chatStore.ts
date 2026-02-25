/**
 * Chat state management via Zustand.
 *
 * Manages the conversation history, streaming state, session lifecycle,
 * and confirmation flow. Communicates with the Rust backend via Tauri IPC.
 */

import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import type {
  ChatMessage,
  ConfirmationRequest,
  ConfirmationResponse,
  ContextBudget,
} from "../types";
import { useFileBrowserStore } from "./fileBrowserStore";

// ─── Backend Response Types ─────────────────────────────────────────────────

/** Mirrors the Rust `SessionInfo` struct returned by `start_session`. */
interface SessionInfo {
  readonly session_id: string;
  readonly resumed: boolean;
}

// ─── State Interface ────────────────────────────────────────────────────────

interface ChatState {
  /** Current session ID (null if no active session). */
  sessionId: string | null;
  /** Ordered list of messages in the current session. */
  messages: readonly ChatMessage[];
  /** Whether the assistant is currently generating a response. */
  isGenerating: boolean;
  /** Tokens being streamed for the current assistant response. */
  streamingContent: string;
  /** Whether a confirmation dialog is pending. */
  pendingConfirmation: ConfirmationRequest | null;
  /** Current context window budget. */
  contextBudget: ContextBudget | null;
  /** Error message from the last failed operation. */
  error: string | null;
  /** Whether the store is initializing a session. */
  isInitializing: boolean;
  /** Whether the session sidebar is open. */
  isSidebarOpen: boolean;

  // ─── Actions ────────────────────────────────────────────────────────

  /** Start or resume a chat session (resumes most recent by default). */
  startSession: (forceNew?: boolean) => Promise<void>;
  /** Send a user message and get an assistant response. */
  sendMessage: (content: string) => Promise<void>;
  /** Respond to a confirmation request. */
  respondToConfirmation: (response: ConfirmationResponse) => Promise<void>;
  /** Switch to an existing session with its messages. */
  switchSession: (
    sessionId: string,
    messages: readonly ChatMessage[],
  ) => void;
  /** Toggle the session sidebar. */
  toggleSidebar: () => void;
  /** Clear the current error. */
  clearError: () => void;
  /** Set up event listeners for backend events. Returns cleanup function. */
  setupListeners: () => Promise<() => void>;
}

// ─── Listener guard ─────────────────────────────────────────────────────────
// React.StrictMode double-mounts in dev. This guard ensures we only register
// event listeners once, even if setupListeners is called multiple times.

let activeUnlisteners: UnlistenFn[] = [];
let listenersRegistered = false;

// ─── Store ──────────────────────────────────────────────────────────────────

export const useChatStore = create<ChatState>((set, get) => ({
  sessionId: null,
  messages: [],
  isGenerating: false,
  streamingContent: "",
  pendingConfirmation: null,
  contextBudget: null,
  error: null,
  isInitializing: false,
  isSidebarOpen: false,

  startSession: async (forceNew?: boolean): Promise<void> => {
    set({ isInitializing: true, error: null });
    try {
      // Clean up orphan empty sessions on startup
      await invoke("cleanup_empty_sessions").catch(() => {
        /* non-critical */
      });

      const info = await invoke<SessionInfo>("start_session", {
        forceNew: forceNew === true,
      });

      if (info.resumed) {
        // Load the existing session's messages
        const messages = await invoke<ChatMessage[]>("load_session", {
          sessionId: info.session_id,
        });
        set({
          sessionId: info.session_id,
          messages,
          contextBudget: null,
          isInitializing: false,
        });
        // Fetch context budget for the resumed session
        void invoke<ContextBudget>("get_context_budget", {
          sessionId: info.session_id,
        })
          .then((budget) => set({ contextBudget: budget }))
          .catch(() => {
            /* non-critical */
          });
      } else {
        set({
          sessionId: info.session_id,
          messages: [],
          contextBudget: null,
          isInitializing: false,
        });
      }
    } catch (e) {
      set({
        error: `Failed to start session: ${String(e)}`,
        isInitializing: false,
      });
    }
  },

  sendMessage: async (content: string): Promise<void> => {
    const { sessionId } = get();
    if (!sessionId) {
      set({ error: "No active session" });
      return;
    }

    // Optimistically add user message
    const userMessage: ChatMessage = {
      id: Date.now(),
      sessionId,
      timestamp: new Date().toISOString(),
      role: "user",
      content,
      tokenCount: Math.ceil(content.length / 4),
    };

    set((state) => ({
      messages: [...state.messages, userMessage],
      isGenerating: true,
      streamingContent: "",
      error: null,
    }));

    try {
      // Send message to backend — response will come via streaming events.
      // Include the current working directory so the model knows which folder
      // the user is operating in (Cowork-style folder context).
      const workingDirectory =
        useFileBrowserStore.getState().workingDirectory;
      await invoke("send_message", {
        sessionId,
        content,
        workingDirectory,
      });
    } catch (e) {
      set({
        error: `Failed to send message: ${String(e)}`,
        isGenerating: false,
      });
    }
  },

  respondToConfirmation: async (
    response: ConfirmationResponse,
  ): Promise<void> => {
    const { pendingConfirmation } = get();
    if (!pendingConfirmation) return;

    try {
      await invoke("respond_to_confirmation", {
        requestId: pendingConfirmation.requestId,
        response,
      });
      set({ pendingConfirmation: null });
    } catch (e) {
      set({ error: `Confirmation error: ${String(e)}` });
    }
  },

  switchSession: (
    sessionId: string,
    messages: readonly ChatMessage[],
  ): void => {
    set({
      sessionId,
      messages,
      streamingContent: "",
      isGenerating: false,
      pendingConfirmation: null,
      contextBudget: null,
      error: null,
    });

    // Fetch the context budget for the loaded session
    void invoke<ContextBudget>("get_context_budget", { sessionId })
      .then((budget) => {
        set({ contextBudget: budget });
      })
      .catch(() => {
        // Non-critical — indicator just won't show
      });
  },

  toggleSidebar: (): void => {
    set((state) => ({ isSidebarOpen: !state.isSidebarOpen }));
  },

  clearError: (): void => {
    set({ error: null });
  },

  setupListeners: async (): Promise<() => void> => {
    // Guard: if listeners are already active, return a no-op cleanup.
    if (listenersRegistered) {
      return () => {
        /* second mount in StrictMode — no-op */
      };
    }
    listenersRegistered = true;

    const unlisteners: UnlistenFn[] = [];

    // Stream token events — incremental text from the model
    unlisteners.push(
      await listen<string>("stream-token", (event) => {
        set((state) => ({
          streamingContent: state.streamingContent + event.payload,
        }));
      }),
    );

    // Stream clear — sent when model decides to use tools after
    // initially streaming some text (avoids "phantom" text).
    unlisteners.push(
      await listen("stream-clear", () => {
        set({ streamingContent: "" });
      }),
    );

    // Stream complete — final assistant response
    unlisteners.push(
      await listen<ChatMessage>("stream-complete", (event) => {
        set((state) => ({
          messages: [...state.messages, event.payload],
          isGenerating: false,
          streamingContent: "",
        }));
      }),
    );

    // Tool call — assistant message with tool calls (before results)
    unlisteners.push(
      await listen<ChatMessage>("tool-call", (event) => {
        set((state) => ({
          messages: [...state.messages, event.payload],
          streamingContent: "",
        }));
      }),
    );

    // Tool result — result of executing a tool
    unlisteners.push(
      await listen<ChatMessage>("tool-result", (event) => {
        set((state) => ({
          messages: [...state.messages, event.payload],
        }));
      }),
    );

    // Confirmation request
    unlisteners.push(
      await listen<ConfirmationRequest>("confirmation-request", (event) => {
        set({ pendingConfirmation: event.payload });
      }),
    );

    // Context budget update
    unlisteners.push(
      await listen<ContextBudget>("context-budget", (event) => {
        set({ contextBudget: event.payload });
      }),
    );

    // Error events
    unlisteners.push(
      await listen<string>("agent-error", (event) => {
        set({ error: event.payload, isGenerating: false });
      }),
    );

    activeUnlisteners = unlisteners;

    // Return cleanup that tears down listeners and resets the guard
    return () => {
      for (const unlisten of activeUnlisteners) {
        unlisten();
      }
      activeUnlisteners = [];
      listenersRegistered = false;
    };
  },
}));
