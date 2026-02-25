/**
 * Chat message types shared between frontend and backend.
 *
 * These mirror the Rust `agent_core::types` structures and the
 * Tauri IPC command payloads.
 */

/** Message role, matching the OpenAI convention. */
export type MessageRole = "system" | "user" | "assistant" | "tool";

/** A tool call made by the assistant. */
export interface ToolCall {
  readonly id: string;
  readonly name: string;
  readonly arguments: Record<string, unknown>;
}

/** A tool call result returned to the assistant. */
export interface ToolResult {
  readonly toolCallId: string;
  readonly toolName: string;
  readonly success: boolean;
  readonly result?: unknown;
  readonly error?: string;
  readonly executionTimeMs: number;
}

/** A single message in the chat history. */
export interface ChatMessage {
  readonly id: number;
  readonly sessionId: string;
  readonly timestamp: string;
  readonly role: MessageRole;
  readonly content?: string;
  readonly toolCalls?: readonly ToolCall[];
  readonly toolCallId?: string;
  readonly toolResult?: unknown;
  readonly tokenCount: number;
  /** Whether this message is still being streamed. */
  readonly isStreaming?: boolean;
}

/** Confirmation request from the backend. */
export interface ConfirmationRequest {
  readonly requestId: string;
  readonly toolName: string;
  readonly arguments: Record<string, unknown>;
  readonly preview: string;
  readonly confirmationRequired: boolean;
  readonly undoSupported: boolean;
  readonly isDestructive: boolean;
}

/** User's response to a confirmation request. */
export type ConfirmationResponse =
  | { type: "confirmed" }
  | { type: "confirmedForSession" }
  | { type: "confirmedAlways" }
  | { type: "rejected" }
  | { type: "edited"; newArguments: Record<string, unknown> };

/** Context window budget snapshot. */
export interface ContextBudget {
  readonly total: number;
  readonly systemPrompt: number;
  readonly toolDefinitions: number;
  readonly conversationHistory: number;
  readonly activeContext: number;
  readonly remaining: number;
}

/** Session status from the backend. */
export interface SessionStatus {
  readonly sessionId: string;
  readonly messageCount: number;
  readonly contextBudget: ContextBudget;
  readonly runningServers: readonly string[];
  readonly toolCount: number;
}
