/**
 * MessageBubble — renders a single chat message.
 *
 * Handles user, assistant, and tool result messages with appropriate
 * styling and content rendering. Integrates ToolTrace for assistant
 * messages that include tool calls.
 */

import type { ChatMessage } from "../../types";
import { MarkdownContent } from "./MarkdownContent";
import { ToolTrace } from "./ToolTrace";

interface MessageBubbleProps {
  readonly message: ChatMessage;
  /** Full message history for correlating tool results. */
  readonly allMessages: readonly ChatMessage[];
  /** Whether the assistant is still generating. */
  readonly isGenerating: boolean;
}

/** Get a human-friendly label for the message role. */
function roleLabel(role: string): string {
  switch (role) {
    case "user":
      return "You";
    case "assistant":
      return "Assistant";
    case "tool":
      return "Tool Result";
    case "system":
      return "System";
    default:
      return role;
  }
}

/**
 * Check if a tool result message has a corresponding ToolTrace parent.
 *
 * If a preceding assistant message contains a toolCall with a matching ID,
 * the tool result will be rendered inside the ToolTrace instead.
 */
function hasToolTraceParent(
  message: ChatMessage,
  allMessages: readonly ChatMessage[],
): boolean {
  if (message.role !== "tool" || message.toolCallId == null) {
    return false;
  }
  return allMessages.some(
    (m) =>
      m.role === "assistant" &&
      m.toolCalls != null &&
      m.toolCalls.some((tc) => tc.id === message.toolCallId),
  );
}

export function MessageBubble({
  message,
  allMessages,
  isGenerating,
}: MessageBubbleProps): React.JSX.Element {
  const isUser = message.role === "user";
  const isTool = message.role === "tool";

  // Hide tool result messages that have a ToolTrace parent —
  // they'll be rendered inside the ToolTrace tree view.
  if (isTool && hasToolTraceParent(message, allMessages)) {
    return <div className="message-hidden" />;
  }

  // Render orphan tool results (no matching ToolTrace parent) as
  // a compact inline result — this happens during live streaming
  // or when history is loaded without full ToolCall metadata.
  if (isTool) {
    return (
      <div className="message-bubble message-tool">
        <div className="message-header">
          <span className="message-role">{roleLabel(message.role)}</span>
          <span className="message-time">
            {new Date(message.timestamp).toLocaleTimeString()}
          </span>
        </div>
        <div className="message-content">
          <pre className="message-tool-result">{message.content}</pre>
        </div>
      </div>
    );
  }

  return (
    <div className={`message-bubble message-${message.role}`}>
      <div className="message-header">
        <span className="message-role">{roleLabel(message.role)}</span>
        <span className="message-time">
          {new Date(message.timestamp).toLocaleTimeString()}
        </span>
      </div>

      <div className="message-content">
        {/* Text content */}
        {message.content != null && message.content.length > 0 && (
          <div
            className={
              isUser ? "message-text-user" : "message-text-assistant"
            }
          >
            {isUser ? (
              message.content
            ) : (
              <MarkdownContent content={message.content} />
            )}
          </div>
        )}

        {/* Tool trace (assistant messages with tool calls) */}
        {message.toolCalls != null && message.toolCalls.length > 0 && (
          <ToolTrace
            toolCalls={message.toolCalls}
            allMessages={allMessages}
            isExecuting={isGenerating}
          />
        )}
      </div>
    </div>
  );
}
