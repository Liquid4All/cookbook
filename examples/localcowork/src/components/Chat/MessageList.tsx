/**
 * MessageList — scrollable container for all chat messages.
 *
 * Auto-scrolls to the bottom on new messages and displays a streaming
 * indicator when the assistant is generating.
 */

import { useEffect, useRef } from "react";

import type { ChatMessage } from "../../types";
import { MarkdownContent } from "./MarkdownContent";
import { MessageBubble } from "./MessageBubble";
import { PresetCards } from "./PresetCards";

interface MessageListProps {
  readonly messages: readonly ChatMessage[];
  readonly isGenerating: boolean;
  readonly streamingContent: string;
}

export function MessageList({
  messages,
  isGenerating,
  streamingContent,
}: MessageListProps): React.JSX.Element {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages or streaming content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, streamingContent]);

  return (
    <div className="message-list">
      {messages.length === 0 && !isGenerating && (
        <div className="message-list-empty">
          <p className="empty-title">Your AI workspace, entirely on-device</p>
          <p className="empty-subtitle">
            Everything runs locally. Nothing leaves your machine.
          </p>
          <PresetCards />
        </div>
      )}

      {messages.map((msg) => (
        <MessageBubble
          key={msg.id}
          message={msg}
          allMessages={messages}
          isGenerating={isGenerating}
        />
      ))}

      {/* Streaming indicator */}
      {isGenerating && (
        <div className="message-bubble message-assistant">
          <div className="message-header">
            <span className="message-role">Assistant</span>
          </div>
          <div className="message-content">
            {streamingContent ? (
              <div className="message-text-assistant">
                <MarkdownContent content={streamingContent} />
              </div>
            ) : (
              <div className="streaming-indicator">
                <span className="dot" />
                <span className="dot" />
                <span className="dot" />
              </div>
            )}
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
