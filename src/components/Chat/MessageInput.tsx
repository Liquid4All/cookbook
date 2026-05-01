/**
 * MessageInput — text input area for sending messages.
 *
 * Supports Enter to send (Shift+Enter for newline) and disables
 * input while the assistant is generating. Includes an InputToolbar
 * below the textarea for folder context (Cowork-style "Work in a folder").
 * Implements debouncing to prevent duplicate sends.
 */

import { useCallback, useRef, useState } from "react";

import { InputToolbar } from "./InputToolbar";

interface MessageInputProps {
  readonly onSend: (content: string) => void;
  readonly disabled: boolean;
}

/** Minimum time between send requests to prevent duplicates (500ms) */
const SEND_DEBOUNCE_MS = 500;

export function MessageInput({
  onSend,
  disabled,
}: MessageInputProps): React.JSX.Element {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const lastSendTimeRef = useRef<number>(0);
  const [isDebouncing, setIsDebouncing] = useState(false);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;

    // Debounce: ignore clicks within 500ms
    const now = Date.now();
    if (now - lastSendTimeRef.current < SEND_DEBOUNCE_MS) {
      setIsDebouncing(true);
      setTimeout(() => setIsDebouncing(false), SEND_DEBOUNCE_MS);
      return;
    }
    lastSendTimeRef.current = now;

    onSend(trimmed);
    setValue("");

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [value, disabled, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>): void => {
    setValue(e.target.value);

    // Auto-resize textarea
    const textarea = e.target;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  };

  const isLoading = disabled || isDebouncing;

  return (
    <div className="message-input-wrapper">
      <div className="message-input-row">
        <textarea
          ref={textareaRef}
          className="message-input"
          value={value}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={
            isDebouncing
              ? "Please wait..."
              : disabled
                ? "Waiting for response..."
                : "Type a message..."
          }
          disabled={isLoading}
          rows={1}
        />
        <button
          className={`send-button ${isDebouncing ? "debouncing" : ""}`}
          onClick={handleSend}
          disabled={isLoading || !value.trim()}
          aria-label={isDebouncing ? "Please wait..." : "Send message"}
        >
          {isDebouncing ? (
            <span className="send-button-spinner">⏳</span>
          ) : (
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </button>
      </div>
      <InputToolbar />
    </div>
  );
}
