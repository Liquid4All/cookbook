/**
 * MessageInput — text input area for sending messages.
 *
 * Supports Enter to send (Shift+Enter for newline) and disables
 * input while the assistant is generating. Includes an InputToolbar
 * below the textarea for folder context (Cowork-style "Work in a folder").
 */

import { useRef, useState } from "react";

import { InputToolbar } from "./InputToolbar";

interface MessageInputProps {
  readonly onSend: (content: string) => void;
  readonly disabled: boolean;
}

export function MessageInput({
  onSend,
  disabled,
}: MessageInputProps): React.JSX.Element {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = (): void => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;

    onSend(trimmed);
    setValue("");

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

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
            disabled ? "Waiting for response..." : "Type a message..."
          }
          disabled={disabled}
          rows={1}
        />
        <button
          className="send-button"
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          aria-label="Send message"
        >
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
        </button>
      </div>
      <InputToolbar />
    </div>
  );
}
