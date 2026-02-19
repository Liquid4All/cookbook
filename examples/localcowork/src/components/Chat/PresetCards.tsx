/**
 * PresetCards — curated demo prompt cards shown on the empty chat state.
 *
 * Each card targets a use case where LFM2-24B-A2B has high accuracy:
 * - Security Scan (UC-3, 90% accuracy)
 * - Daily Briefing (UC-7, calendar 100% + task 88%)
 * - Read a Receipt (UC-1 simplified, document 83%)
 * - Organize Downloads (UC-4 simplified, file-ops 80%)
 *
 * Clicking a card fires `sendMessage` immediately — no input population.
 */

import { useChatStore } from "../../stores/chatStore";

interface Preset {
  readonly icon: string;
  readonly title: string;
  readonly description: string;
  readonly prompt: string;
}

const PRESETS: readonly Preset[] = [
  {
    icon: "\u{1F512}",
    title: "Security Scan",
    description:
      "Find exposed passwords, API keys, and personal data in a folder",
    prompt:
      "Scan my Documents folder for any exposed passwords, API keys, credit card numbers, or other sensitive data. Show me what you find.",
  },
  {
    icon: "\u{1F4CB}",
    title: "Daily Briefing",
    description: "Today\u2019s calendar, tasks, and priorities at a glance",
    prompt:
      "Give me my daily briefing: what's on my calendar today, what tasks are due, and what's overdue.",
  },
  {
    icon: "\u{1F9FE}",
    title: "Read a Receipt",
    description:
      "Extract vendor, date, amount, and line items from a photo",
    prompt:
      "I have a receipt photo on my Desktop. Extract the vendor name, date, total amount, and line items from it.",
  },
  {
    icon: "\u{1F4C2}",
    title: "Organize Downloads",
    description: "Sort and rename files in your Downloads folder",
    prompt:
      "Look at my Downloads folder and suggest how to organize it. Group files by type and suggest better names for any that have generic names like 'Screenshot' or 'Untitled'.",
  },
] as const;

export function PresetCards(): React.JSX.Element {
  const sendMessage = useChatStore((s) => s.sendMessage);
  const isGenerating = useChatStore((s) => s.isGenerating);
  const sessionId = useChatStore((s) => s.sessionId);

  const handleClick = (prompt: string): void => {
    if (!isGenerating && sessionId) {
      void sendMessage(prompt);
    }
  };

  return (
    <div className="preset-card-grid">
      {PRESETS.map((preset) => (
        <button
          key={preset.title}
          className="preset-card"
          disabled={isGenerating || !sessionId}
          onClick={() => handleClick(preset.prompt)}
          type="button"
        >
          <span className="preset-card-icon">{preset.icon}</span>
          <span className="preset-card-title">{preset.title}</span>
          <span className="preset-card-desc">{preset.description}</span>
        </button>
      ))}
    </div>
  );
}
