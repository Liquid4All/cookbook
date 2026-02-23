/**
 * PresetCards — demo-aligned starter prompt cards shown on the empty chat state.
 *
 * Each card maps to a verified demo workflow from `docs/demo/lfm2-24b-demo.md`.
 * Prompts use absolute paths resolved from the user's home directory at mount time.
 *
 * Shows 3 randomly-selected cards at a time. A shuffle button re-randomizes.
 * Clicking a card fires `sendMessage` immediately with the resolved prompt.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

import { useChatStore } from "../../stores/chatStore";

interface Preset {
  readonly icon: string;
  readonly title: string;
  readonly description: string;
  /** Prompt template — `{home}` is replaced with the user's home directory. */
  readonly promptTemplate: string;
}

const ALL_PRESETS: readonly Preset[] = [
  {
    icon: "\u{1F50D}",
    title: "Scan for leaked secrets",
    description: "Find exposed API keys and passwords on your Desktop",
    promptTemplate:
      "Scan {home}/Desktop for any exposed API keys, passwords, or secrets",
  },
  {
    icon: "\u{1F4C4}",
    title: "Compare two documents",
    description: "Diff two files and summarize the changes",
    promptTemplate:
      "List the files in {home}/Documents and help me compare two of them",
  },
  {
    icon: "\u{1F4F8}",
    title: "Capture my screen",
    description: "Take a screenshot of what's on screen right now",
    promptTemplate: "Take a screenshot of my screen",
  },
  {
    icon: "\u{1F6E1}\uFE0F",
    title: "Find personal data",
    description: "Scan for SSNs, emails, and suggest a cleanup plan",
    promptTemplate:
      "Scan {home}/Desktop for personal data like SSNs, credit card numbers, and emails, then suggest a cleanup plan",
  },
  {
    icon: "\u{1F4C2}",
    title: "Organize my Downloads",
    description: "List Downloads and suggest how to organize the files",
    promptTemplate:
      "List what's in {home}/Downloads and suggest how to organize it by file type",
  },
] as const;

const VISIBLE_COUNT = 3;

/** Pick `count` unique random indices from `[0, total)`. */
function pickRandom(total: number, count: number): readonly number[] {
  const indices = Array.from({ length: total }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices.slice(0, count);
}

export function PresetCards(): React.JSX.Element {
  const sendMessage = useChatStore((s) => s.sendMessage);
  const isGenerating = useChatStore((s) => s.isGenerating);
  const sessionId = useChatStore((s) => s.sessionId);

  const [homeDir, setHomeDir] = useState<string>("");
  const [visibleIndices, setVisibleIndices] = useState<readonly number[]>(() =>
    pickRandom(ALL_PRESETS.length, VISIBLE_COUNT),
  );

  // Resolve home directory once on mount.
  useEffect(() => {
    void invoke<string>("get_home_dir").then((dir) => {
      setHomeDir(dir);
    });
  }, []);

  const visiblePresets = useMemo(
    () => visibleIndices.map((i) => ALL_PRESETS[i]),
    [visibleIndices],
  );

  /** Replace `{home}` placeholder with the actual home directory. */
  const resolvePrompt = useCallback(
    (template: string): string => template.replaceAll("{home}", homeDir),
    [homeDir],
  );

  const handleClick = useCallback(
    (template: string): void => {
      if (!isGenerating && sessionId && homeDir) {
        void sendMessage(resolvePrompt(template));
      }
    },
    [isGenerating, sessionId, homeDir, sendMessage, resolvePrompt],
  );

  const handleShuffle = useCallback((): void => {
    setVisibleIndices(pickRandom(ALL_PRESETS.length, VISIBLE_COUNT));
  }, []);

  return (
    <div className="preset-section">
      <div className="preset-section-header">
        <span className="preset-section-label">Try one of these</span>
        <button
          className="preset-shuffle-btn"
          onClick={handleShuffle}
          type="button"
          aria-label="Shuffle prompts"
          title="Shuffle"
        >
          &#x21C4;
        </button>
      </div>
      <div className="preset-card-list">
        {visiblePresets.map((preset) => (
          <button
            key={preset.title}
            className="preset-card"
            disabled={isGenerating || !sessionId || !homeDir}
            onClick={() => {
              handleClick(preset.promptTemplate);
            }}
            type="button"
          >
            <span className="preset-card-icon">{preset.icon}</span>
            <div className="preset-card-text">
              <span className="preset-card-title">{preset.title}</span>
              <span className="preset-card-desc">{preset.description}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
