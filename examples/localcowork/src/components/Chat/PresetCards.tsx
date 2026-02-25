/**
 * PresetCards — starter prompt cards shown on the empty chat state.
 *
 * Each card fires a single tool call against a verified, real implementation:
 * - Scan for leaked secrets   → security.scan_for_secrets (90%)
 * - What's on my clipboard?   → clipboard.get_clipboard   (80%)
 * - Tell me about my system   → system.get_system_info     (100%)
 * - Find personal data         → security.scan_for_pii     (90%)
 * - Organize my Downloads      → filesystem.list_dir       (80%)
 *
 * Security scan presets use `{cwd}` resolved from the file browser's
 * **working directory**. When no working directory is set, clicking a
 * scan card opens the native OS folder picker — pick a folder and the
 * scan starts immediately. Nothing is uploaded or copied; the agent is
 * just pointed at a local folder.
 *
 * Shows 3 randomly-selected cards at a time. A shuffle button re-randomizes.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

import { useChatStore } from "../../stores/chatStore";
import { useFileBrowserStore } from "../../stores/fileBrowserStore";

interface Preset {
  readonly icon: string;
  readonly title: string;
  readonly description: string;
  /**
   * Prompt template with placeholders:
   * - `{home}` → user's home directory
   * - `{cwd}`  → file browser's working directory
   */
  readonly promptTemplate: string;
  /** Whether this preset requires a working directory ({cwd}). */
  readonly needsCwd: boolean;
}

const ALL_PRESETS: readonly Preset[] = [
  {
    icon: "\u{1F50D}",
    title: "Scan for leaked secrets",
    description: "Find exposed API keys and passwords",
    promptTemplate:
      "Scan {cwd} for any exposed API keys, passwords, or secrets",
    needsCwd: true,
  },
  {
    icon: "\u{1F4CB}",
    title: "What's on my clipboard?",
    description: "Show the current contents of your system clipboard",
    promptTemplate: "Show me what's currently on my clipboard",
    needsCwd: false,
  },
  {
    icon: "\u{1F5A5}\uFE0F",
    title: "Tell me about my system",
    description: "Hardware, OS, memory, and disk usage at a glance",
    promptTemplate:
      "Show me my system information \u2014 hardware, OS, memory, and disk usage",
    needsCwd: false,
  },
  {
    icon: "\u{1F6E1}\uFE0F",
    title: "Find personal data",
    description: "Scan for SSNs, emails, and credit card numbers",
    promptTemplate:
      "Scan {cwd} for personal data like SSNs, credit card numbers, and emails",
    needsCwd: true,
  },
  {
    icon: "\u{1F4C2}",
    title: "Organize my Downloads",
    description: "List Downloads and suggest how to organize the files",
    promptTemplate:
      "List what's in {home}/Downloads and suggest how to organize it by file type",
    needsCwd: false,
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

/**
 * Open the native OS folder picker and return the selected path.
 * Returns null if the user cancels or the plugin is unavailable.
 */
async function pickFolder(): Promise<string | null> {
  try {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const selected = await open({ directory: true, multiple: false });
    if (selected && typeof selected === "string") {
      return selected;
    }
  } catch {
    // Plugin unavailable — fall through
  }
  return null;
}

export function PresetCards(): React.JSX.Element {
  const sendMessage = useChatStore((s) => s.sendMessage);
  const isGenerating = useChatStore((s) => s.isGenerating);
  const sessionId = useChatStore((s) => s.sessionId);
  const workingDir = useFileBrowserStore((s) => s.workingDirectory);
  const setWorkingDir = useFileBrowserStore((s) => s.setWorkingDirectory);

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

  /** Replace `{home}` and `{cwd}` placeholders with resolved paths. */
  const resolvePrompt = useCallback(
    (template: string, cwd: string): string =>
      template.replaceAll("{home}", homeDir).replaceAll("{cwd}", cwd),
    [homeDir],
  );

  /** Handle clicking a preset card. */
  const handleClick = useCallback(
    async (preset: Preset): Promise<void> => {
      if (isGenerating || !sessionId || !homeDir) {
        return;
      }

      let cwd = workingDir;

      // If the preset needs a working directory and none is set,
      // open the native folder picker. If the user picks a folder,
      // set it as working dir and immediately send the prompt.
      if (preset.needsCwd && cwd == null) {
        const picked = await pickFolder();
        if (picked == null) {
          return; // User cancelled the picker
        }
        setWorkingDir(picked);
        cwd = picked;
      }

      if (preset.needsCwd && cwd == null) {
        return;
      }

      void sendMessage(resolvePrompt(preset.promptTemplate, cwd ?? ""));
    },
    [isGenerating, sessionId, homeDir, workingDir, setWorkingDir, sendMessage, resolvePrompt],
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
        {visiblePresets.map((preset) => {
          const needsDir = preset.needsCwd && workingDir == null;
          return (
            <button
              key={preset.title}
              className={`preset-card${needsDir ? " preset-card-needs-folder" : ""}`}
              disabled={isGenerating || !sessionId || !homeDir}
              onClick={() => {
                void handleClick(preset);
              }}
              type="button"
              title={
                needsDir
                  ? "Click to choose a folder, then scan starts automatically"
                  : undefined
              }
            >
              <span className="preset-card-icon">{preset.icon}</span>
              <div className="preset-card-text">
                <span className="preset-card-title">{preset.title}</span>
                <span className="preset-card-desc">
                  {needsDir
                    ? "Choose a folder to scan"
                    : preset.description}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
