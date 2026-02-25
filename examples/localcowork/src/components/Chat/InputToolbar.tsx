/**
 * InputToolbar — contextual controls below the chat textarea.
 *
 * Renders a "Work in a folder" button that opens the native OS folder picker.
 * When a folder is selected, shows the folder name as a chip with a clear button.
 * Mirrors the Claude Cowork pattern of attaching folder context to the input area.
 */

import { useCallback } from "react";

import { useFileBrowserStore } from "../../stores/fileBrowserStore";

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
    // Plugin unavailable (e.g., browser dev mode) — fall through
  }
  return null;
}

/** Extract the last path component (folder name) from a full path. */
function folderName(fullPath: string): string {
  const parts = fullPath.replace(/\/+$/, "").split("/");
  return parts[parts.length - 1] || fullPath;
}

export function InputToolbar(): React.JSX.Element {
  const workingDir = useFileBrowserStore((s) => s.workingDirectory);
  const setWorkingDir = useFileBrowserStore((s) => s.setWorkingDirectory);
  const clearWorkingDir = useFileBrowserStore((s) => s.clearWorkingDirectory);

  const handlePickFolder = useCallback(async (): Promise<void> => {
    const picked = await pickFolder();
    if (picked != null) {
      setWorkingDir(picked);
    }
  }, [setWorkingDir]);

  const handleClear = useCallback(
    (e: React.MouseEvent): void => {
      e.stopPropagation(); // Don't re-trigger the picker from the chip click
      clearWorkingDir();
    },
    [clearWorkingDir],
  );

  return (
    <div className="input-toolbar">
      {workingDir != null ? (
        <button
          className="input-toolbar-folder-chip"
          type="button"
          title={workingDir}
          onClick={() => void handlePickFolder()}
        >
          <span className="input-toolbar-folder-icon">&#128194;</span>
          <span className="input-toolbar-folder-name">
            {folderName(workingDir)}
          </span>
          <span
            className="input-toolbar-folder-clear"
            role="button"
            tabIndex={0}
            onClick={handleClear}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") handleClear(e as unknown as React.MouseEvent);
            }}
          >
            &times;
          </span>
        </button>
      ) : (
        <button
          className="input-toolbar-folder-btn"
          type="button"
          onClick={() => void handlePickFolder()}
        >
          <span className="input-toolbar-folder-icon">&#128194;</span>
          <span className="input-toolbar-folder-label">Work in a folder</span>
        </button>
      )}
    </div>
  );
}
