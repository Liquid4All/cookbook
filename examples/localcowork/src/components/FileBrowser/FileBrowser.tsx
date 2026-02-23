/**
 * FileBrowser — left sidebar showing a directory tree.
 *
 * Displays the user's file system in a collapsible tree view.
 * Integrates with the fileBrowserStore for state management
 * and Tauri IPC commands for filesystem operations.
 *
 * The **working folder** — the directory that file operations scope to —
 * can be set via the "Choose Folder" button (native OS picker) or by
 * clicking the pin icon next to any directory in the tree.
 */

import { useCallback, useEffect } from "react";

import { useFileBrowserStore } from "../../stores/fileBrowserStore";
import { getParentPath } from "../../utils/pathUtils";
import { DirectoryTree } from "./DirectoryTree";
import { PathBreadcrumb } from "./PathBreadcrumb";

/** Extract just the folder name from an absolute path. */
function folderName(path: string): string {
  const parts = path.replace(/\/+$/, "").split("/");
  return parts[parts.length - 1] || path;
}

/** Open the native OS folder picker. Returns null if cancelled. */
async function pickFolder(): Promise<string | null> {
  try {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const selected = await open({ directory: true, multiple: false });
    if (selected && typeof selected === "string") {
      return selected;
    }
  } catch {
    // Plugin unavailable
  }
  return null;
}

export function FileBrowser(): React.JSX.Element {
  const {
    rootPath,
    directoryContents,
    expandedDirs,
    selectedPath,
    loadingPaths,
    error,
    isVisible,
    workingDirectory,
    initialize,
    toggleDir,
    selectPath,
    navigateTo,
    clearError,
    setWorkingDirectory,
    clearWorkingDirectory,
  } = useFileBrowserStore();

  // Initialize on mount
  useEffect(() => {
    void initialize();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleNavigateUp = useCallback(() => {
    if (rootPath.length === 0) {
      return;
    }
    const parent = getParentPath(rootPath);
    if (parent !== rootPath) {
      navigateTo(parent);
    }
  }, [rootPath, navigateTo]);

  /** Open native folder picker and set as working folder. */
  const handleChooseFolder = useCallback(async (): Promise<void> => {
    const picked = await pickFolder();
    if (picked != null) {
      setWorkingDirectory(picked);
    }
  }, [setWorkingDirectory]);

  /** Set a directory from the tree as the working folder. */
  const handleSetWorkingDir = useCallback(
    (path: string) => {
      setWorkingDirectory(path);
    },
    [setWorkingDirectory],
  );

  if (!isVisible) {
    return <></>;
  }

  return (
    <aside className="file-browser" role="navigation" aria-label="File browser">
      {/* Header */}
      <div className="file-browser-header">
        <span className="file-browser-title">Files</span>
        <div className="file-browser-header-actions">
          <button
            className="file-browser-up-btn"
            onClick={() => void handleChooseFolder()}
            type="button"
            title="Choose working folder"
          >
            &#128194;
          </button>
          <button
            className="file-browser-up-btn"
            onClick={handleNavigateUp}
            type="button"
            title="Go to parent directory"
            disabled={rootPath.length === 0}
          >
            &#8593;
          </button>
        </div>
      </div>

      {/* Working folder indicator or choose CTA */}
      {workingDirectory != null ? (
        <div className="working-dir-badge">
          <span className="working-dir-icon">&#128194;</span>
          <span className="working-dir-path" title={workingDirectory}>
            {folderName(workingDirectory)}
          </span>
          <button
            className="working-dir-clear"
            onClick={clearWorkingDirectory}
            type="button"
            title="Clear working folder"
          >
            &times;
          </button>
        </div>
      ) : (
        <button
          className="working-dir-choose-btn"
          onClick={() => void handleChooseFolder()}
          type="button"
        >
          <span className="working-dir-choose-icon">&#128194;</span>
          <span className="working-dir-choose-label">Choose Folder</span>
          <span className="working-dir-choose-hint">
            Point LocalCowork at a folder on your machine
          </span>
        </button>
      )}

      {/* Path breadcrumb */}
      {rootPath.length > 0 && (
        <PathBreadcrumb path={rootPath} onNavigate={navigateTo} />
      )}

      {/* Error banner */}
      {error != null && (
        <div className="file-browser-error">
          <span className="file-browser-error-text">{error}</span>
          <button
            className="file-browser-error-dismiss"
            onClick={clearError}
            type="button"
          >
            &times;
          </button>
        </div>
      )}

      {/* Directory tree */}
      <div className="file-browser-tree" role="tree">
        {rootPath.length > 0 ? (
          <DirectoryTree
            dirPath={rootPath}
            depth={0}
            directoryContents={directoryContents}
            expandedDirs={expandedDirs}
            selectedPath={selectedPath}
            loadingPaths={loadingPaths}
            workingDirectory={workingDirectory}
            onToggle={toggleDir}
            onSelect={selectPath}
            onSetWorkingDir={handleSetWorkingDir}
          />
        ) : (
          <div className="file-browser-loading">Loading...</div>
        )}
      </div>
    </aside>
  );
}
