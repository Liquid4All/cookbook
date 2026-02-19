/**
 * FileBrowser — left sidebar showing a directory tree.
 *
 * Displays the user's file system in a collapsible tree view.
 * Integrates with the fileBrowserStore for state management
 * and Tauri IPC commands for filesystem operations.
 */

import { useCallback, useEffect } from "react";

import { useFileBrowserStore } from "../../stores/fileBrowserStore";
import { getParentPath } from "../../utils/pathUtils";
import { DirectoryTree } from "./DirectoryTree";
import { PathBreadcrumb } from "./PathBreadcrumb";

export function FileBrowser(): React.JSX.Element {
  const {
    rootPath,
    directoryContents,
    expandedDirs,
    selectedPath,
    loadingPaths,
    error,
    isVisible,
    initialize,
    toggleDir,
    selectPath,
    navigateTo,
    clearError,
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

  if (!isVisible) {
    return <></>;
  }

  return (
    <aside className="file-browser" role="navigation" aria-label="File browser">
      {/* Header */}
      <div className="file-browser-header">
        <span className="file-browser-title">Files</span>
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
            onToggle={toggleDir}
            onSelect={selectPath}
          />
        ) : (
          <div className="file-browser-loading">Loading...</div>
        )}
      </div>
    </aside>
  );
}
