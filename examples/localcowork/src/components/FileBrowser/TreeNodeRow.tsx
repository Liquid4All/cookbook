/**
 * TreeNodeRow — a single row in the directory tree.
 *
 * Renders a file or directory entry with indentation, expand/collapse
 * toggle for directories, icon, name, and file size.
 *
 * Directories show a small folder-pin button on hover to set the
 * entry as the working folder for file operations.
 */

import { useCallback } from "react";

import type { FileEntry } from "../../types";
import { FileIcon } from "./FileIcon";

interface TreeNodeRowProps {
  readonly entry: FileEntry;
  readonly depth: number;
  readonly isExpanded: boolean;
  readonly isSelected: boolean;
  readonly isLoading: boolean;
  /** Whether this directory is the active working folder. */
  readonly isWorkingDir: boolean;
  readonly onToggle: (path: string) => void;
  readonly onSelect: (path: string) => void;
  /** Callback to set this directory as the working folder. */
  readonly onSetWorkingDir: (path: string) => void;
}

/** Format file size in human-readable form. */
function formatSize(bytes: number): string {
  if (bytes === 0) {
    return "";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

export function TreeNodeRow({
  entry,
  depth,
  isExpanded,
  isSelected,
  isLoading,
  isWorkingDir,
  onToggle,
  onSelect,
  onSetWorkingDir,
}: TreeNodeRowProps): React.JSX.Element {
  const isDir = entry.entryType === "dir";
  const paddingLeft = 12 + depth * 16;

  const handleClick = useCallback(() => {
    onSelect(entry.path);
    if (isDir) {
      onToggle(entry.path);
    }
  }, [entry.path, isDir, onSelect, onToggle]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        handleClick();
      }
    },
    [handleClick],
  );

  /** Set this directory as working folder (stop event propagation). */
  const handleSetWorkingDir = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onSetWorkingDir(entry.path);
    },
    [entry.path, onSetWorkingDir],
  );

  return (
    <div
      className={`tree-node-row${isSelected ? " tree-node-selected" : ""}${isWorkingDir ? " tree-node-working-dir" : ""}`}
      style={{ paddingLeft: `${paddingLeft}px` }}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="treeitem"
      aria-expanded={isDir ? isExpanded : undefined}
      aria-selected={isSelected}
      tabIndex={0}
    >
      {/* Expand/collapse toggle for directories */}
      <span className="tree-node-toggle">
        {isDir ? (
          isLoading ? (
            <span className="tree-node-spinner">{"\u21BB"}</span>
          ) : isExpanded ? (
            "\u25BC"
          ) : (
            "\u25B6"
          )
        ) : (
          ""
        )}
      </span>

      <FileIcon name={entry.name} entryType={entry.entryType} />

      <span className="tree-node-name" title={entry.path}>
        {entry.name}
      </span>

      {/* Working folder pin button — shown on hover for directories */}
      {isDir && (
        <button
          className={`tree-node-pin${isWorkingDir ? " tree-node-pin-active" : ""}`}
          onClick={handleSetWorkingDir}
          type="button"
          title={isWorkingDir ? "Current working folder" : "Set as working folder"}
          aria-label={
            isWorkingDir
              ? "Current working folder"
              : `Set ${entry.name} as working folder`
          }
        >
          {"\uD83D\uDCC2"}
        </button>
      )}

      {!isDir && entry.size > 0 && (
        <span className="tree-node-size">{formatSize(entry.size)}</span>
      )}
    </div>
  );
}
