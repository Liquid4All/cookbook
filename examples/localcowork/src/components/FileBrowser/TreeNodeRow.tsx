/**
 * TreeNodeRow — a single row in the directory tree.
 *
 * Renders a file or directory entry with indentation, expand/collapse
 * toggle for directories, icon, name, and file size.
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
  readonly onToggle: (path: string) => void;
  readonly onSelect: (path: string) => void;
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
  onToggle,
  onSelect,
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

  return (
    <div
      className={`tree-node-row ${isSelected ? "tree-node-selected" : ""}`}
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
            <span className="tree-node-spinner">\u21BB</span>
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

      {!isDir && entry.size > 0 && (
        <span className="tree-node-size">{formatSize(entry.size)}</span>
      )}
    </div>
  );
}
