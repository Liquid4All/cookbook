/**
 * DirectoryTree — recursive tree rendering of directory contents.
 *
 * Lazily loads directory contents when a folder is expanded.
 * Renders TreeNodeRow for each entry and recursively renders
 * children of expanded directories.
 */

import type { FileEntry } from "../../types";
import { TreeNodeRow } from "./TreeNodeRow";

interface DirectoryTreeProps {
  /** Path of the directory whose contents to render. */
  readonly dirPath: string;
  /** Current indentation depth (0 = root). */
  readonly depth: number;
  /** Map of directory path → loaded entries. */
  readonly directoryContents: Record<string, readonly FileEntry[]>;
  /** Set of expanded directory paths. */
  readonly expandedDirs: ReadonlySet<string>;
  /** Currently selected path. */
  readonly selectedPath: string | null;
  /** Set of paths currently being loaded. */
  readonly loadingPaths: ReadonlySet<string>;
  /** The active working directory (null if not set). */
  readonly workingDirectory: string | null;
  /** Callback to toggle a directory's expanded state. */
  readonly onToggle: (path: string) => void;
  /** Callback to select a path. */
  readonly onSelect: (path: string) => void;
  /** Callback to set a directory as the working folder. */
  readonly onSetWorkingDir: (path: string) => void;
}

export function DirectoryTree({
  dirPath,
  depth,
  directoryContents,
  expandedDirs,
  selectedPath,
  loadingPaths,
  workingDirectory,
  onToggle,
  onSelect,
  onSetWorkingDir,
}: DirectoryTreeProps): React.JSX.Element {
  const entries = directoryContents[dirPath];

  if (entries == null) {
    return <></>;
  }

  return (
    <div className="directory-tree" role="group">
      {entries.map((entry) => {
        const isDir = entry.entryType === "dir";
        const isExpanded = expandedDirs.has(entry.path);
        const isSelected = entry.path === selectedPath;
        const isLoading = loadingPaths.has(entry.path);
        const isWorkingDir = entry.path === workingDirectory;

        return (
          <div key={entry.path}>
            <TreeNodeRow
              entry={entry}
              depth={depth}
              isExpanded={isExpanded}
              isSelected={isSelected}
              isLoading={isLoading}
              isWorkingDir={isWorkingDir}
              onToggle={onToggle}
              onSelect={onSelect}
              onSetWorkingDir={onSetWorkingDir}
            />
            {/* Recursively render children of expanded directories */}
            {isDir && isExpanded && (
              <DirectoryTree
                dirPath={entry.path}
                depth={depth + 1}
                directoryContents={directoryContents}
                expandedDirs={expandedDirs}
                selectedPath={selectedPath}
                loadingPaths={loadingPaths}
                workingDirectory={workingDirectory}
                onToggle={onToggle}
                onSelect={onSelect}
                onSetWorkingDir={onSetWorkingDir}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
