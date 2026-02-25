/**
 * File browser and filesystem types shared between frontend and backend.
 *
 * These mirror the filesystem MCP server's return types and the
 * Tauri IPC command payloads for directory browsing.
 */

/** Type of a filesystem entry. */
export type FileEntryType = "file" | "dir" | "symlink";

/** A single file or directory entry returned from list_directory. */
export interface FileEntry {
  readonly name: string;
  readonly path: string;
  readonly entryType: FileEntryType;
  readonly size: number;
  readonly modified: string;
}

/** A node in the directory tree with optional lazy-loaded children. */
export interface TreeNode {
  readonly entry: FileEntry;
  readonly children?: readonly TreeNode[];
  readonly isLoading: boolean;
}
