/**
 * File Browser Zustand store.
 *
 * Manages directory tree state: root path, expanded folders, selected
 * file, and lazy-loaded directory contents from Tauri IPC commands.
 *
 * Also manages the **working directory** — the folder that file operations
 * (scans, searches) scope to. This mirrors Claude Cowork's "project
 * directory" pattern: the product provides context, not the model.
 *
 * When `workingDirectory` is null, security scan presets are disabled
 * and the user is prompted to select a folder first.
 */

import { invoke } from "@tauri-apps/api/core";
import { create } from "zustand";

import type { FileEntry } from "../types";

/** State shape for the file browser. */
interface FileBrowserState {
  /** The root directory path being browsed. */
  rootPath: string;
  /** Map of directory path → its loaded entries. */
  directoryContents: Record<string, readonly FileEntry[]>;
  /** Set of currently expanded directory paths. */
  expandedDirs: Set<string>;
  /** Currently selected file/dir path. */
  selectedPath: string | null;
  /** Set of paths currently loading. */
  loadingPaths: Set<string>;
  /** Last error message, if any. */
  error: string | null;
  /** Whether the sidebar is visible. */
  isVisible: boolean;
  /**
   * The active working directory for file operations.
   *
   * Null until the user explicitly selects a folder. Preset prompts
   * that need a path resolve `{cwd}` from this value. When null,
   * those presets are disabled with a "select a folder" hint.
   */
  workingDirectory: string | null;
}

/** Actions for the file browser. */
interface FileBrowserActions {
  /** Initialize the file browser with the user's home directory. */
  initialize: () => Promise<void>;
  /** Load the contents of a directory. */
  loadDirectory: (path: string) => Promise<void>;
  /** Toggle a directory's expanded/collapsed state. */
  toggleDir: (path: string) => void;
  /** Select a file or directory. */
  selectPath: (path: string) => void;
  /** Navigate to a new root directory. */
  navigateTo: (path: string) => void;
  /** Toggle sidebar visibility. */
  toggleSidebar: () => void;
  /** Clear any error. */
  clearError: () => void;
  /** Set a folder as the working directory for file operations. */
  setWorkingDirectory: (path: string) => void;
  /** Clear the working directory. */
  clearWorkingDirectory: () => void;
}

type FileBrowserStore = FileBrowserState & FileBrowserActions;

export const useFileBrowserStore = create<FileBrowserStore>((set, get) => ({
  // ─── Initial state ────────────────────────────────────────────────────
  rootPath: "",
  directoryContents: {},
  expandedDirs: new Set<string>(),
  selectedPath: null,
  loadingPaths: new Set<string>(),
  error: null,
  isVisible: true,
  workingDirectory: null,

  // ─── Actions ──────────────────────────────────────────────────────────

  initialize: async (): Promise<void> => {
    try {
      const homePath = await invoke<string>("get_home_dir");
      set({ rootPath: homePath });
      await get().loadDirectory(homePath);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      set({ error: `Failed to initialize file browser: ${message}` });
    }
  },

  loadDirectory: async (path: string): Promise<void> => {
    const { loadingPaths, directoryContents } = get();

    // Skip if already loading or already loaded
    if (loadingPaths.has(path) || directoryContents[path] != null) {
      return;
    }

    // Mark as loading
    const newLoading = new Set(loadingPaths);
    newLoading.add(path);
    set({ loadingPaths: newLoading });

    try {
      const entries = await invoke<FileEntry[]>("list_directory", { path });

      set((state) => {
        const updatedLoading = new Set(state.loadingPaths);
        updatedLoading.delete(path);
        return {
          directoryContents: { ...state.directoryContents, [path]: entries },
          loadingPaths: updatedLoading,
        };
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      set((state) => {
        const updatedLoading = new Set(state.loadingPaths);
        updatedLoading.delete(path);
        return {
          loadingPaths: updatedLoading,
          error: `Failed to load ${path}: ${message}`,
        };
      });
    }
  },

  toggleDir: (path: string): void => {
    set((state) => {
      const newExpanded = new Set(state.expandedDirs);
      if (newExpanded.has(path)) {
        newExpanded.delete(path);
      } else {
        newExpanded.add(path);
        // Trigger lazy load if not already loaded
        if (state.directoryContents[path] == null) {
          void get().loadDirectory(path);
        }
      }
      return { expandedDirs: newExpanded };
    });
  },

  selectPath: (path: string): void => {
    set({ selectedPath: path });
  },

  navigateTo: (path: string): void => {
    set({
      rootPath: path,
      expandedDirs: new Set<string>(),
      directoryContents: {},
      selectedPath: null,
      error: null,
    });
    void get().loadDirectory(path);
  },

  toggleSidebar: (): void => {
    set((state) => ({ isVisible: !state.isVisible }));
  },

  clearError: (): void => {
    set({ error: null });
  },

  setWorkingDirectory: (path: string): void => {
    set({ workingDirectory: path });
  },

  clearWorkingDirectory: (): void => {
    set({ workingDirectory: null });
  },
}));
