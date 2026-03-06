/**
 * Settings Zustand store.
 *
 * Manages model configuration, MCP server status, permission grants,
 * and inference parameters via Tauri IPC commands.
 */

import { invoke } from "@tauri-apps/api/core";
import { create } from "zustand";

import type {
  McpServerStatus,
  ModelsOverview,
  PermissionGrant,
  SamplingConfig,
} from "../types";

/** State shape for the settings panel. */
interface SettingsState {
  /** Models configuration overview. */
  modelsOverview: ModelsOverview | null;
  /** MCP server statuses. */
  serverStatuses: readonly McpServerStatus[];
  /** Persistent permission grants. */
  permissionGrants: readonly PermissionGrant[];
  /** Runtime sampling hyperparameters. */
  samplingConfig: SamplingConfig | null;
  /** Whether the settings panel is visible. */
  isOpen: boolean;
  /** Which tab is active. */
  activeTab: "model" | "servers" | "permissions";
  /** Loading state. */
  isLoading: boolean;
  /** Error message, if any. */
  error: string | null;
  /** Toast message for config reload notification */
  configReloadNotification: string | null;
  /** Polling interval ID for config watching */
  configWatchInterval: ReturnType<typeof setInterval> | null;
}

/** Actions for the settings panel. */
interface SettingsActions {
  /** Load model configuration from backend. */
  loadModelsConfig: () => Promise<void>;
  /** Check if config file has changed and reload if needed. */
  checkAndReloadConfig: () => Promise<boolean>;
  /** Start polling for config file changes. */
  startConfigWatch: () => void;
  /** Stop polling for config file changes. */
  stopConfigWatch: () => void;
  /** Clear the config reload notification. */
  clearConfigReloadNotification: () => void;
  /** Load MCP server statuses from backend. */
  loadServerStatuses: () => Promise<void>;
  /** Load persistent permission grants from backend. */
  loadPermissionGrants: () => Promise<void>;
  /** Revoke a persistent permission grant. */
  revokePermission: (toolName: string) => Promise<void>;
  /** Load sampling configuration from backend. */
  loadSamplingConfig: () => Promise<void>;
  /** Update sampling configuration and persist to disk. */
  updateSamplingConfig: (config: SamplingConfig) => Promise<void>;
  /** Reset sampling configuration to defaults. */
  resetSamplingConfig: () => Promise<void>;
  /** Toggle settings panel visibility. */
  togglePanel: () => void;
  /** Set active tab. */
  setActiveTab: (tab: "model" | "servers" | "permissions") => void;
  /** Clear error. */
  clearError: () => void;
}

type SettingsStore = SettingsState & SettingsActions;

export const useSettingsStore = create<SettingsStore>((set, get) => ({
  // ─── Initial state ────────────────────────────────────────────────────
  modelsOverview: null,
  serverStatuses: [],
  permissionGrants: [],
  samplingConfig: null,
  isOpen: false,
  activeTab: "model",
  isLoading: false,
  error: null,
  configReloadNotification: null,
  configWatchInterval: null,

  // ─── Actions ──────────────────────────────────────────────────────────

  loadModelsConfig: async (): Promise<void> => {
    set({ isLoading: true, error: null });
    try {
      const overview = await invoke<ModelsOverview>("get_models_config");
      set({ modelsOverview: overview, isLoading: false });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      set({
        error: `Failed to load models config: ${message}`,
        isLoading: false,
      });
    }
  },

  checkAndReloadConfig: async (): Promise<boolean> => {
    try {
      const needsReload = await invoke<boolean>("check_config_reload");
      if (needsReload) {
        const overview = await invoke<ModelsOverview>("reload_model_config");
        set({
          modelsOverview: overview,
          configReloadNotification:
            "Config file changed - reloaded automatically",
        });
        // Auto-clear notification after 5 seconds
        setTimeout(() => {
          get().clearConfigReloadNotification();
        }, 5000);
        return true;
      }
      return false;
    } catch (err: unknown) {
      // Non-critical - just log and continue
      console.warn("Config reload check failed:", err);
      return false;
    }
  },

  startConfigWatch: (): void => {
    const { configWatchInterval } = get();
    if (configWatchInterval) return; // Already watching

    const interval = setInterval(() => {
      get().checkAndReloadConfig();
    }, 3000); // Check every 3 seconds

    set({ configWatchInterval: interval });
  },

  stopConfigWatch: (): void => {
    const { configWatchInterval } = get();
    if (interval) {
      clearInterval(interval);
    }
    set({ configWatchInterval: null });
  },

  clearConfigReloadNotification: (): void => {
    set({ configReloadNotification: null });
  },

  loadServerStatuses: async (): Promise<void> => {
    try {
      const statuses = await invoke<McpServerStatus[]>(
        "get_mcp_servers_status",
      );
      set({ serverStatuses: statuses });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      set({ error: `Failed to load server statuses: ${message}` });
    }
  },

  loadPermissionGrants: async (): Promise<void> => {
    try {
      const grants = await invoke<PermissionGrant[]>("list_permission_grants");
      set({ permissionGrants: grants });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      set({ error: `Failed to load permission grants: ${message}` });
    }
  },

  revokePermission: async (toolName: string): Promise<void> => {
    try {
      await invoke<boolean>("revoke_permission", { toolName });
      // Reload grants after revocation
      const grants = await invoke<PermissionGrant[]>("list_permission_grants");
      set({ permissionGrants: grants });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      set({ error: `Failed to revoke permission: ${message}` });
    }
  },

  loadSamplingConfig: async (): Promise<void> => {
    try {
      const config = await invoke<SamplingConfig>("get_sampling_config");
      set({ samplingConfig: config });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      set({ error: `Failed to load sampling config: ${message}` });
    }
  },

  updateSamplingConfig: async (config: SamplingConfig): Promise<void> => {
    try {
      const updated = await invoke<SamplingConfig>("update_sampling_config", {
        config,
      });
      set({ samplingConfig: updated });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      set({ error: `Failed to update sampling config: ${message}` });
    }
  },

  resetSamplingConfig: async (): Promise<void> => {
    try {
      const config = await invoke<SamplingConfig>("reset_sampling_config");
      set({ samplingConfig: config });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      set({ error: `Failed to reset sampling config: ${message}` });
    }
  },

  togglePanel: (): void => {
    set((state) => ({ isOpen: !state.isOpen }));
  },

  setActiveTab: (tab: "model" | "servers" | "permissions"): void => {
    set({ activeTab: tab });
  },

  clearError: (): void => {
    set({ error: null });
  },
}));
