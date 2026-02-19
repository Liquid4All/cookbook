/**
 * Onboarding wizard Zustand store.
 *
 * Manages the first-run experience: step navigation, hardware detection,
 * llama-server detection (LFM2-24B-A2B), Ollama model detection/pull,
 * GGUF download, folder selection, and server toggles.
 * Persists completion state to localStorage.
 */

import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { documentDir } from "@tauri-apps/api/path";

import type {
  HardwareInfo,
  ModelDownloadProgress,
  OllamaModelInfo,
  OllamaPullProgress,
  PythonEnvStatus,
  PythonEnvProgress,
} from "../types";

/** All 13 MCP server names. */
const ALL_SERVERS: readonly string[] = [
  "filesystem",
  "document",
  "ocr",
  "knowledge",
  "meeting",
  "security",
  "calendar",
  "email",
  "task",
  "data",
  "audit",
  "clipboard",
  "system",
] as const;

/** Total number of onboarding wizard steps. */
const TOTAL_STEPS = 8;

/** localStorage key for persisting onboarding completion. */
const STORAGE_KEY = "localcowork-onboarding-complete";

/** The recommended model key (matches _models/config.yaml active_model). */
export const RECOMMENDED_MODEL_KEY = "lfm2-24b-a2b";

/** The recommended model display name. */
export const RECOMMENDED_MODEL_DISPLAY = "LFM2-24B-A2B";

/** Approximate download / disk size for the recommended model. */
export const RECOMMENDED_MODEL_SIZE = "~16 GB";

/** Benchmark accuracy for the recommended model (67-tool suite). */
export const RECOMMENDED_MODEL_ACCURACY = "80%";

/** The fallback Ollama model name for lower-end hardware. */
export const FALLBACK_OLLAMA_MODEL = "qwen3:30b-a3b";

/** The fallback model display name. */
export const FALLBACK_MODEL_DISPLAY = "Qwen3-30B-A3B";

/** The fallback model approximate download size. */
export const FALLBACK_MODEL_SIZE = "~4 GB";

// ─── State Interface ────────────────────────────────────────────────────────

interface OnboardingState {
  /** Zero-based index of the current wizard step. */
  currentStep: number;
  /** Total number of steps in the wizard. */
  totalSteps: number;
  /** Detected hardware info (null until detection runs). */
  hardware: HardwareInfo | null;
  /** Path or identifier of the selected/detected model. */
  modelPath: string | null;
  /** User-selected working directory. */
  workingDirectory: string;
  /** List of enabled MCP server names. */
  enabledServers: string[];
  /** Whether a model download (GGUF) is in progress. */
  isDownloading: boolean;
  /** Current GGUF download progress (null when not downloading). */
  downloadProgress: ModelDownloadProgress | null;
  /** Whether onboarding has been completed. */
  isComplete: boolean;
  /** Error message from the last failed operation. */
  error: string | null;
  /** Whether hardware detection is in progress. */
  isDetectingHardware: boolean;
  /** Whether llama-server (localhost:8080) is running and reachable. */
  llamaServerAvailable: boolean;
  /** Whether we are currently checking llama-server status. */
  isCheckingLlamaServer: boolean;
  /** Whether Ollama is running and reachable. */
  ollamaAvailable: boolean;
  /** Models currently available in the local Ollama instance. */
  ollamaModels: OllamaModelInfo[];
  /** Whether we are currently checking Ollama status/models. */
  isCheckingOllama: boolean;
  /** Whether an Ollama pull is in progress. */
  isPullingOllama: boolean;
  /** Current Ollama pull progress. */
  ollamaPullProgress: OllamaPullProgress | null;
  /** Python server environment provisioning statuses (Setup step). */
  pythonEnvStatuses: PythonEnvStatus[];
  /** Whether Python env provisioning is in progress. */
  isProvisioningPython: boolean;
  /** Current Python env progress event (for the spinner). */
  pythonEnvProgress: PythonEnvProgress | null;
}

interface OnboardingActions {
  /** Advance to the next step. */
  nextStep: () => void;
  /** Go back to the previous step. */
  prevStep: () => void;
  /** Set detected hardware info. */
  setHardware: (hw: HardwareInfo) => void;
  /** Set the model path/identifier. */
  setModelPath: (path: string) => void;
  /** Set the working directory. */
  setWorkingDirectory: (dir: string) => void;
  /** Toggle a server on or off by name. */
  toggleServer: (name: string) => void;
  /** Update download progress. */
  setDownloadProgress: (progress: ModelDownloadProgress) => void;
  /** Mark onboarding as complete and persist to storage. */
  completeOnboarding: () => void;
  /** Clear the current error. */
  clearError: () => void;
  /** Detect hardware via Tauri IPC. */
  detectHardware: () => Promise<void>;
  /** Start model download (GGUF) via Tauri IPC. Returns an unlisten function. */
  startDownload: (url: string) => Promise<UnlistenFn | null>;
  /** Set downloading state to false (used on completion/error). */
  stopDownloading: () => void;
  /** Check llama-server health (localhost:8080). */
  checkLlamaServer: () => Promise<void>;
  /** Select LFM2 via llama-server as the active model. */
  selectLlamaServer: () => void;
  /** Check Ollama status and list available models. */
  checkOllama: () => Promise<void>;
  /** Pull a model via Ollama. Returns an unlisten function. */
  pullOllamaModel: (modelName: string) => Promise<UnlistenFn | null>;
  /** Select an already-available Ollama model. */
  selectOllamaModel: (modelName: string) => void;
  /** Provision Python server venvs via Tauri IPC. */
  provisionPythonEnvs: () => Promise<void>;
  /** Retry provisioning a single Python server. */
  retryPythonEnv: (serverName: string) => Promise<void>;
  /** Reset onboarding state (for testing). */
  reset: () => void;
}

type OnboardingStore = OnboardingState & OnboardingActions;

/** Check if onboarding was previously completed. */
function wasCompleted(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

// ─── Store ──────────────────────────────────────────────────────────────────

export const useOnboardingStore = create<OnboardingStore>((set, get) => ({
  currentStep: 0,
  totalSteps: TOTAL_STEPS,
  hardware: null,
  modelPath: null,
  workingDirectory: "",
  enabledServers: [...ALL_SERVERS],
  isDownloading: false,
  downloadProgress: null,
  isComplete: wasCompleted(),
  error: null,
  isDetectingHardware: false,
  llamaServerAvailable: false,
  isCheckingLlamaServer: false,
  ollamaAvailable: false,
  ollamaModels: [],
  isCheckingOllama: false,
  isPullingOllama: false,
  ollamaPullProgress: null,
  pythonEnvStatuses: [],
  isProvisioningPython: false,
  pythonEnvProgress: null,

  nextStep: (): void => {
    set((state) => ({
      currentStep: Math.min(state.currentStep + 1, TOTAL_STEPS - 1),
      error: null,
    }));
  },

  prevStep: (): void => {
    set((state) => ({
      currentStep: Math.max(state.currentStep - 1, 0),
      error: null,
    }));
  },

  setHardware: (hw: HardwareInfo): void => {
    set({ hardware: hw, isDetectingHardware: false });
  },

  setModelPath: (path: string): void => {
    set({ modelPath: path });
  },

  setWorkingDirectory: (dir: string): void => {
    set({ workingDirectory: dir });
  },

  toggleServer: (name: string): void => {
    set((state) => {
      const current = state.enabledServers;
      const exists = current.includes(name);
      return {
        enabledServers: exists
          ? current.filter((s) => s !== name)
          : [...current, name],
      };
    });
  },

  setDownloadProgress: (progress: ModelDownloadProgress): void => {
    set({ downloadProgress: progress });
  },

  completeOnboarding: (): void => {
    try {
      localStorage.setItem(STORAGE_KEY, "true");
    } catch {
      // Storage may be unavailable in some contexts
    }
    set({ isComplete: true });
  },

  clearError: (): void => {
    set({ error: null });
  },

  detectHardware: async (): Promise<void> => {
    set({ isDetectingHardware: true, error: null });
    try {
      const hw = await invoke<HardwareInfo>("detect_hardware");
      // Resolve platform-correct documents directory if not already set
      const state = get();
      let dir = state.workingDirectory;
      if (dir.length === 0) {
        try {
          dir = await documentDir();
        } catch {
          dir = "~/Documents";
        }
      }
      set({ hardware: hw, isDetectingHardware: false, workingDirectory: dir });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      set({
        error: `Hardware detection failed: ${message}`,
        isDetectingHardware: false,
      });
    }
  },

  startDownload: async (url: string): Promise<UnlistenFn | null> => {
    set({ isDownloading: true, downloadProgress: null, error: null });

    // Listen for progress events
    let unlisten: UnlistenFn | null = null;
    try {
      unlisten = await listen<ModelDownloadProgress>(
        "model-download-progress",
        (event) => {
          get().setDownloadProgress(event.payload);
        },
      );
    } catch {
      // Event listening may fail in test environments
    }

    try {
      const modelDir = await invoke<string>("get_model_dir");
      const result = await invoke<{ success: boolean; modelPath: string }>(
        "download_model",
        { url, targetDir: modelDir },
      );
      if (result.success) {
        set({
          modelPath: result.modelPath,
          isDownloading: false,
        });
      } else {
        set({ error: "Download failed", isDownloading: false });
      }
    } catch (err: unknown) {
      const raw = err instanceof Error ? err.message : String(err);
      let error: string;
      if (raw.includes("404")) {
        error =
          "This model file is no longer available at the expected URL. " +
          "Use Ollama to install the recommended model instead, or browse for a local file.";
      } else if (raw.includes("status:")) {
        error = `Download failed (${raw}). Check your internet connection and try again.`;
      } else {
        error = `Download failed: ${raw}`;
      }
      set({ error, isDownloading: false });
    }

    return unlisten;
  },

  stopDownloading: (): void => {
    set({ isDownloading: false });
  },

  checkLlamaServer: async (): Promise<void> => {
    set({ isCheckingLlamaServer: true, error: null });
    try {
      const available = await invoke<boolean>("check_llama_server_status");
      set({ llamaServerAvailable: available, isCheckingLlamaServer: false });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      set({
        llamaServerAvailable: false,
        isCheckingLlamaServer: false,
        error: `llama-server check failed: ${message}`,
      });
    }
  },

  selectLlamaServer: (): void => {
    set({ modelPath: "llama-server:lfm2-24b-a2b" });
  },

  checkOllama: async (): Promise<void> => {
    set({ isCheckingOllama: true, error: null });
    try {
      const available = await invoke<boolean>("check_ollama_status");
      if (available) {
        const models = await invoke<OllamaModelInfo[]>("list_ollama_models");
        set({
          ollamaAvailable: true,
          ollamaModels: models,
          isCheckingOllama: false,
        });
      } else {
        set({
          ollamaAvailable: false,
          ollamaModels: [],
          isCheckingOllama: false,
        });
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      set({
        ollamaAvailable: false,
        ollamaModels: [],
        isCheckingOllama: false,
        error: `Ollama check failed: ${message}`,
      });
    }
  },

  pullOllamaModel: async (
    modelName: string,
  ): Promise<UnlistenFn | null> => {
    set({ isPullingOllama: true, ollamaPullProgress: null, error: null });

    let unlisten: UnlistenFn | null = null;
    try {
      unlisten = await listen<OllamaPullProgress>(
        "ollama-pull-progress",
        (event) => {
          set({ ollamaPullProgress: event.payload });
        },
      );
    } catch {
      // Event listening may fail in test environments
    }

    try {
      await invoke<boolean>("pull_ollama_model", { modelName });
      // After pull completes, re-check models and auto-select
      const models = await invoke<OllamaModelInfo[]>("list_ollama_models");
      set({
        ollamaModels: models,
        isPullingOllama: false,
        modelPath: `ollama:${modelName}`,
      });
    } catch (err: unknown) {
      const raw = err instanceof Error ? err.message : String(err);
      const hint = raw.toLowerCase().includes("cannot reach")
        ? " Check that Ollama is still running (ollama serve)."
        : "";
      set({
        error: `Model pull failed: ${raw}.${hint}`,
        isPullingOllama: false,
      });
    }

    return unlisten;
  },

  selectOllamaModel: (modelName: string): void => {
    set({ modelPath: `ollama:${modelName}` });
  },

  provisionPythonEnvs: async (): Promise<void> => {
    set({ isProvisioningPython: true, pythonEnvStatuses: [], error: null });

    // Listen for per-server progress events
    let unlisten: UnlistenFn | null = null;
    try {
      unlisten = await listen<PythonEnvProgress>("python-env-progress", (event) => {
        set({ pythonEnvProgress: event.payload });
      });
    } catch {
      // Event listening may fail in test environments
    }

    try {
      const results = await invoke<PythonEnvStatus[]>("ensure_all_python_envs");
      set({ pythonEnvStatuses: results, isProvisioningPython: false, pythonEnvProgress: null });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      set({ error: `Python setup failed: ${msg}`, isProvisioningPython: false });
    }

    if (unlisten) {
      unlisten();
    }
  },

  retryPythonEnv: async (serverName: string): Promise<void> => {
    // Update the specific server's status to show it's retrying
    set((state) => ({
      pythonEnvStatuses: state.pythonEnvStatuses.map((s) =>
        s.server === serverName ? { ...s, ready: false, error: null } : s,
      ),
      error: null,
    }));

    try {
      const result = await invoke<PythonEnvStatus>("ensure_python_server_env", { serverName });
      set((state) => ({
        pythonEnvStatuses: state.pythonEnvStatuses.map((s) =>
          s.server === serverName ? result : s,
        ),
      }));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      set((state) => ({
        pythonEnvStatuses: state.pythonEnvStatuses.map((s) =>
          s.server === serverName ? { ...s, ready: false, error: msg } : s,
        ),
      }));
    }
  },

  reset: (): void => {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore
    }
    set({
      currentStep: 0,
      hardware: null,
      modelPath: null,
      workingDirectory: "",
      enabledServers: [...ALL_SERVERS],
      isDownloading: false,
      downloadProgress: null,
      isComplete: false,
      error: null,
      isDetectingHardware: false,
      llamaServerAvailable: false,
      isCheckingLlamaServer: false,
      ollamaAvailable: false,
      ollamaModels: [],
      isCheckingOllama: false,
      isPullingOllama: false,
      ollamaPullProgress: null,
      pythonEnvStatuses: [],
      isProvisioningPython: false,
      pythonEnvProgress: null,
    });
  },
}));
