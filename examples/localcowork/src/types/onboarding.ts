/**
 * Onboarding wizard types for hardware detection, model download,
 * and first-run configuration.
 *
 * These mirror the Rust structs in commands/hardware.rs and
 * commands/model_download.rs.
 */

/** GPU information detected on the system. */
export interface GpuInfo {
  readonly vendor: string;
  readonly model: string;
  readonly vramGb: number | null;
}

/** Complete hardware profile for the local machine. */
export interface HardwareInfo {
  readonly cpuVendor: string;
  readonly cpuModel: string;
  readonly cpuCores: number;
  readonly cpuThreads: number;
  readonly ramTotalGb: number;
  readonly ramAvailableGb: number;
  readonly osName: string;
  readonly osVersion: string;
  readonly arch: string;
  readonly gpu: GpuInfo | null;
  readonly recommendedRuntime: string;
  readonly recommendedQuantization: string;
}

/** Progress update emitted during model download. */
export interface ModelDownloadProgress {
  readonly bytesDownloaded: number;
  readonly bytesTotal: number;
  readonly percent: number;
  readonly speedMbps: number;
  readonly etaSeconds: number;
}

/** Result of a completed model download. */
export interface ModelDownloadResult {
  readonly success: boolean;
  readonly modelPath: string;
  readonly sha256: string;
  readonly sizeBytes: number;
}

/** Information about a model available in the local Ollama instance. */
export interface OllamaModelInfo {
  readonly name: string;
  readonly sizeBytes: number;
  readonly parameterSize: string;
  readonly quantizationLevel: string;
}

/** Progress update emitted while pulling an Ollama model. */
export interface OllamaPullProgress {
  readonly status: string;
  readonly total: number;
  readonly completed: number;
  readonly percent: number;
}

/** Names of all available onboarding steps. */
export type OnboardingStepName =
  | "welcome"
  | "hardware"
  | "model"
  | "folder"
  | "servers"
  | "setup"
  | "demo"
  | "ready";

/** Status of a single Python server's environment provisioning. */
export interface PythonEnvStatus {
  readonly server: string;
  readonly ready: boolean;
  readonly error: string | null;
}

/** Progress event emitted during Python venv provisioning. */
export interface PythonEnvProgress {
  readonly server: string;
  readonly stage: "checking" | "creating_venv" | "installing_deps" | "done" | "failed";
  readonly message: string;
}

/** An MCP server entry shown in the servers step. */
export interface McpServerEntry {
  readonly name: string;
  readonly description: string;
  readonly language: "TypeScript" | "Python";
}

/** A demo prompt shown in the demo step. */
export interface DemoPrompt {
  readonly prompt: string;
  readonly description: string;
  readonly tools: readonly string[];
}
