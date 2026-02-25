/**
 * Settings panel types for model config and MCP server status.
 *
 * These mirror the Rust inference config and MCP client status
 * exposed via Tauri IPC commands.
 */

/** Configuration for a single model. */
export interface ModelConfig {
  readonly key: string;
  readonly displayName: string;
  readonly runtime: string;
  readonly baseUrl: string;
  readonly contextWindow: number;
  readonly temperature: number;
  readonly maxTokens: number;
  readonly estimatedVramGb: number | null;
  readonly capabilities: readonly string[];
  readonly toolCallFormat: string;
}

/** Overall model configuration state. */
export interface ModelsOverview {
  readonly activeModel: string;
  readonly models: readonly ModelConfig[];
  readonly fallbackChain: readonly string[];
}

/** Status of a single MCP server. */
export interface McpServerStatus {
  readonly name: string;
  readonly status: "initialized" | "starting" | "failed" | "unavailable";
  readonly toolCount: number;
  readonly lastCheck: string;
  readonly error?: string;
}

/** A persistent permission grant for a tool. */
export interface PermissionGrant {
  readonly toolName: string;
  readonly scope: "session" | "always";
  readonly grantedAt: string;
}

/** Runtime sampling hyperparameters for the agent loop. */
export interface SamplingConfig {
  readonly toolTemperature: number;
  readonly toolTopP: number;
  readonly conversationalTemperature: number;
  readonly conversationalTopP: number;
}
