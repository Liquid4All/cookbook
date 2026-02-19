/**
 * Utility functions for the Model setup step.
 */

import type { OllamaModelInfo } from "../../types";
import {
  RECOMMENDED_MODEL_KEY,
  FALLBACK_OLLAMA_MODEL,
} from "../../stores/onboardingStore";

/** Format bytes into a human-readable string (e.g., "14.2 GB"). */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const value = bytes / Math.pow(k, i);
  return `${value.toFixed(1)} ${units[i]}`;
}

/** Format seconds into MM:SS display. */
export function formatEta(seconds: number): string {
  if (seconds <= 0) return "--:--";
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** Check if a model path refers to the LFM2 model via llama-server. */
export function isLfm2Model(path: string): boolean {
  return path.includes(RECOMMENDED_MODEL_KEY) || path.startsWith("llama-server:");
}

/** Check if an Ollama model name matches the fallback model. */
export function isFallbackModel(name: string): boolean {
  return (
    name === FALLBACK_OLLAMA_MODEL ||
    name.startsWith(`${FALLBACK_OLLAMA_MODEL}-`) ||
    name === `${FALLBACK_OLLAMA_MODEL}:latest`
  );
}

/** Filter to non-fallback Ollama models. */
export function getOtherModels(
  models: readonly OllamaModelInfo[],
): OllamaModelInfo[] {
  return models.filter((m) => !isFallbackModel(m.name));
}
