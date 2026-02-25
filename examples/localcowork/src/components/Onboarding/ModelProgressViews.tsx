/**
 * Progress and completion sub-components for the Model setup step.
 *
 * - ModelReady: shows the confirmed model with a checkmark and accuracy badge
 * - OllamaPullProgressView: progress bar for Ollama model pull
 * - GgufDownloadProgressView: progress bar for direct GGUF file download
 */

import type { ModelDownloadProgress, OllamaPullProgress } from "../../types";
import { RECOMMENDED_MODEL_ACCURACY } from "../../stores/onboardingStore";
import { formatBytes, formatEta, isLfm2Model } from "./modelStepUtils";

/** Shows the confirmed model with a checkmark. */
export function ModelReady({
  modelPath,
}: {
  readonly modelPath: string;
}): React.JSX.Element {
  const isLfm2 = isLfm2Model(modelPath);

  let displayPath: string;
  if (modelPath.startsWith("llama-server:")) {
    displayPath = `llama-server \u2014 ${modelPath.slice(13)}`;
  } else if (modelPath.startsWith("ollama:")) {
    displayPath = `Ollama \u2014 ${modelPath.slice(7)}`;
  } else {
    displayPath = modelPath;
  }

  return (
    <div className="model-complete">
      <div className="model-complete-icon">&#10003;</div>
      <div className="model-complete-text">
        <span className="model-complete-label">Model ready</span>
        <span className="model-complete-path">{displayPath}</span>
        {isLfm2 && (
          <span className="model-accuracy-badge">
            {RECOMMENDED_MODEL_ACCURACY} tool-calling accuracy
          </span>
        )}
      </div>
    </div>
  );
}

/** Ollama pull progress bar. */
export function OllamaPullProgressView({
  progress,
}: {
  readonly progress: OllamaPullProgress | null;
}): React.JSX.Element {
  return (
    <div className="model-progress">
      <div className="progress-bar-container">
        <div
          className="progress-bar-fill"
          style={{ width: `${progress?.percent ?? 0}%` }}
        />
      </div>
      <div className="progress-stats">
        <span>{progress?.status ?? "Starting..."}</span>
        <span>
          {progress && progress.total > 0
            ? `${formatBytes(progress.completed)} / ${formatBytes(progress.total)}`
            : ""}
        </span>
      </div>
      <div className="progress-percent">
        {progress ? `${progress.percent.toFixed(1)}%` : "0%"}
      </div>
    </div>
  );
}

/** GGUF file download progress bar. */
export function GgufDownloadProgressView({
  progress,
}: {
  readonly progress: ModelDownloadProgress | null;
}): React.JSX.Element {
  return (
    <div className="model-progress">
      <div className="progress-bar-container">
        <div
          className="progress-bar-fill"
          style={{ width: `${progress?.percent ?? 0}%` }}
        />
      </div>
      <div className="progress-stats">
        <span>
          {progress
            ? `${formatBytes(progress.bytesDownloaded)} / ${formatBytes(progress.bytesTotal)}`
            : "Starting..."}
        </span>
        <span>
          {progress ? `${progress.speedMbps.toFixed(1)} MB/s` : ""}
        </span>
        <span>
          {progress ? `ETA: ${formatEta(progress.etaSeconds)}` : ""}
        </span>
      </div>
      <div className="progress-percent">
        {progress ? `${progress.percent.toFixed(1)}%` : "0%"}
      </div>
    </div>
  );
}
