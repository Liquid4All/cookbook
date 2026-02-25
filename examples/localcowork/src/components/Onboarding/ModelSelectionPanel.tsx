/**
 * Model selection panel with three option cards:
 * 1. LFM2-24B-A2B via llama-server (recommended)
 * 2. Use Ollama instead (secondary)
 * 3. Use a local model file (Browse Files)
 */

import type { OllamaModelInfo } from "../../types";
import {
  RECOMMENDED_MODEL_DISPLAY,
  RECOMMENDED_MODEL_SIZE,
  RECOMMENDED_MODEL_ACCURACY,
  FALLBACK_MODEL_DISPLAY,
  FALLBACK_MODEL_SIZE,
} from "../../stores/onboardingStore";
import { formatBytes, getOtherModels } from "./modelStepUtils";
import { OllamaInstallGuide } from "./OllamaInstallGuide";
import { LlamaServerSetupGuide } from "./LlamaServerSetupGuide";

interface ModelSelectionPanelProps {
  readonly llamaServerAvailable: boolean;
  readonly isCheckingLlamaServer: boolean;
  readonly ollamaAvailable: boolean;
  readonly ollamaModels: readonly OllamaModelInfo[];
  readonly isCheckingOllama: boolean;
  readonly onSelectLlamaServer: () => void;
  readonly onRefreshLlamaServer: () => void;
  readonly onPullFallback: () => void;
  readonly onSelectOllamaModel: (model: OllamaModelInfo) => void;
  readonly onUseLocalFile: () => void;
  readonly onRefreshOllama: () => void;
}

/** The main model selection panel with three option cards. */
export function ModelSelectionPanel({
  llamaServerAvailable,
  isCheckingLlamaServer,
  ollamaAvailable,
  ollamaModels,
  isCheckingOllama,
  onSelectLlamaServer,
  onRefreshLlamaServer,
  onPullFallback,
  onSelectOllamaModel,
  onUseLocalFile,
  onRefreshOllama,
}: ModelSelectionPanelProps): React.JSX.Element {
  const otherModels = getOtherModels(ollamaModels);

  return (
    <div className="model-selection-panel">
      {/* Card 1: LFM2-24B-A2B via llama-server (primary) */}
      <div className="model-option model-option-primary">
        <div className="model-option-header">
          <span className="model-option-badge">
            Recommended &mdash; {RECOMMENDED_MODEL_ACCURACY} accuracy
          </span>
          <h3 className="model-option-title">
            {RECOMMENDED_MODEL_DISPLAY} via llama-server
          </h3>
        </div>

        {isCheckingLlamaServer ? (
          <p className="model-option-status">Checking llama-server...</p>
        ) : llamaServerAvailable ? (
          <LlamaServerAvailableBody onSelect={onSelectLlamaServer} />
        ) : (
          <LlamaServerSetupGuide onRefresh={onRefreshLlamaServer} />
        )}
      </div>

      {/* Card 2: Ollama (secondary) */}
      <div className="model-option model-option-secondary">
        <h3 className="model-option-title">Use Ollama Instead</h3>
        <p className="model-option-hint">
          {FALLBACK_MODEL_DISPLAY} (MoE) &mdash; only {FALLBACK_MODEL_SIZE},
          works on most hardware. Or pick any model already in Ollama.
        </p>

        {isCheckingOllama ? (
          <p className="model-option-status">Checking Ollama...</p>
        ) : ollamaAvailable ? (
          <OllamaAvailableBody
            ollamaModels={ollamaModels}
            otherModels={otherModels}
            onPullFallback={onPullFallback}
            onSelectOllamaModel={onSelectOllamaModel}
          />
        ) : (
          <OllamaInstallGuide onRefresh={onRefreshOllama} />
        )}
      </div>

      {/* Card 3: Browse for local file */}
      <div className="model-option model-option-secondary">
        <h3 className="model-option-title">Use a Local Model File</h3>
        <p className="model-option-hint">
          Select a GGUF, safetensors, or bin file from your machine.
        </p>
        <button
          className="onboarding-btn secondary"
          onClick={onUseLocalFile}
          type="button"
        >
          Browse Files
        </button>
      </div>
    </div>
  );
}

// ---- Sub-components ----

/** Body shown when llama-server is running. */
function LlamaServerAvailableBody({
  onSelect,
}: {
  readonly onSelect: () => void;
}): React.JSX.Element {
  return (
    <div className="model-option-body">
      <div className="ollama-status ollama-status-ok">
        <span className="ollama-status-dot" />
        llama-server is running
      </div>
      <p className="model-option-hint">
        {RECOMMENDED_MODEL_DISPLAY} ({RECOMMENDED_MODEL_SIZE}) &mdash;{" "}
        {RECOMMENDED_MODEL_ACCURACY} tool-calling accuracy across all 67 tools.
      </p>
      <button
        className="onboarding-btn primary"
        onClick={onSelect}
        type="button"
      >
        Use {RECOMMENDED_MODEL_DISPLAY}
      </button>
    </div>
  );
}

/** Body shown when Ollama is running and available. */
function OllamaAvailableBody({
  ollamaModels,
  otherModels,
  onPullFallback,
  onSelectOllamaModel,
}: {
  readonly ollamaModels: readonly OllamaModelInfo[];
  readonly otherModels: OllamaModelInfo[];
  readonly onPullFallback: () => void;
  readonly onSelectOllamaModel: (model: OllamaModelInfo) => void;
}): React.JSX.Element {
  return (
    <div className="model-option-body">
      <div className="ollama-status ollama-status-ok">
        <span className="ollama-status-dot" />
        Ollama is running
      </div>

      <button
        className="onboarding-btn secondary model-download-btn"
        onClick={onPullFallback}
        type="button"
      >
        Install {FALLBACK_MODEL_DISPLAY} ({FALLBACK_MODEL_SIZE})
      </button>

      {otherModels.length > 0 ? (
        <div className="ollama-models-list">
          <p className="ollama-models-heading">Models in Ollama:</p>
          {ollamaModels.map((model) => (
            <button
              key={model.name}
              className="ollama-model-item"
              onClick={() => onSelectOllamaModel(model)}
              type="button"
            >
              <span className="ollama-model-name">{model.name}</span>
              <span className="ollama-model-meta">
                {model.parameterSize ? `${model.parameterSize}` : ""}
                {model.parameterSize && model.sizeBytes ? " \u00b7 " : ""}
                {model.sizeBytes ? formatBytes(model.sizeBytes) : ""}
              </span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
