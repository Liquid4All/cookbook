/**
 * ModelTab — model configuration and inference parameters in the settings panel.
 *
 * Shows the active model, available models, fallback chain,
 * and editable sampling hyperparameters (temperature, top_p).
 */

import { useCallback, useState } from "react";

import type { ModelsOverview, SamplingConfig } from "../../types";

interface ModelTabProps {
  readonly overview: ModelsOverview;
  readonly samplingConfig: SamplingConfig | null;
  readonly onUpdateSampling: (config: SamplingConfig) => Promise<void>;
  readonly onResetSampling: () => Promise<void>;
}

/** Badge for a capability. */
function CapabilityBadge({
  label,
}: {
  readonly label: string;
}): React.JSX.Element {
  return <span className="settings-capability-badge">{label}</span>;
}

/** Slider row for a single sampling parameter. */
function SamplingSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  readonly label: string;
  readonly value: number;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly onChange: (value: number) => void;
}): React.JSX.Element {
  return (
    <div className="settings-sampling-row">
      <div className="settings-sampling-header">
        <span className="settings-detail-label">{label}</span>
        <span className="settings-sampling-value">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        className="settings-sampling-slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => {
          onChange(parseFloat(e.target.value));
        }}
      />
    </div>
  );
}

export function ModelTab({
  overview,
  samplingConfig,
  onUpdateSampling,
  onResetSampling,
}: ModelTabProps): React.JSX.Element {
  const activeModel = overview.models.find(
    (m) => m.key === overview.activeModel,
  );

  // Local state for slider values (committed on change).
  const [localConfig, setLocalConfig] = useState<SamplingConfig | null>(null);
  const config = localConfig ?? samplingConfig;

  const handleSliderChange = useCallback(
    (field: keyof SamplingConfig, value: number) => {
      if (config == null) return;
      const updated: SamplingConfig = { ...config, [field]: value };
      setLocalConfig(updated);
      void onUpdateSampling(updated);
    },
    [config, onUpdateSampling],
  );

  const handleReset = useCallback(() => {
    setLocalConfig(null);
    void onResetSampling();
  }, [onResetSampling]);

  return (
    <div className="settings-tab-content">
      {/* Active model section */}
      <div className="settings-section">
        <h3 className="settings-section-title">Active Model</h3>
        {activeModel != null ? (
          <div className="settings-model-card settings-model-active">
            <div className="settings-model-header">
              <span className="settings-model-name">
                {activeModel.displayName}
              </span>
              <span className="settings-model-badge active">Active</span>
            </div>
            <div className="settings-model-details">
              <div className="settings-detail-row">
                <span className="settings-detail-label">Runtime</span>
                <span className="settings-detail-value">
                  {activeModel.runtime}
                </span>
              </div>
              <div className="settings-detail-row">
                <span className="settings-detail-label">Endpoint</span>
                <span className="settings-detail-value settings-mono">
                  {activeModel.baseUrl}
                </span>
              </div>
              <div className="settings-detail-row">
                <span className="settings-detail-label">Context Window</span>
                <span className="settings-detail-value">
                  {activeModel.contextWindow.toLocaleString()} tokens
                </span>
              </div>
              <div className="settings-detail-row">
                <span className="settings-detail-label">Temperature</span>
                <span className="settings-detail-value">
                  {activeModel.temperature.toFixed(2)}
                </span>
              </div>
              <div className="settings-detail-row">
                <span className="settings-detail-label">Max Tokens</span>
                <span className="settings-detail-value">
                  {activeModel.maxTokens.toLocaleString()}
                </span>
              </div>
              {activeModel.estimatedVramGb != null && (
                <div className="settings-detail-row">
                  <span className="settings-detail-label">Est. VRAM</span>
                  <span className="settings-detail-value">
                    {activeModel.estimatedVramGb} GB
                  </span>
                </div>
              )}
              <div className="settings-detail-row">
                <span className="settings-detail-label">Capabilities</span>
                <span className="settings-detail-value">
                  {activeModel.capabilities.map((cap) => (
                    <CapabilityBadge key={cap} label={cap} />
                  ))}
                </span>
              </div>
            </div>
          </div>
        ) : (
          <p className="settings-muted">No active model configured.</p>
        )}
      </div>

      {/* Inference parameters */}
      {config != null && (
        <div className="settings-section">
          <div className="settings-section-header">
            <h3 className="settings-section-title">Inference Parameters</h3>
            <button
              className="settings-reset-btn"
              onClick={handleReset}
              type="button"
            >
              Reset
            </button>
          </div>
          <p className="settings-section-desc">
            Tool-calling turns use low temperature for deterministic tool
            selection. Conversational turns use higher temperature for natural
            language.
          </p>
          <div className="settings-sampling-group">
            <span className="settings-sampling-group-label">
              Tool-Calling Turns
            </span>
            <SamplingSlider
              label="Temperature"
              value={config.toolTemperature}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => {
                handleSliderChange("toolTemperature", v);
              }}
            />
            <SamplingSlider
              label="Top-P"
              value={config.toolTopP}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => {
                handleSliderChange("toolTopP", v);
              }}
            />
          </div>
          <div className="settings-sampling-group">
            <span className="settings-sampling-group-label">
              Conversational Turns
            </span>
            <SamplingSlider
              label="Temperature"
              value={config.conversationalTemperature}
              min={0}
              max={2}
              step={0.05}
              onChange={(v) => {
                handleSliderChange("conversationalTemperature", v);
              }}
            />
            <SamplingSlider
              label="Top-P"
              value={config.conversationalTopP}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => {
                handleSliderChange("conversationalTopP", v);
              }}
            />
          </div>
        </div>
      )}

      {/* Fallback chain */}
      <div className="settings-section">
        <h3 className="settings-section-title">Fallback Chain</h3>
        <div className="settings-fallback-chain">
          {overview.fallbackChain.map((key, index) => (
            <span key={key} className="settings-fallback-item">
              {index > 0 && (
                <span className="settings-fallback-arrow">&rarr;</span>
              )}
              <span
                className={`settings-fallback-name ${
                  key === overview.activeModel ? "active" : ""
                }`}
              >
                {key}
              </span>
            </span>
          ))}
        </div>
      </div>

      {/* All models catalog */}
      <div className="settings-section">
        <h3 className="settings-section-title">Available Models</h3>
        <div className="settings-model-list">
          {overview.models.map((model) => (
            <div
              key={model.key}
              className={`settings-model-card ${
                model.key === overview.activeModel
                  ? "settings-model-active"
                  : ""
              }`}
            >
              <div className="settings-model-header">
                <span className="settings-model-name">
                  {model.displayName}
                </span>
                <span className="settings-model-runtime">{model.runtime}</span>
              </div>
              <div className="settings-model-meta">
                {model.estimatedVramGb != null && (
                  <span>{model.estimatedVramGb} GB VRAM</span>
                )}
                <span>{model.contextWindow.toLocaleString()} ctx</span>
                <span>{model.toolCallFormat}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
