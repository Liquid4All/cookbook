/**
 * ModelTab — model configuration display in the settings panel.
 *
 * Shows the active model, available models, fallback chain,
 * and inference parameters.
 */

import type { ModelsOverview } from "../../types";

interface ModelTabProps {
  readonly overview: ModelsOverview;
}

/** Badge for a capability. */
function CapabilityBadge({
  label,
}: {
  readonly label: string;
}): React.JSX.Element {
  return <span className="settings-capability-badge">{label}</span>;
}

export function ModelTab({ overview }: ModelTabProps): React.JSX.Element {
  const activeModel = overview.models.find(
    (m) => m.key === overview.activeModel,
  );

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
                  {activeModel.temperature}
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
