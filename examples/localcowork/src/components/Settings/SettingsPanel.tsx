/**
 * SettingsPanel — slide-out panel for model, server, and permissions config.
 *
 * Contains tabs for model settings, MCP server status, and permission grants.
 * Opens/closes via the settings button in the header.
 */

import { useCallback, useEffect } from "react";

import { useSettingsStore } from "../../stores/settingsStore";
import { ModelTab } from "./ModelTab";
import { PermissionsTab } from "./PermissionsTab";
import { ServersTab } from "./ServersTab";

export function SettingsPanel(): React.JSX.Element {
  const {
    modelsOverview,
    serverStatuses,
    permissionGrants,
    samplingConfig,
    isOpen,
    activeTab,
    isLoading,
    error,
    loadModelsConfig,
    loadServerStatuses,
    loadPermissionGrants,
    loadSamplingConfig,
    updateSamplingConfig,
    resetSamplingConfig,
    revokePermission,
    togglePanel,
    setActiveTab,
    clearError,
  } = useSettingsStore();

  // Load data when panel opens
  useEffect(() => {
    if (isOpen) {
      void loadModelsConfig();
      void loadServerStatuses();
      void loadPermissionGrants();
      void loadSamplingConfig();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  const handleClose = useCallback(() => {
    togglePanel();
  }, [togglePanel]);

  const handleTabModel = useCallback(() => {
    setActiveTab("model");
  }, [setActiveTab]);

  const handleTabServers = useCallback(() => {
    setActiveTab("servers");
  }, [setActiveTab]);

  const handleTabPermissions = useCallback(() => {
    setActiveTab("permissions");
  }, [setActiveTab]);

  const handleRefreshServers = useCallback(() => {
    void loadServerStatuses();
  }, [loadServerStatuses]);

  const handleRevokePermission = useCallback(
    (toolName: string) => {
      void revokePermission(toolName);
    },
    [revokePermission],
  );

  if (!isOpen) {
    return <></>;
  }

  return (
    <div className="settings-overlay" onClick={handleClose} role="presentation">
      <aside
        className="settings-panel"
        onClick={(e) => {
          e.stopPropagation();
        }}
        role="dialog"
        aria-label="Settings"
      >
        {/* Header */}
        <div className="settings-header">
          <h2 className="settings-title">Settings</h2>
          <button
            className="settings-close-btn"
            onClick={handleClose}
            type="button"
            aria-label="Close settings"
          >
            &times;
          </button>
        </div>

        {/* Error banner */}
        {error != null && (
          <div className="settings-error">
            <span className="settings-error-text">{error}</span>
            <button
              className="settings-error-dismiss"
              onClick={clearError}
              type="button"
            >
              &times;
            </button>
          </div>
        )}

        {/* Tabs */}
        <div className="settings-tabs">
          <button
            className={`settings-tab ${activeTab === "model" ? "active" : ""}`}
            onClick={handleTabModel}
            type="button"
          >
            Model
          </button>
          <button
            className={`settings-tab ${activeTab === "servers" ? "active" : ""}`}
            onClick={handleTabServers}
            type="button"
          >
            Servers
          </button>
          <button
            className={`settings-tab ${activeTab === "permissions" ? "active" : ""}`}
            onClick={handleTabPermissions}
            type="button"
          >
            Permissions
          </button>
        </div>

        {/* Tab content */}
        <div className="settings-body">
          {isLoading ? (
            <div className="settings-loading">Loading...</div>
          ) : (
            <>
              {activeTab === "model" && modelsOverview != null && (
                <ModelTab
                  overview={modelsOverview}
                  samplingConfig={samplingConfig}
                  onUpdateSampling={updateSamplingConfig}
                  onResetSampling={resetSamplingConfig}
                />
              )}
              {activeTab === "servers" && (
                <ServersTab
                  statuses={serverStatuses}
                  onRefresh={handleRefreshServers}
                />
              )}
              {activeTab === "permissions" && (
                <PermissionsTab
                  grants={permissionGrants}
                  onRevoke={handleRevokePermission}
                />
              )}
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
