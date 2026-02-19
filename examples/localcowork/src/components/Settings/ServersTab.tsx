/**
 * ServersTab — MCP server status display in the settings panel.
 *
 * Shows the health status of all MCP servers with tool counts,
 * last check timestamps, error messages, and a Repair button
 * for failed Python servers.
 */

import { useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

import type { McpServerStatus } from "../../types";
import type { PythonEnvStatus } from "../../types";

/** Python MCP server names that can be repaired. */
const PYTHON_SERVER_NAMES = new Set([
  "document", "ocr", "knowledge", "meeting", "security", "screenshot-pipeline",
]);

interface ServersTabProps {
  readonly statuses: readonly McpServerStatus[];
  readonly onRefresh: () => void;
}

/** Status indicator dot and label. */
function StatusIndicator({
  status,
}: {
  readonly status: string;
}): React.JSX.Element {
  let dotClass = "status-dot-unavailable";
  let label = status;

  switch (status) {
    case "initialized":
      dotClass = "status-dot-initialized";
      label = "Running";
      break;
    case "starting":
      dotClass = "status-dot-starting";
      label = "Starting";
      break;
    case "failed":
      dotClass = "status-dot-failed";
      label = "Failed";
      break;
    case "unavailable":
      dotClass = "status-dot-unavailable";
      label = "Unavailable";
      break;
  }

  return (
    <span className="server-status-indicator">
      <span className={`server-status-dot ${dotClass}`} />
      <span className="server-status-label">{label}</span>
    </span>
  );
}

/** Format a timestamp for display. */
function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString();
  } catch {
    return iso;
  }
}

export function ServersTab({
  statuses,
  onRefresh,
}: ServersTabProps): React.JSX.Element {
  const [repairingServer, setRepairingServer] = useState<string | null>(null);
  const [repairResult, setRepairResult] = useState<string | null>(null);

  const handleRefresh = useCallback(() => {
    onRefresh();
  }, [onRefresh]);

  const handleRepair = useCallback(async (serverName: string) => {
    setRepairingServer(serverName);
    setRepairResult(null);
    try {
      const result = await invoke<PythonEnvStatus>("ensure_python_server_env", { serverName });
      if (result.ready) {
        setRepairResult(`${serverName}: repaired. Restart the app to use it.`);
      } else {
        setRepairResult(`${serverName}: ${result.error ?? "unknown error"}`);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setRepairResult(`${serverName}: ${msg}`);
    } finally {
      setRepairingServer(null);
    }
  }, []);

  const initializedCount = statuses.filter(
    (s) => s.status === "initialized",
  ).length;

  return (
    <div className="settings-tab-content">
      <div className="settings-section">
        <div className="settings-section-header">
          <h3 className="settings-section-title">MCP Servers</h3>
          <button
            className="settings-refresh-btn"
            onClick={handleRefresh}
            type="button"
          >
            &#8635; Refresh
          </button>
        </div>
        <p className="settings-muted">
          {initializedCount} of {statuses.length} servers running
        </p>
      </div>

      {repairResult != null && (
        <div className="settings-repair-result">{repairResult}</div>
      )}

      <div className="settings-server-list">
        {statuses.map((server) => {
          const isPythonServer = PYTHON_SERVER_NAMES.has(server.name);
          const canRepair = isPythonServer &&
            (server.status === "failed" || server.status === "unavailable");
          const isRepairing = repairingServer === server.name;

          return (
            <div key={server.name} className="settings-server-card">
              <div className="settings-server-header">
                <span className="settings-server-name">{server.name}</span>
                <StatusIndicator status={server.status} />
                {canRepair && (
                  <button
                    className="server-repair-btn"
                    onClick={() => void handleRepair(server.name)}
                    disabled={isRepairing}
                    type="button"
                  >
                    {isRepairing ? "Repairing..." : "Repair"}
                  </button>
                )}
              </div>
              <div className="settings-server-meta">
                <span className="settings-server-tools">
                  {server.toolCount} tool{server.toolCount !== 1 ? "s" : ""}
                </span>
                <span className="settings-server-check">
                  Last check: {formatTimestamp(server.lastCheck)}
                </span>
              </div>
              {server.error != null && (
                <div className="settings-server-error">{server.error}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
