/**
 * Setup step — provisions Python MCP server virtual environments.
 *
 * Shown after the Servers step and before Demo. Automatically calls
 * `ensure_all_python_envs` on mount. Displays per-server progress
 * and allows retrying individual failures.
 *
 * Skips automatically if no Python servers are enabled.
 */
import { useCallback, useEffect, useRef } from "react";
import { useOnboardingStore } from "../../stores/onboardingStore";

/** Python MCP server names (must be enabled in the Servers step to provision). */
const PYTHON_SERVERS: ReadonlyArray<{ name: string; label: string }> = [
  { name: "document", label: "Document" },
  { name: "ocr", label: "OCR" },
  { name: "knowledge", label: "Knowledge" },
  { name: "meeting", label: "Meeting" },
  { name: "security", label: "Security" },
  { name: "screenshot-pipeline", label: "Screenshot" },
];

/** Setup step component. */
export function PythonEnvStep(): React.JSX.Element {
  const enabledServers = useOnboardingStore((s) => s.enabledServers);
  const isProvisioning = useOnboardingStore((s) => s.isProvisioningPython);
  const statuses = useOnboardingStore((s) => s.pythonEnvStatuses);
  const progress = useOnboardingStore((s) => s.pythonEnvProgress);
  const provisionPythonEnvs = useOnboardingStore((s) => s.provisionPythonEnvs);
  const retryPythonEnv = useOnboardingStore((s) => s.retryPythonEnv);
  const nextStep = useOnboardingStore((s) => s.nextStep);
  const prevStep = useOnboardingStore((s) => s.prevStep);
  const error = useOnboardingStore((s) => s.error);

  const hasStarted = useRef(false);

  // Enabled Python servers only
  const enabledPython = PYTHON_SERVERS.filter((s) => enabledServers.includes(s.name));

  // Auto-provision on mount (once)
  useEffect(() => {
    if (hasStarted.current || enabledPython.length === 0) return;
    hasStarted.current = true;
    void provisionPythonEnvs();
  }, [enabledPython.length, provisionPythonEnvs]);

  // Auto-advance if no Python servers are enabled
  useEffect(() => {
    if (enabledPython.length === 0) {
      nextStep();
    }
  }, [enabledPython.length, nextStep]);

  const handleRetry = useCallback(
    (serverName: string) => {
      void retryPythonEnv(serverName);
    },
    [retryPythonEnv],
  );

  const allReady = statuses.length > 0 && statuses.every((s) => s.ready);
  const hasFailures = statuses.some((s) => !s.ready && s.error);

  /** Get display status for a server. */
  const getServerStatus = (
    name: string,
  ): { icon: string; text: string; error: string | null } => {
    const status = statuses.find((s) => s.server === name);
    if (!status) {
      // Check if this is the currently progressing server
      if (progress?.server === name) {
        return { icon: "⏳", text: progress.message, error: null };
      }
      return { icon: "⏳", text: "Waiting...", error: null };
    }
    if (status.ready) {
      return { icon: "✅", text: "Ready", error: null };
    }
    if (status.error) {
      return { icon: "❌", text: "Failed", error: status.error };
    }
    return { icon: "⏳", text: "In progress...", error: null };
  };

  if (enabledPython.length === 0) {
    return <div className="onboarding-step" />;
  }

  return (
    <div className="onboarding-step python-env-step">
      <h2 className="step-title">Setting Up Tools</h2>
      <p className="step-description">
        Installing dependencies for Python-based tool servers.
        This happens once and takes about a minute.
      </p>

      <div className="python-env-list">
        {enabledPython.map((srv) => {
          const st = getServerStatus(srv.name);
          return (
            <div key={srv.name} className="python-env-row">
              <span className="python-env-icon">{st.icon}</span>
              <div className="python-env-info">
                <span className="python-env-name">{srv.label}</span>
                <span className="python-env-status">{st.text}</span>
                {st.error ? (
                  <div className="python-env-error">
                    <span className="python-env-error-text">{st.error}</span>
                    <button
                      className="python-env-retry-btn"
                      onClick={() => handleRetry(srv.name)}
                      type="button"
                    >
                      Retry
                    </button>
                  </div>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>

      {error ? <p className="step-error">{error}</p> : null}

      {allReady ? (
        <p className="python-env-success">All tool servers are ready.</p>
      ) : null}

      <div className="step-actions">
        <button
          className="onboarding-btn secondary"
          onClick={prevStep}
          type="button"
          disabled={isProvisioning}
        >
          Back
        </button>
        <button
          className="onboarding-btn primary"
          onClick={nextStep}
          type="button"
          disabled={isProvisioning}
        >
          {allReady ? "Continue" : hasFailures ? "Continue Anyway" : "Setting up..."}
        </button>
      </div>
    </div>
  );
}
