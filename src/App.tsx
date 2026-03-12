import { useEffect } from "react";

import { ChatPanel } from "./components/Chat";
import { FileBrowser } from "./components/FileBrowser";
import { OnboardingWizard } from "./components/Onboarding";
import { SettingsPanel } from "./components/Settings";
import { useOnboardingStore } from "./stores/onboardingStore";
import { useSettingsStore } from "./stores/settingsStore";

/**
 * Root application component.
 *
 * Shows the OnboardingWizard on first run, then the main app layout.
 */
export function App(): React.JSX.Element {
  const toggleSettings = useSettingsStore((s) => s.togglePanel);
  const isSettingsOpen = useSettingsStore((s) => s.isOpen);
  const startConfigWatch = useSettingsStore((s) => s.startConfigWatch);
  const stopConfigWatch = useSettingsStore((s) => s.stopConfigWatch);
  const configReloadNotification = useSettingsStore(
    (s) => s.configReloadNotification,
  );
  const clearConfigReloadNotification = useSettingsStore(
    (s) => s.clearConfigReloadNotification,
  );
  const isOnboardingComplete = useOnboardingStore((s) => s.isComplete);

  // Start/stop config file watching based on settings panel state
  useEffect(() => {
    if (isSettingsOpen) {
      startConfigWatch();
    } else {
      stopConfigWatch();
    }
    return () => stopConfigWatch();
  }, [isSettingsOpen, startConfigWatch, stopConfigWatch]);

  if (!isOnboardingComplete) {
    return <OnboardingWizard />;
  }

  return (
    <div className="app-container">
      {/* Config reload toast notification */}
      {configReloadNotification && (
        <div
          className="config-reload-toast"
          onClick={clearConfigReloadNotification}
        >
          <span className="toast-icon">🔄</span>
          <span className="toast-message">{configReloadNotification}</span>
          <button className="toast-close" aria-label="Dismiss">
            ×
          </button>
        </div>
      )}

      <header className="app-header">
        <div className="app-title-group">
          <div className="app-title-row">
            <h1>LocalCowork</h1>
            <span className="app-badge">on-device</span>
          </div>
          <span className="app-subtitle">
            powered by LFM2-24B-A2B from Liquid AI
          </span>
        </div>
        <div className="app-header-spacer" />
        <button
          className="app-settings-btn"
          onClick={toggleSettings}
          type="button"
          title="Settings"
          aria-label="Open settings"
        >
          &#9881;
        </button>
      </header>

      <main className="app-main">
        <FileBrowser />
        <ChatPanel />
      </main>

      <footer className="app-footer">
        <span>v0.1.0 &mdash; Agent Core</span>
      </footer>

      <SettingsPanel />
    </div>
  );
}
