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
  const isOnboardingComplete = useOnboardingStore((s) => s.isComplete);

  if (!isOnboardingComplete) {
    return <OnboardingWizard />;
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>LocalCowork</h1>
        <span className="app-badge">on-device</span>
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
