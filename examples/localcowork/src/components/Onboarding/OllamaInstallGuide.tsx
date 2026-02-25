/**
 * Platform-specific Ollama installation guide.
 *
 * Reads the detected OS from the onboarding store's hardware info
 * and shows tailored install + start commands for macOS, Linux, or Windows.
 */

import { useOnboardingStore } from "../../stores/onboardingStore";

interface OllamaInstallGuideProps {
  readonly onRefresh: () => void;
}

/** Detect platform from hardware.osName. */
function detectPlatform(osName: string | undefined): "mac" | "linux" | "windows" | "unknown" {
  if (!osName) return "unknown";
  const lower = osName.toLowerCase();
  if (lower.includes("mac") || lower.includes("darwin")) return "mac";
  if (lower.includes("linux")) return "linux";
  if (lower.includes("windows")) return "windows";
  return "unknown";
}

/** macOS install instructions. */
function MacInstructions(): React.JSX.Element {
  return (
    <div className="ollama-install-guide">
      <div className="install-step">
        <span className="install-step-number">Step 1</span>
        <span className="install-step-label">Install Ollama</span>
        <code className="install-code">brew install ollama</code>
        <span className="install-step-alt">
          Or download from{" "}
          <a href="https://ollama.ai/download" target="_blank" rel="noopener noreferrer">
            ollama.ai/download
          </a>
        </span>
      </div>
      <div className="install-step">
        <span className="install-step-number">Step 2</span>
        <span className="install-step-label">Start the Ollama service</span>
        <code className="install-code">ollama serve</code>
      </div>
    </div>
  );
}

/** Linux install instructions. */
function LinuxInstructions(): React.JSX.Element {
  return (
    <div className="ollama-install-guide">
      <div className="install-step">
        <span className="install-step-number">Step 1</span>
        <span className="install-step-label">Install Ollama</span>
        <code className="install-code">
          curl -fsSL https://ollama.ai/install.sh | sh
        </code>
      </div>
      <div className="install-step">
        <span className="install-step-number">Step 2</span>
        <span className="install-step-label">Start the Ollama service</span>
        <code className="install-code">ollama serve</code>
      </div>
    </div>
  );
}

/** Windows install instructions. */
function WindowsInstructions(): React.JSX.Element {
  return (
    <div className="ollama-install-guide">
      <div className="install-step">
        <span className="install-step-number">Step 1</span>
        <span className="install-step-label">
          Download the installer from{" "}
          <a href="https://ollama.ai/download" target="_blank" rel="noopener noreferrer">
            ollama.ai/download
          </a>
        </span>
      </div>
      <div className="install-step">
        <span className="install-step-number">Step 2</span>
        <span className="install-step-label">
          Run the installer and follow the prompts. Ollama starts automatically.
        </span>
      </div>
    </div>
  );
}

/** Platform-specific Ollama install guide with Refresh button. */
export function OllamaInstallGuide({
  onRefresh,
}: OllamaInstallGuideProps): React.JSX.Element {
  const hardware = useOnboardingStore((s) => s.hardware);
  const platform = detectPlatform(hardware?.osName);

  return (
    <div className="model-option-body">
      <div className="ollama-status ollama-status-off">
        <span className="ollama-status-dot" />
        Ollama is not running
      </div>

      {platform === "mac" ? (
        <MacInstructions />
      ) : platform === "linux" ? (
        <LinuxInstructions />
      ) : platform === "windows" ? (
        <WindowsInstructions />
      ) : (
        /* Unknown OS — show generic instructions */
        <div className="ollama-install-guide">
          <p className="install-step-label">
            Install Ollama from{" "}
            <a href="https://ollama.ai/download" target="_blank" rel="noopener noreferrer">
              ollama.ai/download
            </a>
            , then run <code className="install-code-inline">ollama serve</code> in
            your terminal.
          </p>
        </div>
      )}

      <p className="install-step-hint">
        Once Ollama is running, click Refresh to detect it.
      </p>
      <button
        className="onboarding-btn secondary"
        onClick={onRefresh}
        type="button"
      >
        Refresh
      </button>
    </div>
  );
}
