/**
 * Platform-specific llama-server + LFM2-24B-A2B setup guide.
 *
 * Reads the detected OS from the onboarding store's hardware info
 * and shows tailored install, download, and start commands.
 */

import { useOnboardingStore } from "../../stores/onboardingStore";

interface LlamaServerSetupGuideProps {
  readonly onRefresh: () => void;
}

/** Detect platform from hardware.osName. */
function detectPlatform(
  osName: string | undefined,
): "mac" | "linux" | "windows" | "unknown" {
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
        <span className="install-step-label">Install llama.cpp</span>
        <code className="install-code">brew install llama.cpp</code>
      </div>
      <div className="install-step">
        <span className="install-step-number">Step 2</span>
        <span className="install-step-label">Download the model (~16 GB)</span>
        <code className="install-code">
          pip install huggingface-hub{"\n"}
          huggingface-cli download LiquidAI/LFM2-24B-A2B-GGUF --local-dir _models/
        </code>
        <span className="install-step-alt">
          Or download directly from{" "}
          <a
            href="https://huggingface.co/LiquidAI/LFM2-24B-A2B-GGUF"
            target="_blank"
            rel="noopener noreferrer"
          >
            HuggingFace
          </a>
        </span>
      </div>
      <div className="install-step">
        <span className="install-step-number">Step 3</span>
        <span className="install-step-label">Start the model server</span>
        <code className="install-code">./scripts/start-model.sh</code>
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
        <span className="install-step-label">Install llama.cpp</span>
        <code className="install-code">
          git clone https://github.com/ggerganov/llama.cpp{"\n"}
          cd llama.cpp && make -j
        </code>
      </div>
      <div className="install-step">
        <span className="install-step-number">Step 2</span>
        <span className="install-step-label">Download the model (~16 GB)</span>
        <code className="install-code">
          pip install huggingface-hub{"\n"}
          huggingface-cli download LiquidAI/LFM2-24B-A2B-GGUF --local-dir _models/
        </code>
      </div>
      <div className="install-step">
        <span className="install-step-number">Step 3</span>
        <span className="install-step-label">Start the model server</span>
        <code className="install-code">./scripts/start-model.sh</code>
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
          Download llama.cpp from{" "}
          <a
            href="https://github.com/ggerganov/llama.cpp/releases"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub Releases
          </a>
        </span>
      </div>
      <div className="install-step">
        <span className="install-step-number">Step 2</span>
        <span className="install-step-label">Download the model (~16 GB)</span>
        <code className="install-code">
          pip install huggingface-hub{"\n"}
          huggingface-cli download LiquidAI/LFM2-24B-A2B-GGUF --local-dir _models/
        </code>
      </div>
      <div className="install-step">
        <span className="install-step-number">Step 3</span>
        <span className="install-step-label">
          Start the model server using <code className="install-code-inline">scripts\start-model.sh</code>{" "}
          or run llama-server manually.
        </span>
      </div>
    </div>
  );
}

/** Platform-specific llama-server setup guide with Refresh button. */
export function LlamaServerSetupGuide({
  onRefresh,
}: LlamaServerSetupGuideProps): React.JSX.Element {
  const hardware = useOnboardingStore((s) => s.hardware);
  const platform = detectPlatform(hardware?.osName);

  return (
    <div className="model-option-body">
      <div className="ollama-status ollama-status-off">
        <span className="ollama-status-dot" />
        llama-server is not running
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
            Install llama.cpp, download the LFM2-24B-A2B model from{" "}
            <a
              href="https://huggingface.co/LiquidAI/LFM2-24B-A2B-GGUF"
              target="_blank"
              rel="noopener noreferrer"
            >
              HuggingFace
            </a>
            , then run{" "}
            <code className="install-code-inline">./scripts/start-model.sh</code>
          </p>
        </div>
      )}

      <p className="install-step-hint">
        Once llama-server is running on port 8080, click Refresh to detect it.
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
