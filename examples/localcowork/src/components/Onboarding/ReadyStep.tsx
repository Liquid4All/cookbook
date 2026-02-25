/**
 * Completion step -- all set, ready to go.
 * Shows model name and accuracy in the checklist.
 */
import { useOnboardingStore } from "../../stores/onboardingStore";
import { RECOMMENDED_MODEL_ACCURACY } from "../../stores/onboardingStore";
import { isLfm2Model } from "./modelStepUtils";

/** Format model path for display in the checklist. */
function formatModelLabel(modelPath: string): string {
  if (modelPath.startsWith("llama-server:")) {
    const name = modelPath.slice(13);
    return `Model: ${name} (${RECOMMENDED_MODEL_ACCURACY} accuracy)`;
  }
  if (modelPath.startsWith("ollama:")) {
    return `Model: ${modelPath.slice(7)} (via Ollama)`;
  }
  if (isLfm2Model(modelPath)) {
    return `Model: LFM2-24B-A2B (${RECOMMENDED_MODEL_ACCURACY} accuracy)`;
  }
  // Local file — truncate long paths
  const name = modelPath.split("/").pop() ?? modelPath;
  return `Model: ${name}`;
}

/** Ready step component. */
export function ReadyStep(): React.JSX.Element {
  const modelPath = useOnboardingStore((s) => s.modelPath);
  const workingDirectory = useOnboardingStore((s) => s.workingDirectory);
  const enabledServers = useOnboardingStore((s) => s.enabledServers);
  const completeOnboarding = useOnboardingStore((s) => s.completeOnboarding);
  const prevStep = useOnboardingStore((s) => s.prevStep);

  const modelLabel = modelPath ? formatModelLabel(modelPath) : "Model loaded";

  const checks = [
    { label: modelLabel, ok: !!modelPath },
    { label: `${enabledServers.length} tools connected`, ok: enabledServers.length > 0 },
    { label: "Working folder set", ok: !!workingDirectory },
  ];

  return (
    <div className="onboarding-step ready-step">
      <h2 className="step-title">You Are All Set</h2>
      <p className="step-description">LocalCowork is ready to go.</p>
      <div className="ready-checklist">
        {checks.map((c) => (
          <div key={c.label} className="ready-check-item">
            <span className={"ready-check-icon" + (c.ok ? " ready-check-ok" : "")}>
              {c.ok ? "\u2713" : "\u2014"}
            </span>
            <span className="ready-check-label">{c.label}</span>
          </div>
        ))}
      </div>
      <div className="step-actions">
        <button className="onboarding-btn secondary" onClick={prevStep} type="button">Back</button>
        <button className="onboarding-btn primary" onClick={completeOnboarding} type="button">
          Start Using LocalCowork
        </button>
      </div>
    </div>
  );
}
