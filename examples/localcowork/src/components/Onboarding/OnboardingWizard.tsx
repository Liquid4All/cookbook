/**
 * Onboarding wizard -- main container with step tracking.
 *
 * Steps: Welcome > Hardware > Model > Folder > Servers > Setup > Demo > Ready
 */
import { useOnboardingStore } from "../../stores/onboardingStore";
import { WelcomeStep } from "./WelcomeStep";
import { HardwareStep } from "./HardwareStep";
import { ModelStep } from "./ModelStep";
import { FolderStep } from "./FolderStep";
import { ServerStep } from "./ServerStep";
import { PythonEnvStep } from "./PythonEnvStep";
import { DemoStep } from "./DemoStep";
import { ReadyStep } from "./ReadyStep";

/** Step labels for the progress indicator. */
const STEP_LABELS: readonly string[] = [
  "Welcome", "Hardware", "Model", "Folder", "Servers", "Setup", "Demo", "Ready",
];

/** Render the current step component. */
function CurrentStep({ step }: { readonly step: number }): React.JSX.Element {
  switch (step) {
    case 0: return <WelcomeStep />;
    case 1: return <HardwareStep />;
    case 2: return <ModelStep />;
    case 3: return <FolderStep />;
    case 4: return <ServerStep />;
    case 5: return <PythonEnvStep />;
    case 6: return <DemoStep />;
    case 7: return <ReadyStep />;
    default: return <WelcomeStep />;
  }
}

/** Onboarding wizard container. */
export function OnboardingWizard(): React.JSX.Element {
  const currentStep = useOnboardingStore((s) => s.currentStep);
  const totalSteps = useOnboardingStore((s) => s.totalSteps);

  return (
    <div className="onboarding-overlay">
      <div className="onboarding-container">
        <div className="onboarding-progress">
          {STEP_LABELS.map((label, i) => (
            <div
              key={label}
              className={"onboarding-dot-wrapper" + (i === currentStep ? " active" : "")
                + (i < currentStep ? " done" : "")}
            >
              <div className="onboarding-dot" />
              <span className="onboarding-dot-label">{label}</span>
            </div>
          ))}
        </div>
        <div className="onboarding-step-container">
          <CurrentStep step={currentStep} />
        </div>
        <div className="onboarding-step-counter">
          Step {currentStep + 1} of {totalSteps}
        </div>
      </div>
    </div>
  );
}
