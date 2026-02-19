/**
 * Welcome step — first screen of the onboarding wizard.
 *
 * Shows the LocalCowork logo, tagline, and a "Get Started" button.
 */

import { useOnboardingStore } from "../../stores/onboardingStore";

/** Welcome step component. */
export function WelcomeStep(): React.JSX.Element {
  const nextStep = useOnboardingStore((s) => s.nextStep);

  return (
    <div className="onboarding-step welcome-step">
      <div className="welcome-logo">LC</div>
      <h2 className="welcome-title">Welcome to LocalCowork</h2>
      <p className="welcome-tagline">
        Your on-device AI assistant. Private, powerful, and entirely local.
        No cloud, no data leaves your machine.
      </p>
      <p className="welcome-description">
        This wizard will set up your hardware, download a language model,
        and configure your workspace. It only takes a few minutes.
      </p>
      <button
        className="onboarding-btn primary"
        onClick={nextStep}
        type="button"
      >
        Get Started
      </button>
    </div>
  );
}
