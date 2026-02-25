/**
 * Hardware detection step — shows detected system specs,
 * recommended runtime/quantization, and model recommendation.
 */

import { useEffect } from "react";
import { useOnboardingStore } from "../../stores/onboardingStore";
import {
  RECOMMENDED_MODEL_DISPLAY,
  RECOMMENDED_MODEL_SIZE,
  FALLBACK_MODEL_DISPLAY,
  FALLBACK_MODEL_SIZE,
} from "../../stores/onboardingStore";

/** Hardware step component. */
export function HardwareStep(): React.JSX.Element {
  const hardware = useOnboardingStore((s) => s.hardware);
  const isDetecting = useOnboardingStore((s) => s.isDetectingHardware);
  const error = useOnboardingStore((s) => s.error);
  const detectHardware = useOnboardingStore((s) => s.detectHardware);
  const nextStep = useOnboardingStore((s) => s.nextStep);
  const prevStep = useOnboardingStore((s) => s.prevStep);

  useEffect(() => {
    if (!hardware && !isDetecting) {
      void detectHardware();
    }
  }, [hardware, isDetecting, detectHardware]);

  if (isDetecting) {
    return (
      <div className="onboarding-step hardware-step">
        <h2 className="step-title">Detecting Hardware</h2>
        <p className="step-description">Scanning your system capabilities...</p>
        <div className="hardware-loading">
          <div className="spinner" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="onboarding-step hardware-step">
        <h2 className="step-title">Hardware Detection</h2>
        <div className="onboarding-error">{error}</div>
        <div className="step-actions">
          <button className="onboarding-btn secondary" onClick={prevStep} type="button">
            Back
          </button>
          <button className="onboarding-btn primary" onClick={() => void detectHardware()} type="button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!hardware) {
    return <div className="onboarding-step hardware-step" />;
  }

  return (
    <div className="onboarding-step hardware-step">
      <h2 className="step-title">Your Hardware</h2>
      <p className="step-description">
        LocalCowork detected the following system capabilities.
      </p>

      <div className="hardware-card">
        <div className="hardware-section">
          <span className="hardware-label">CPU</span>
          <span className="hardware-value">{hardware.cpuModel}</span>
        </div>
        <div className="hardware-section">
          <span className="hardware-label">Cores / Threads</span>
          <span className="hardware-value">
            {hardware.cpuCores} cores / {hardware.cpuThreads} threads
          </span>
        </div>
        <div className="hardware-section">
          <span className="hardware-label">RAM</span>
          <span className="hardware-value">
            {hardware.ramTotalGb} GB total ({hardware.ramAvailableGb} GB available)
          </span>
        </div>
        <div className="hardware-section">
          <span className="hardware-label">OS</span>
          <span className="hardware-value">
            {hardware.osName} {hardware.osVersion} ({hardware.arch})
          </span>
        </div>
        {hardware.gpu ? (
          <div className="hardware-section">
            <span className="hardware-label">GPU</span>
            <span className="hardware-value">{hardware.gpu.model}</span>
          </div>
        ) : null}
      </div>

      <div className="hardware-recommendations">
        <div className="recommendation-badge">
          <span className="recommendation-label">Recommended Runtime</span>
          <span className="recommendation-value">{hardware.recommendedRuntime}</span>
        </div>
        <div className="recommendation-badge">
          <span className="recommendation-label">Recommended Quantization</span>
          <span className="recommendation-value">{hardware.recommendedQuantization}</span>
        </div>
      </div>

      <div className="hardware-model-recommendation">
        {hardware.ramTotalGb >= 16 ? (
          <p className="model-rec model-rec-primary">
            Your hardware can run <strong>{RECOMMENDED_MODEL_DISPLAY}</strong>{" "}
            ({RECOMMENDED_MODEL_SIZE}) &mdash; recommended for best accuracy.
          </p>
        ) : (
          <p className="model-rec model-rec-fallback">
            Your hardware is best suited for <strong>{FALLBACK_MODEL_DISPLAY}</strong>{" "}
            ({FALLBACK_MODEL_SIZE}) &mdash; a lightweight MoE model.
          </p>
        )}
      </div>

      <div className="step-actions">
        <button className="onboarding-btn secondary" onClick={prevStep} type="button">
          Back
        </button>
        <button className="onboarding-btn primary" onClick={nextStep} type="button">
          Continue
        </button>
      </div>
    </div>
  );
}
