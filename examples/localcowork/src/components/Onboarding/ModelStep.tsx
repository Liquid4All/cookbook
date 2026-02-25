/**
 * Model setup step — detect llama-server (LFM2-24B-A2B), Ollama models,
 * or select a local GGUF file.
 *
 * Detection priority:
 * 1. llama-server on localhost:8080 (LFM2-24B-A2B — recommended)
 * 2. Ollama on localhost:11434 (secondary, any model)
 * 3. Browse for a local GGUF file
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  useOnboardingStore,
  RECOMMENDED_MODEL_DISPLAY,
  RECOMMENDED_MODEL_ACCURACY,
  FALLBACK_OLLAMA_MODEL,
} from "../../stores/onboardingStore";
import type { OllamaModelInfo } from "../../types";
import type { UnlistenFn } from "@tauri-apps/api/event";

import {
  ModelReady,
  OllamaPullProgressView,
  GgufDownloadProgressView,
} from "./ModelProgressViews";
import { ModelSelectionPanel } from "./ModelSelectionPanel";
import { LocalModelGuide } from "./LocalModelGuide";

/** Model setup step component. */
export function ModelStep(): React.JSX.Element {
  const modelPath = useOnboardingStore((s) => s.modelPath);
  const isDownloading = useOnboardingStore((s) => s.isDownloading);
  const downloadProgress = useOnboardingStore((s) => s.downloadProgress);
  const error = useOnboardingStore((s) => s.error);
  const setModelPath = useOnboardingStore((s) => s.setModelPath);
  const nextStep = useOnboardingStore((s) => s.nextStep);
  const prevStep = useOnboardingStore((s) => s.prevStep);
  const clearError = useOnboardingStore((s) => s.clearError);

  // llama-server state
  const llamaServerAvailable = useOnboardingStore((s) => s.llamaServerAvailable);
  const isCheckingLlamaServer = useOnboardingStore((s) => s.isCheckingLlamaServer);
  const checkLlamaServer = useOnboardingStore((s) => s.checkLlamaServer);
  const selectLlamaServer = useOnboardingStore((s) => s.selectLlamaServer);

  // Ollama state
  const ollamaAvailable = useOnboardingStore((s) => s.ollamaAvailable);
  const ollamaModels = useOnboardingStore((s) => s.ollamaModels);
  const isCheckingOllama = useOnboardingStore((s) => s.isCheckingOllama);
  const isPullingOllama = useOnboardingStore((s) => s.isPullingOllama);
  const ollamaPullProgress = useOnboardingStore((s) => s.ollamaPullProgress);
  const checkOllama = useOnboardingStore((s) => s.checkOllama);
  const pullOllamaModel = useOnboardingStore((s) => s.pullOllamaModel);
  const selectOllamaModel = useOnboardingStore((s) => s.selectOllamaModel);

  const unlistenRef = useRef<UnlistenFn | null>(null);
  const [showAlternatives, setShowAlternatives] = useState(false);

  // Auto-detect: llama-server first, then Ollama
  useEffect(() => {
    void checkLlamaServer();
    void checkOllama();
  }, [checkLlamaServer, checkOllama]);

  // Auto-select if llama-server is running and no model chosen yet
  useEffect(() => {
    if (llamaServerAvailable && !modelPath) {
      selectLlamaServer();
    }
  }, [llamaServerAvailable, modelPath, selectLlamaServer]);

  // Cleanup event listener on unmount
  useEffect(() => {
    return () => {
      if (unlistenRef.current) {
        unlistenRef.current();
      }
    };
  }, []);

  const handlePullFallback = useCallback(async (): Promise<void> => {
    clearError();
    const unlisten = await pullOllamaModel(FALLBACK_OLLAMA_MODEL);
    unlistenRef.current = unlisten;
  }, [pullOllamaModel, clearError]);

  const handleSelectOllamaModel = useCallback(
    (model: OllamaModelInfo): void => {
      clearError();
      selectOllamaModel(model.name);
      setShowAlternatives(false);
    },
    [selectOllamaModel, clearError],
  );

  const handleUseLocalFile = useCallback(async (): Promise<void> => {
    clearError();
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        multiple: false,
        filters: [
          { name: "Model Files", extensions: ["gguf", "bin", "safetensors"] },
        ],
      });
      if (selected && typeof selected === "string") {
        setModelPath(selected);
        setShowAlternatives(false);
      }
    } catch {
      setModelPath("(manual selection)");
      setShowAlternatives(false);
    }
  }, [setModelPath, clearError]);

  const handleSelectLlamaServer = useCallback((): void => {
    clearError();
    selectLlamaServer();
    setShowAlternatives(false);
  }, [selectLlamaServer, clearError]);

  const handleRefreshLlamaServer = useCallback((): void => {
    void checkLlamaServer();
  }, [checkLlamaServer]);

  const isBusy = isDownloading || isPullingOllama || isCheckingOllama || isCheckingLlamaServer;
  const isLocalFile = modelPath !== null &&
    !modelPath.startsWith("ollama:") &&
    !modelPath.startsWith("llama-server:");

  const selectionPanel = (
    <ModelSelectionPanel
      llamaServerAvailable={llamaServerAvailable}
      isCheckingLlamaServer={isCheckingLlamaServer}
      ollamaAvailable={ollamaAvailable}
      ollamaModels={ollamaModels}
      isCheckingOllama={isCheckingOllama}
      onSelectLlamaServer={handleSelectLlamaServer}
      onRefreshLlamaServer={handleRefreshLlamaServer}
      onPullFallback={() => void handlePullFallback()}
      onSelectOllamaModel={handleSelectOllamaModel}
      onUseLocalFile={() => void handleUseLocalFile()}
      onRefreshOllama={() => void checkOllama()}
    />
  );

  return (
    <div className="onboarding-step model-step">
      <h2 className="step-title">Set Up Model</h2>
      <p className="step-description">
        LocalCowork needs a language model to run locally on your device.
        We recommend <strong>{RECOMMENDED_MODEL_DISPLAY}</strong> &mdash;
        it scores {RECOMMENDED_MODEL_ACCURACY} tool-calling accuracy across
        all 67 tools.
      </p>

      {error ? <div className="onboarding-error">{error}</div> : null}

      {/* Progress views */}
      {isPullingOllama ? (
        <OllamaPullProgressView progress={ollamaPullProgress} />
      ) : isDownloading ? (
        <GgufDownloadProgressView progress={downloadProgress} />
      ) : modelPath ? (
        <>
          <ModelReady modelPath={modelPath} />
          {isLocalFile ? <LocalModelGuide modelPath={modelPath} /> : null}
          <button
            className="model-change-btn"
            onClick={() => setShowAlternatives((v) => !v)}
            type="button"
          >
            {showAlternatives ? "Hide alternatives" : "Pick a different model"}
          </button>
          {showAlternatives ? selectionPanel : null}
        </>
      ) : (
        selectionPanel
      )}

      <div className="step-actions">
        <button
          className="onboarding-btn secondary"
          onClick={prevStep}
          type="button"
          disabled={isBusy}
        >
          Back
        </button>
        <button
          className="onboarding-btn ghost"
          onClick={nextStep}
          type="button"
        >
          Skip
        </button>
        {modelPath ? (
          <button
            className="onboarding-btn primary"
            onClick={nextStep}
            type="button"
          >
            Continue
          </button>
        ) : null}
      </div>
    </div>
  );
}
