/**
 * Folder selection step -- pick the working directory.
 */
import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useOnboardingStore } from "../../stores/onboardingStore";

/** Folder step component. */
export function FolderStep(): React.JSX.Element {
  const workingDirectory = useOnboardingStore((s) => s.workingDirectory);
  const setWorkingDirectory = useOnboardingStore((s) => s.setWorkingDirectory);
  const nextStep = useOnboardingStore((s) => s.nextStep);
  const prevStep = useOnboardingStore((s) => s.prevStep);
  const [folderContents, setFolderContents] = useState<readonly string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    const go = async (): Promise<void> => {
      try {
        type E = { name: string; entryType: string };
        const e = await invoke<readonly E[]>("list_directory", { path: workingDirectory });
        if (!cancelled) setFolderContents(
          e.slice(0, 8).map((x) => (x.entryType === "dir" ? "[dir] " : "[file] ") + x.name));
      } catch { if (!cancelled) setFolderContents(["(unable to preview)"]); }
      finally { if (!cancelled) setIsLoading(false); }
    };
    void go();
    return () => { cancelled = true; };
  }, [workingDirectory]);

  const handleBrowse = useCallback(async (): Promise<void> => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({ directory: true, multiple: false });
      if (selected && typeof selected === "string") setWorkingDirectory(selected);
    } catch { /* plugin unavailable */ }
  }, [setWorkingDirectory]);

  return (
    <div className="onboarding-step folder-step">
      <h2 className="step-title">Working Directory</h2>
      <p className="step-description">
        Choose the folder where LocalCowork will operate.
      </p>
      <div className="folder-selector">
        <div className="folder-path-display">
          <span className="folder-path-text">{workingDirectory}</span>
          <button className="onboarding-btn secondary folder-browse-btn"
            onClick={() => void handleBrowse()} type="button">Browse</button>
        </div>
        <div className="folder-preview">
          <span className="folder-preview-title">Folder contents</span>
          {isLoading ? <span>Loading...</span> : (
            <ul className="folder-preview-list">
              {folderContents.map((item) => (
                <li key={item} className="folder-preview-item">{item}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <div className="step-actions">
        <button className="onboarding-btn secondary" onClick={prevStep} type="button">Back</button>
        <button className="onboarding-btn primary" onClick={nextStep} type="button">Continue</button>
      </div>
    </div>
  );
}
