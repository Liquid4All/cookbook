/**
 * Interactive demo step showing sample prompts.
 */
import { useState } from "react";
import { useOnboardingStore } from "../../stores/onboardingStore";

/** Sample prompts for the demo. */
const DEMO_PROMPTS: ReadonlyArray<{ prompt: string; desc: string; tools: readonly string[] }> = [
  {
    prompt: "Summarize the PDF on my desktop",
    desc: "Extracts text from a PDF and generates a summary.",
    tools: ["filesystem.list_dir", "document.extract_text", "document.summarize"],
  },
  {
    prompt: "Find all receipts from last month and total them",
    desc: "Searches files, runs OCR on images, extracts amounts.",
    tools: ["filesystem.search", "ocr.extract_text", "data.query_csv"],
  },
  {
    prompt: "Create a meeting summary from the recording",
    desc: "Transcribes audio, identifies speakers, and summarizes.",
    tools: ["meeting.transcribe", "meeting.diarize", "document.summarize"],
  },
  {
    prompt: "Check this contract for sensitive information",
    desc: "Scans document for PII, secrets, and confidential data.",
    tools: ["document.extract_text", "security.scan_pii", "security.scan_secrets"],
  },
];

/** Demo step component. */
export function DemoStep(): React.JSX.Element {
  const nextStep = useOnboardingStore((s) => s.nextStep);
  const prevStep = useOnboardingStore((s) => s.prevStep);
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="onboarding-step demo-step">
      <h2 className="step-title">See What LocalCowork Can Do</h2>
      <p className="step-description">
        Click a prompt to see which tools would be called. This is a read-only preview.
      </p>
      <div className="demo-prompt-list">
        {DEMO_PROMPTS.map((dp, i) => (
          <button
            key={dp.prompt}
            className={"demo-prompt-card" + (selected === i ? " demo-prompt-selected" : "")}
            onClick={() => setSelected(selected === i ? null : i)}
            type="button"
          >
            <span className="demo-prompt-text">{dp.prompt}</span>
            <span className="demo-prompt-desc">{dp.desc}</span>
            {selected === i ? (
              <div className="demo-prompt-tools">
                <span className="demo-prompt-tools-label">Tools:</span>
                {dp.tools.map((tool) => (
                  <span key={tool} className="demo-prompt-tool-badge">{tool}</span>
                ))}
              </div>
            ) : null}
          </button>
        ))}
      </div>
      <div className="step-actions">
        <button className="onboarding-btn secondary" onClick={prevStep} type="button">Back</button>
        <button className="onboarding-btn primary" onClick={nextStep} type="button">Continue</button>
      </div>
    </div>
  );
}
