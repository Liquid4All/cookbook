/**
 * MCP server configuration step.
 */
import { useOnboardingStore } from "../../stores/onboardingStore";

/** Server metadata for display. */
const SERVER_LIST: ReadonlyArray<{ name: string; desc: string; lang: string }> = [
  { name: "filesystem", desc: "File CRUD, watch, and search", lang: "TS" },
  { name: "document", desc: "Extraction, conversion, diff, PDF", lang: "Py" },
  { name: "ocr", desc: "Tesseract + PaddleOCR", lang: "Py" },
  { name: "knowledge", desc: "SQLite-vec RAG pipeline", lang: "Py" },
  { name: "meeting", desc: "Whisper.cpp + diarization", lang: "Py" },
  { name: "security", desc: "PII/secrets scan + encryption", lang: "Py" },
  { name: "calendar", desc: ".ics + system calendar API", lang: "TS" },
  { name: "email", desc: "MBOX/Maildir + SMTP", lang: "TS" },
  { name: "task", desc: "Local SQLite task DB", lang: "TS" },
  { name: "data", desc: "CSV + SQLite operations", lang: "TS" },
  { name: "audit", desc: "Audit log reader + reports", lang: "TS" },
  { name: "clipboard", desc: "OS clipboard (Tauri bridge)", lang: "TS" },
  { name: "system", desc: "OS APIs (Tauri bridge)", lang: "TS" },
];

/** Server configuration step component. */
export function ServerStep(): React.JSX.Element {
  const enabledServers = useOnboardingStore((s) => s.enabledServers);
  const toggleServer = useOnboardingStore((s) => s.toggleServer);
  const nextStep = useOnboardingStore((s) => s.nextStep);
  const prevStep = useOnboardingStore((s) => s.prevStep);

  return (
    <div className="onboarding-step server-step">
      <h2 className="step-title">MCP Servers</h2>
      <p className="step-description">
        These tool servers power LocalCowork. All are enabled by default.
      </p>
      <span className="server-step-advanced">Advanced</span>
      <div className="server-list">
        {SERVER_LIST.map((srv) => (
          <label key={srv.name} className="server-toggle-row">
            <div className="server-toggle-info">
              <span className="server-toggle-name">{srv.name}</span>
              <span className="server-toggle-lang">{srv.lang}</span>
              <span className="server-toggle-desc">{srv.desc}</span>
            </div>
            <input
              type="checkbox"
              className="server-toggle-checkbox"
              checked={enabledServers.includes(srv.name)}
              onChange={() => toggleServer(srv.name)}
            />
          </label>
        ))}
      </div>
      <div className="step-actions">
        <button className="onboarding-btn secondary" onClick={prevStep} type="button">Back</button>
        <button className="onboarding-btn primary" onClick={nextStep} type="button">Continue</button>
      </div>
    </div>
  );
}
