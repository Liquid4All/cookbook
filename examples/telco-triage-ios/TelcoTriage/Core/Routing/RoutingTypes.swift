import Foundation

/// Where this query is handled. Every chat path terminates at an
/// LFM-generated assistant message on device; cloud-assist signals are
/// surfaced in the engineering trace and approval UI, not sent silently.
public enum RoutingPath: String, Sendable, Equatable {
    /// Grounded Q&A. The local retriever selects the top KB article, the base LFM
    /// synthesizes a short answer citing the retrieved material, and the
    /// user can tap "Read full article" to see the source KB entry.
    case answerWithRAG

    /// The LFM tool selector picked a tool. The assistant renders a
    /// ToolDecisionCard with the extracted arguments and a Confirm button;
    /// confirming invokes `ToolExecutor`, which runs `Tool.execute(...)`
    /// against real customer state (e.g. pausing a device in
    /// `CustomerContext.managedDevices`).
    case toolCall

    /// No retrieval, no tool — the LFM reasons directly over the
    /// `CustomerContext` snapshot (plan, equipment, managed devices,
    /// usage). Used by the "summarize my home network" chip.
    case personalized

    /// The intent classifier emitted `unknown` with low confidence. The
    /// LFM composes a polite "I only handle home internet support" reply
    /// that reinforces the privacy guarantee: nothing left the device.
    case outOfScope

    /// The on-device classifier determined the request needs live operator
    /// systems. The app prepares a redacted handoff payload and does not
    /// silently send raw customer text.
    case cloudAssist
}

/// Tool intents the assistant recognizes. Maps 1:1 to the tool catalog
/// emitted by the `telco-tool-selector` LoRA fine-tune. The model's
/// `tool_id` is hyphenated (`"restart-router"`, `"enable-wps"`, …);
/// Swift case names stay camelCase for ergonomics. Use `toolID` /
/// `init?(toolID:)` to bridge the two.
///
/// The model also emits `"none"` when no tool matches — that maps to the
/// `ToolSelection.none` sentinel, not to an enum case, so callers can
/// distinguish "no tool" from "unknown tool string".
///
/// NOTE: `setDowntime` is intentionally absent. `SetDowntimeTool` exists
/// in the codebase as forward-looking scaffolding but the fine-tuned
/// adapter has never been trained on time-bounded pauses — see
/// `docs/FUTURE_SCOPE.md` for the retrain plan.
public enum ToolIntent: String, Sendable, Equatable, Codable, CaseIterable {
    case restartRouter
    case runSpeedTest
    case checkConnection
    case wpsPair
    case runDiagnostics
    case scheduleTechnician
    case toggleParentalControls
    case rebootExtender

    /// Hyphenated identifier the model emits in its `tool_id` JSON field.
    /// Matches the `id` on each concrete `Tool` in `ToolRegistry.default`.
    public var toolID: String {
        switch self {
        case .restartRouter:            return "restart-router"
        case .runSpeedTest:             return "run-speed-test"
        case .checkConnection:          return "check-connection"
        case .wpsPair:                  return "enable-wps"
        case .runDiagnostics:           return "run-diagnostics"
        case .scheduleTechnician:       return "schedule-technician"
        case .toggleParentalControls:   return "toggle-parental-controls"
        case .rebootExtender:           return "reboot-extender"
        }
    }

    /// Parse a hyphenated tool id emitted by the fine-tuned model.
    /// Returns nil for `"none"` (caller should translate to no selection)
    /// or any unknown string (including `"set-downtime"`, which the
    /// adapter is not yet trained on).
    public init?(toolID: String) {
        switch toolID {
        case "restart-router":              self = .restartRouter
        case "run-speed-test":              self = .runSpeedTest
        case "check-connection":            self = .checkConnection
        case "enable-wps":                  self = .wpsPair
        case "run-diagnostics":             self = .runDiagnostics
        case "schedule-technician":         self = .scheduleTechnician
        case "toggle-parental-controls":    self = .toggleParentalControls
        case "reboot-extender":             self = .rebootExtender
        default:                            return nil
        }
    }

    /// True for tools whose execution has irreversible or user-visible
    /// side effects. Matches the `requires_confirmation` / `is_destructive`
    /// flags in the model's tool catalog.
    public var requiresConfirmation: Bool {
        switch self {
        case .runSpeedTest, .checkConnection, .runDiagnostics: return false
        case .restartRouter, .wpsPair, .scheduleTechnician,
             .toggleParentalControls, .rebootExtender: return true
        }
    }

    /// Human-readable label for UI surfaces (chat trace chips, confirm
    /// sheets, pipeline-step rows). Kept next to the enum so any future
    /// case is visibly incomplete without a label.
    public var displayName: String {
        switch self {
        case .restartRouter:            return "Restart Router"
        case .runSpeedTest:             return "Run Speed Test"
        case .checkConnection:          return "Check Connection"
        case .wpsPair:                  return "WPS Pair"
        case .runDiagnostics:           return "Run Diagnostics"
        case .scheduleTechnician:       return "Schedule Technician"
        case .toggleParentalControls:   return "Parental Controls"
        case .rebootExtender:           return "Reboot Extender"
        }
    }
}

// `RoutingDecision` (the struct that bundled a TF-IDF-era routing
// choice with its piiSpans, topMatch, and reason) was removed in
// Phase A.3.b alongside `SupportRouter`. Chat routing is now a pure
// `ChatMode` decision handled directly in `ChatViewModel`; the
// `RoutingPath` enum above survives only as the UI-surface language
// for `RoutingSummary` (ChatMessage.swift), mapped from ChatMode via
// `ChatMode.routingPath`.
