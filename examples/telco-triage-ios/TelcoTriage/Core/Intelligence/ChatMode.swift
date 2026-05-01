import Foundation

/// A first-class chat-mode classifier. The mode gate is the primary
/// branch in the telco support chat flow:
///
///  - `.kbQuestion`      → generative retrieval over the 32-entry KB
///                         (`KBExtractor` emits `[cite_passage(...)]`).
///  - `.toolAction`      → `ToolSelector` picks one of 8 tools and
///                         fills parameters; user confirms; tool runs.
///  - `.personalSummary` → grounded on `CustomerContext`, not the KB.
///                         ("Summarize my home network.")
///  - `.outOfScope`      → decline locally or escalate to cloud.
///
/// This replaces the TF-IDF-driven `SupportRouter` path that couples
/// retrieval to routing. Routing is now a pure LFM decision; retrieval
/// happens only on the question branch and is itself generative.
///
/// Production implementation is `LFMChatModeRouter`, backed by the
/// base LFM2.5-350M on device (or a sidecar backend — same protocol,
/// injectable). `StubChatModeRouter` is the compile-only deterministic
/// stub used in tests and during the BUG-022 on-device inference
/// outage so the surrounding pipeline stays exercisable.
public protocol ChatModeRouter: Sendable {
    func classify(query: String) async -> ChatModePrediction
}

/// Four-mode taxonomy. Wire-format strings are snake_case for parity
/// with training data formats used by `scripts/generate_telco_*.py`
/// conventions; they also serialize cleanly to the trace row and
/// audit log.
public enum ChatMode: String, Sendable, Codable, CaseIterable {
    case kbQuestion      = "kb_question"
    case toolAction      = "tool_action"
    case personalSummary = "personal_summary"
    case outOfScope      = "out_of_scope"

    /// Human-readable label for the trace row and debug UI. Kept
    /// terse — the trace row has limited width.
    public var displayName: String {
        switch self {
        case .kbQuestion:      return "Question (KB)"
        case .toolAction:      return "Action (tool)"
        case .personalSummary: return "Personal summary"
        case .outOfScope:      return "Out of scope"
        }
    }

    /// Bridge to the legacy `RoutingPath` so `RoutingSummary` renders
    /// identically whether the decision came from the TF-IDF-era
    /// `SupportRouter` or the new `ChatModeRouter`. Will become the
    /// single source of truth once Phase A.3 deletes `RoutingPath`.
    public var routingPath: RoutingPath {
        switch self {
        case .kbQuestion:      return .answerWithRAG
        case .toolAction:      return .toolCall
        case .personalSummary: return .personalized
        case .outOfScope:      return .outOfScope
        }
    }
}

/// A single mode-router call result. `reasoning` is a short rationale
/// the LFM produces alongside the mode — surfaces as a trace-row
/// tooltip so the reviewer can see why the model picked that branch.
/// `runtimeMS` is measured wall-clock including prompt build, chat
/// template application, and detokenization.
public struct ChatModePrediction: Sendable, Equatable {
    public let mode: ChatMode
    public let confidence: Double
    public let reasoning: String
    public let runtimeMS: Int

    public init(
        mode: ChatMode,
        confidence: Double,
        reasoning: String,
        runtimeMS: Int
    ) {
        self.mode = mode
        self.confidence = confidence
        self.reasoning = reasoning
        self.runtimeMS = runtimeMS
    }
}
