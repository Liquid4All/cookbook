import Foundation
import UIKit

/// A single message rendered in the chat UI. Assistant messages carry the
/// full routing + tool context so the UI can render source citations,
/// deep links, deflection badges, confirmation prompts, and tool results.
public struct ChatMessage: Identifiable, Equatable {
    public typealias Role = ChatTurn.Role

    public let id: UUID
    public let role: Role
    public var text: String
    public let timestamp: Date
    public var routing: RoutingSummary?
    public var piiSpans: [PIISpan]
    public var sourceEntry: KBEntry?
    public var deepLinks: [DeepLink]
    public var latencyMS: Int?
    public var isStreaming: Bool

    // Tool intelligence — what the model understood, not what it "executed"
    public var toolDecision: ToolDecision?

    // New: multimodal inputs
    public var attachedImage: UIImage?
    public var voiceInput: Bool

    // New: contextual Next-Best-Action surfaced with this assistant reply.
    // Identified by NBA id; the view layer looks up the concrete action
    // from the engine so accept/decline route through one place.
    public var attachedNBAID: String?

    // New: call-trace metadata — exactly which Liquid model produced this
    // message and how the time broke down. Rendered as a "LFM chip" in
    // the metadata row, tap-to-expand for the full trace. This is the
    // "prove it's really LFM" answer for the telco audience.
    public var trace: CallTrace?

    // New: typed vision diagnosis for photo-to-fix flows. When set, the
    // row renders a VisionDiagnosisCard below the bubble with an
    // optional one-tap "try this tool" button.
    public var visionDiagnosis: VisionDiagnosis?

    public init(
        id: UUID = UUID(),
        role: Role,
        text: String,
        timestamp: Date = Date(),
        routing: RoutingSummary? = nil,
        piiSpans: [PIISpan] = [],
        sourceEntry: KBEntry? = nil,
        deepLinks: [DeepLink] = [],
        latencyMS: Int? = nil,
        isStreaming: Bool = false,
        toolDecision: ToolDecision? = nil,
        attachedImage: UIImage? = nil,
        voiceInput: Bool = false,
        attachedNBAID: String? = nil,
        trace: CallTrace? = nil,
        visionDiagnosis: VisionDiagnosis? = nil
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.timestamp = timestamp
        self.routing = routing
        self.piiSpans = piiSpans
        self.sourceEntry = sourceEntry
        self.deepLinks = deepLinks
        self.latencyMS = latencyMS
        self.isStreaming = isStreaming
        self.toolDecision = toolDecision
        self.attachedImage = attachedImage
        self.voiceInput = voiceInput
        self.attachedNBAID = attachedNBAID
        self.trace = trace
        self.visionDiagnosis = visionDiagnosis
    }

    // Identity-only equality. UIImage isn't Equatable and messages are
    // append-only in the chat list (never mutated in place), so UUID
    // identity is sufficient for SwiftUI diffing and test assertions.
    // If you need value comparison in tests, assert individual fields.
    public static func == (lhs: ChatMessage, rhs: ChatMessage) -> Bool {
        lhs.id == rhs.id
    }
}

/// A "how this answer was produced" trace attached to each assistant
/// message. Gives the customer (and especially a telco engineer
/// reviewing the demo) a clear view of which Liquid model handled the
/// query, what the retrieval layer did, and how time broke down.
///
/// All-local measurements — nothing in here requires cloud telemetry.
public struct CallTrace: Equatable {
    public enum Surface: String, Sendable {
        case onDeviceRAG      // LFM2.5-350M grounded over local KB
        case tool             // LFM2.5-350M emits a tool call, tool runs locally
        case visionPack       // LFM2.5-VL-450M (or heuristic if pack absent)
        case voicePack        // LFM2.5-Audio-1.5B (or Apple Speech fallback)

        public var modelLabel: String {
            switch self {
            case .onDeviceRAG: return "LFM2.5-350M"
            case .tool: return "LFM2.5-350M + Tool"
            case .visionPack: return "LFM2.5-VL-450M"
            case .voicePack: return "LFM2.5-Audio-1.5B"
            }
        }

        public var displayName: String {
            switch self {
            case .onDeviceRAG: return "On-device RAG"
            case .tool: return "Tool call"
            case .visionPack: return "Vision pack"
            case .voicePack: return "Voice pack"
            }
        }
    }

    public let surface: Surface
    public let retrievalMS: Int?
    public let inferenceMS: Int
    public let topKBMatchID: String?
    public let topKBScore: Double?
    public let kbEntriesScanned: Int?
    public let inputTokens: Int?
    public let outputTokens: Int?
    // Populated by the Intelligence layer (chat mode router, KB
    // extractor, tool selector). Each field surfaces in the expanded
    // trace card so the demo can point at "this was an LFM call" per
    // pipeline step.
    public let chatMode: ChatMode?
    public let chatModeConfidence: Double?
    public let chatModeRuntimeMS: Int?
    public let extraction: ExtractionResult?
    public let toolSelectionReasoning: String?
    public let toolSelectionConfidence: Double?
    /// ADR-022 §4.3 Layer 1 — the full 5-head understanding vector
    /// produced by `QueryUnderstandingClassifier`. Renders into the
    /// `UnderstandingTraceCard` when present. Nil for legacy paths
    /// (audio cockpit demos, vision, etc.) that don't run the v2
    /// understanding layer.
    public let understanding: QueryUnderstanding?

    /// ADR-026 normal Telco Triage semantic vector. This is the active
    /// customer/demo control plane: one shared `telco-shared-clf-v1`
    /// forward pass, nine classifier heads. Nil for legacy fallback,
    /// vision, voice, and tests that do not inject it.
    public let telcoUnderstanding: TelcoSharedUnderstanding?
    public let telcoUnderstandingMS: Int?

    // -- Step 6.6 composer telemetry --
    //
    // Surgical four-field set per the Step 6 plan §5. Populated by
    // `ChatViewModel.runVerizonDispatch` from the dispatcher's
    // `VerizonDispatchResult`. Nil for non-Verizon paths (banking
    // POC pipeline, vision, voice) and for legacy Stage B turns.
    public let routePolicyMS: Int?
    public let composerMS: Int?
    public let totalWallMS: Int?
    public let composerRoute: String?
    public let composerCitedPageID: String?
    public let composerRenderedLinkID: String?
    public let composerConfirmationShown: Bool?
    /// ADR-029 §6 measurement only: policy-engine ground truth surfaced for the
    /// phone-flow harness (reuse flag + decision reason). nil off the telco path.
    public let reuseActiveEvidence: Bool?
    public let policyReason: String?
    /// ADR-029 §7 measurement only: the explicit dialogue-state operation (raw
    /// value) and its audit reason, so `situation_eval` scores the state decision
    /// directly. nil off the telco composer path.
    public let stateOperation: String?
    public let stateOperationReason: String?

    public init(
        surface: Surface,
        retrievalMS: Int? = nil,
        inferenceMS: Int,
        topKBMatchID: String? = nil,
        topKBScore: Double? = nil,
        kbEntriesScanned: Int? = nil,
        inputTokens: Int? = nil,
        outputTokens: Int? = nil,
        chatMode: ChatMode? = nil,
        chatModeConfidence: Double? = nil,
        chatModeRuntimeMS: Int? = nil,
        extraction: ExtractionResult? = nil,
        toolSelectionReasoning: String? = nil,
        toolSelectionConfidence: Double? = nil,
        understanding: QueryUnderstanding? = nil,
        telcoUnderstanding: TelcoSharedUnderstanding? = nil,
        telcoUnderstandingMS: Int? = nil,
        routePolicyMS: Int? = nil,
        composerMS: Int? = nil,
        totalWallMS: Int? = nil,
        composerRoute: String? = nil,
        composerCitedPageID: String? = nil,
        composerRenderedLinkID: String? = nil,
        composerConfirmationShown: Bool? = nil,
        reuseActiveEvidence: Bool? = nil,
        policyReason: String? = nil,
        stateOperation: String? = nil,
        stateOperationReason: String? = nil
    ) {
        self.surface = surface
        self.retrievalMS = retrievalMS
        self.inferenceMS = inferenceMS
        self.topKBMatchID = topKBMatchID
        self.topKBScore = topKBScore
        self.kbEntriesScanned = kbEntriesScanned
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.chatMode = chatMode
        self.chatModeConfidence = chatModeConfidence
        self.chatModeRuntimeMS = chatModeRuntimeMS
        self.extraction = extraction
        self.toolSelectionReasoning = toolSelectionReasoning
        self.toolSelectionConfidence = toolSelectionConfidence
        self.understanding = understanding
        self.telcoUnderstanding = telcoUnderstanding
        self.telcoUnderstandingMS = telcoUnderstandingMS
        self.routePolicyMS = routePolicyMS
        self.composerMS = composerMS
        self.totalWallMS = totalWallMS
        self.composerRoute = composerRoute
        self.composerCitedPageID = composerCitedPageID
        self.composerRenderedLinkID = composerRenderedLinkID
        self.composerConfirmationShown = composerConfirmationShown
        self.reuseActiveEvidence = reuseActiveEvidence
        self.policyReason = policyReason
        self.stateOperation = stateOperation
        self.stateOperationReason = stateOperationReason
    }

    public var totalMS: Int { totalWallMS ?? ((retrievalMS ?? 0) + inferenceMS) }

    public var customerVisibleMS: Int {
        switch surface {
        case .onDeviceRAG:
            return totalWallMS ?? ((chatModeRuntimeMS ?? 0) + (retrievalMS ?? 0) + inferenceMS)
        case .tool, .visionPack, .voicePack:
            return totalMS
        }
    }
}

/// Compact view-ready projection of a `RoutingDecision`.
public struct RoutingSummary: Equatable {
    public let path: RoutingPath
    public let toolIntent: ToolIntent?
    public let containsPII: Bool
    public let confidence: Double?

    public init(
        path: RoutingPath,
        toolIntent: ToolIntent?,
        containsPII: Bool,
        confidence: Double? = nil
    ) {
        self.path = path
        self.toolIntent = toolIntent
        self.containsPII = containsPII
        self.confidence = confidence
    }
}

/// What the on-device model understood about the user's request. Replaces
/// the fake-execution flow — the demo value is proving the model picked the
/// right tool with the right arguments, not pretending to restart a router.
///
/// `isCompoundAttachment` distinguishes two visual modes per the
/// ADR-022 compound-response review (2026-05-26):
///   - false (default) — this is the PRIMARY response. The chat bubble
///     above the card carries the short framing ("I'll restart your
///     router."). The card itself reads "Tool selected" — implied
///     primary CTA.
///   - true            — this is a SECONDARY affordance attached to a
///     RAG / unknown-feature / clarification reply whose primary
///     content is a how-to article. The card renders "Want me to do
///     this for you?" above the tool header so the user reads it as
///     a one-tap alternative to following the article themselves.
///
/// Pure visual signal. Confirm/Decline plumbing is identical either way.
public struct ToolDecision: Equatable {
    public let intent: ToolIntent
    public let toolID: String
    public let displayName: String
    public let icon: String
    public let description: String
    public let arguments: [ToolDecisionArgument]
    public let confidence: Double
    public let reasoning: String?
    public let requiresConfirmation: Bool
    public let isDestructive: Bool
    public var isCompoundAttachment: Bool

    public init(
        intent: ToolIntent,
        toolID: String,
        displayName: String,
        icon: String,
        description: String,
        arguments: [ToolDecisionArgument],
        confidence: Double,
        reasoning: String?,
        requiresConfirmation: Bool,
        isDestructive: Bool,
        isCompoundAttachment: Bool = false
    ) {
        self.intent = intent
        self.toolID = toolID
        self.displayName = displayName
        self.icon = icon
        self.description = description
        self.arguments = arguments
        self.confidence = confidence
        self.reasoning = reasoning
        self.requiresConfirmation = requiresConfirmation
        self.isDestructive = isDestructive
        self.isCompoundAttachment = isCompoundAttachment
    }
}

/// A single extracted argument, formatted for display.
public struct ToolDecisionArgument: Identifiable, Equatable {
    public var id: String { "\(label):\(value)" }
    public let label: String   // "Target device"
    public let value: String   // "son's tablet"
}
