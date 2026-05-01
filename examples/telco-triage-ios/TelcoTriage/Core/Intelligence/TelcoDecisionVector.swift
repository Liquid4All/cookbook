import Foundation

/// Output of the telco multi-head classifier.
///
/// Holds two head sets that share one shared-adapter forward pass:
/// 1. Phase-1 heads (`mode`, `kbEntry`, `tool`) — the working triad
///    that drives the existing 4-mode chat dispatch.
/// 2. ADR-015 Phase-2 heads (`supportIntent`, `routingLane`, …) —
///    9 sequence-level heads producing a structured `TelcoLaneDecision`
///    over {local_answer, local_tool, cloud_assist, human_escalation,
///    blocked}. Populated only when the ADR-015 stack is bundled;
///    `.unavailable` otherwise.
public struct TelcoDecisionVector: Sendable {
    public let timestamp: Date
    public let inputHash: Int

    // Phase-1 heads (3-head triad)
    public let mode: TelcoHeadResult
    public let kbEntry: TelcoHeadResult
    public let tool: TelcoHeadResult

    // ADR-015 Phase-2 heads (9 sequence heads). Default to `.unavailable`
    // / empty so existing call sites that construct a vector without
    // ADR-015 heads keep compiling.
    public let supportIntent: TelcoHeadResult
    public let issueComplexity: TelcoHeadResult
    public let routingLane: TelcoHeadResult
    public let cloudRequirements: TelcoMultiLabelHeadResult
    public let requiredTool: TelcoHeadResult
    public let customerEscalationRisk: TelcoHeadResult
    public let piiRisk: TelcoHeadResult
    public let transcriptQuality: TelcoHeadResult
    public let slotCompleteness: TelcoMultiLabelHeadResult

    public let inferenceMode: TelcoClassifierInferenceMode
    public let forwardPassMs: Double
    public let headProjectionMs: Double

    public var totalMs: Double { forwardPassMs + headProjectionMs }

    /// True when at least the routing-lane head produced a real prediction.
    /// Drives whether `TelcoDecisionRouter.routeLane(_:)` should be used.
    public var hasADR015Heads: Bool {
        routingLane.label != "unavailable"
    }

    public init(
        timestamp: Date = Date(),
        inputHash: Int,
        mode: TelcoHeadResult,
        kbEntry: TelcoHeadResult,
        tool: TelcoHeadResult,
        supportIntent: TelcoHeadResult = .unavailable,
        issueComplexity: TelcoHeadResult = .unavailable,
        routingLane: TelcoHeadResult = .unavailable,
        cloudRequirements: TelcoMultiLabelHeadResult = .unavailable,
        requiredTool: TelcoHeadResult = .unavailable,
        customerEscalationRisk: TelcoHeadResult = .unavailable,
        piiRisk: TelcoHeadResult = .unavailable,
        transcriptQuality: TelcoHeadResult = .unavailable,
        slotCompleteness: TelcoMultiLabelHeadResult = .unavailable,
        inferenceMode: TelcoClassifierInferenceMode,
        forwardPassMs: Double,
        headProjectionMs: Double
    ) {
        self.timestamp = timestamp
        self.inputHash = inputHash
        self.mode = mode
        self.kbEntry = kbEntry
        self.tool = tool
        self.supportIntent = supportIntent
        self.issueComplexity = issueComplexity
        self.routingLane = routingLane
        self.cloudRequirements = cloudRequirements
        self.requiredTool = requiredTool
        self.customerEscalationRisk = customerEscalationRisk
        self.piiRisk = piiRisk
        self.transcriptQuality = transcriptQuality
        self.slotCompleteness = slotCompleteness
        self.inferenceMode = inferenceMode
        self.forwardPassMs = forwardPassMs
        self.headProjectionMs = headProjectionMs
    }
}

public enum TelcoClassifierInferenceMode: String, Sendable {
    case sharedAdapter = "shared_adapter"
    case pairedAdapters = "paired_adapters"
}

/// Single-label classifier head output (softmax).
public struct TelcoHeadResult: Sendable, Equatable {
    public let label: String
    public let confidence: Double
    public let probabilities: [Float]
    public let labelIndex: Int

    public init(
        label: String,
        confidence: Double,
        probabilities: [Float] = [],
        labelIndex: Int = 0
    ) {
        self.label = label
        self.confidence = confidence
        self.probabilities = probabilities
        self.labelIndex = labelIndex
    }

    public static func from(_ prediction: ClassifierHead.Prediction) -> TelcoHeadResult {
        TelcoHeadResult(
            label: prediction.label,
            confidence: Double(prediction.confidence),
            probabilities: prediction.probabilities,
            labelIndex: prediction.labelIndex
        )
    }

    public static let unavailable = TelcoHeadResult(label: "unavailable", confidence: 0)
}

/// Multi-label classifier head output (sigmoid per flag).
public struct TelcoMultiLabelHeadResult: Sendable, Equatable {
    public let activeLabels: [String]
    public let probabilities: [Float]

    public init(activeLabels: [String] = [], probabilities: [Float] = []) {
        self.activeLabels = activeLabels
        self.probabilities = probabilities
    }

    public static func from(_ prediction: ClassifierHead.MultiLabelPrediction) -> TelcoMultiLabelHeadResult {
        TelcoMultiLabelHeadResult(
            activeLabels: prediction.activeLabels,
            probabilities: prediction.probabilities
        )
    }

    public static let unavailable = TelcoMultiLabelHeadResult()
}
