import Foundation

/// Per-turn projection of the ADR-015 9-head telco classifier output,
/// shaped for the RBC-style pipeline card: a single `classifyAll` row
/// summarizing the multi-head pass, followed by an optional downstream
/// row that names the next stage the lane decision triggered (KB
/// retrieval, tool call, cloud payload, etc.). The viewer should learn
/// what happened in two glances, not nine.
public struct TelcoPipelineTrace: Equatable {
    /// One-line headline shown in the card header — derived from the
    /// support-intent head (e.g., `outage`, `troubleshooting`).
    public let intent: String
    public let intentConfidence: Double
    public let modelName: String
    public let inferenceMode: String
    public let totalLatencyMs: Double

    /// The single classifier row — collapses all 9 heads into one
    /// "classifyAll (9 heads)" stage with a compact summary.
    public let primaryStep: Step

    /// The next pipeline stage triggered by the lane decision. Nil
    /// when no follow-up stage is meaningful (e.g., a hard block).
    public let downstreamStep: Step?

    /// One-line answer/target summary shown in the header grid
    /// (matches the `Answer` and `Target` rows in the RBC card).
    public let answerSummary: String
    public let target: String?

    /// Lane decision footer (`Reason` line in the RBC card).
    public let laneLabel: String?
    public let laneReason: String?

    public init(
        intent: String,
        intentConfidence: Double,
        modelName: String,
        inferenceMode: String,
        totalLatencyMs: Double,
        primaryStep: Step,
        downstreamStep: Step?,
        answerSummary: String,
        target: String?,
        laneLabel: String?,
        laneReason: String?
    ) {
        self.intent = intent
        self.intentConfidence = intentConfidence
        self.modelName = modelName
        self.inferenceMode = inferenceMode
        self.totalLatencyMs = totalLatencyMs
        self.primaryStep = primaryStep
        self.downstreamStep = downstreamStep
        self.answerSummary = answerSummary
        self.target = target
        self.laneLabel = laneLabel
        self.laneReason = laneReason
    }

    public struct Step: Equatable, Identifiable {
        public let id: String
        public let title: String
        public let detail: String
        public let modelTag: String
        public let confidence: Double?
        public let latencyMs: Double

        public init(
            id: String,
            title: String,
            detail: String,
            modelTag: String,
            confidence: Double?,
            latencyMs: Double
        ) {
            self.id = id
            self.title = title
            self.detail = detail
            self.modelTag = modelTag
            self.confidence = confidence
            self.latencyMs = latencyMs
        }
    }
}

extension TelcoPipelineTrace {
    /// Build a trace from a `TelcoDecisionVector` + lane. `nil` when the
    /// 9-head ADR-015 stack didn't run (paired-adapter fallback or the
    /// vision/voice path).
    public static func from(
        vector: TelcoDecisionVector,
        lane: TelcoLaneDecision?
    ) -> TelcoPipelineTrace? {
        guard vector.hasADR015Heads else { return nil }

        let modelTag: String
        switch vector.inferenceMode {
        case .sharedAdapter:  modelTag = "LFM2.5-350M (shared-clf)"
        case .pairedAdapters: modelTag = "LFM2.5-350M (paired)"
        }

        let primary = TelcoPipelineTrace.Step(
            id: "classify_all",
            title: "classifyAll (9 heads)",
            detail: classifierSummary(vector),
            modelTag: modelTag,
            confidence: vector.supportIntent.confidence,
            latencyMs: vector.totalMs
        )

        let downstream = downstreamStep(for: lane, vector: vector)
        let (answer, target) = answerAndTarget(for: lane, vector: vector)
        let (laneLabel, laneReason) = summarize(lane: lane)

        return TelcoPipelineTrace(
            intent: TelcoLabelDisplay.text(vector.supportIntent.label),
            intentConfidence: vector.supportIntent.confidence,
            modelName: "LFM2.5-350M",
            inferenceMode: vector.inferenceMode.rawValue,
            totalLatencyMs: vector.totalMs,
            primaryStep: primary,
            downstreamStep: downstream,
            answerSummary: answer,
            target: target,
            laneLabel: laneLabel,
            laneReason: laneReason
        )
    }

    /// Compact one-line summary of the 9-head output. Surfaces the
    /// fields that drive routing (intent, complexity, tool, PII,
    /// escalation) and skips the rest. The viewer can correlate any of
    /// these against the labels they expect — a mismatch signals a
    /// classifier error that won't be hidden by the UI.
    private static func classifierSummary(_ vector: TelcoDecisionVector) -> String {
        var parts: [String] = []
        parts.append("Intent: \(TelcoLabelDisplay.text(vector.supportIntent.label))")
        parts.append("Complexity: \(TelcoLabelDisplay.text(vector.issueComplexity.label))")
        parts.append("Tool: \(TelcoLabelDisplay.text(vector.requiredTool.label))")
        if vector.piiRisk.label != "safe" {
            parts.append("PII: \(TelcoLabelDisplay.text(vector.piiRisk.label))")
        }
        if vector.customerEscalationRisk.label != "low" {
            parts.append("Escalation: \(TelcoLabelDisplay.text(vector.customerEscalationRisk.label))")
        }
        let missing = vector.slotCompleteness.activeLabels
        if !missing.isEmpty {
            parts.append("Missing \(missing.count) slot\(missing.count == 1 ? "" : "s")")
        }
        return parts.joined(separator: " · ")
    }

    /// Pick the next pipeline row to surface based on the lane the
    /// router selected. Mirrors the RBC card: one row per logical
    /// stage, not one row per head.
    private static func downstreamStep(
        for lane: TelcoLaneDecision?,
        vector: TelcoDecisionVector
    ) -> TelcoPipelineTrace.Step? {
        guard let lane else { return nil }
        switch lane {
        case .localAnswer:
            return TelcoPipelineTrace.Step(
                id: "kb_retrieval",
                title: "KB Retrieval",
                detail: "answerable from on-device knowledge base",
                modelTag: "on-device · no LFM call",
                confidence: nil,
                latencyMs: 0
            )
        case .localTool(let tool, _, _):
            return TelcoPipelineTrace.Step(
                id: "tool_call",
                title: "Tool Call",
                detail: TelcoLabelDisplay.text(tool.rawValue),
                modelTag: "on-device tool",
                confidence: vector.requiredTool.confidence,
                latencyMs: 0
            )
        case .cloudAssist(_, let reqs, let missing, let pii, _):
            var parts: [String] = []
            if !reqs.isEmpty {
                parts.append("Needs: \(TelcoLabelDisplay.list(reqs.map(\.rawValue)))")
            }
            if !missing.isEmpty {
                parts.append("\(missing.count) missing slot\(missing.count == 1 ? "" : "s")")
            }
            parts.append("PII: \(TelcoLabelDisplay.text(pii.rawValue))")
            return TelcoPipelineTrace.Step(
                id: "cloud_payload",
                title: "Cloud Payload",
                detail: parts.joined(separator: " · "),
                modelTag: "egress prepared, not sent",
                confidence: nil,
                latencyMs: 0
            )
        case .humanEscalation:
            return TelcoPipelineTrace.Step(
                id: "human_handoff",
                title: "Human Handoff",
                detail: "hand off to live agent",
                modelTag: "no LFM call",
                confidence: vector.customerEscalationRisk.confidence,
                latencyMs: 0
            )
        case .blocked(let reason):
            return TelcoPipelineTrace.Step(
                id: "blocked",
                title: "Refusal",
                detail: "blocked: \(reason)",
                modelTag: "policy guard",
                confidence: nil,
                latencyMs: 0
            )
        case .degraded:
            return nil
        }
    }

    private static func answerAndTarget(
        for lane: TelcoLaneDecision?,
        vector: TelcoDecisionVector
    ) -> (String, String?) {
        guard let lane else {
            return ("multi-head classification", nil)
        }
        switch lane {
        case .localAnswer:
            return ("On-device KB grounded answer", "Local answer")
        case .localTool(let tool, _, _):
            return ("On-device tool execution",
                    "Local tool · \(TelcoLabelDisplay.text(tool.rawValue))")
        case .cloudAssist(_, _, _, _, _):
            return ("Cloud assist payload prepared", "Cloud assist")
        case .humanEscalation:
            return ("Human agent handoff", "Human escalation")
        case .blocked:
            return ("Refused locally", "Blocked")
        case .degraded:
            return ("Degraded — deterministic fallback", "Degraded")
        }
    }

    private static func summarize(lane: TelcoLaneDecision?) -> (String?, String?) {
        guard let lane else { return (nil, nil) }
        switch lane {
        case .localAnswer(_, let reason):
            return ("Local answer", reason)
        case .localTool(let tool, _, let reason):
            return ("Local tool · \(TelcoLabelDisplay.text(tool.rawValue))", reason)
        case .cloudAssist(_, let reqs, _, _, let reason):
            return ("Cloud assist (\(reqs.count) requirement\(reqs.count == 1 ? "" : "s"))", reason)
        case .humanEscalation(_, let reason):
            return ("Human escalation", reason)
        case .blocked(let reason):
            return ("Blocked", String(describing: reason))
        case .degraded(let reason):
            return ("Degraded", reason)
        }
    }
}
