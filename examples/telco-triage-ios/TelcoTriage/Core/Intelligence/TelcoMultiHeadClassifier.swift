import Foundation
import os.log

public protocol TelcoMultiHeadClassifying: Sendable {
    func classifyAll(query: String) async throws -> TelcoDecisionVector
}

public protocol TelcoDecisionEngine: Sendable {
    func decide(
        query: String,
        kb: [KBEntry],
        extraction: ExtractionResult,
        availableTools: [Tool]
    ) async -> TelcoDecisionResult
}

/// Combined output of `TelcoDecisionEngine.decide(...)` — the routing
/// decision the chat dispatcher acts on, plus the raw 9-head vector
/// for the engineering-mode pipeline trace UI. The vector is `nil` only
/// when the decision engine fails inside its catch block.
public struct TelcoDecisionResult: Sendable {
    public let decision: TelcoRoutingDecision
    public let vector: TelcoDecisionVector?
    public let lane: TelcoLaneDecision?

    public init(
        decision: TelcoRoutingDecision,
        vector: TelcoDecisionVector?,
        lane: TelcoLaneDecision?
    ) {
        self.decision = decision
        self.vector = vector
        self.lane = lane
    }
}

/// Telco variant of the RBC shared multi-head classifier.
///
/// Two head sets coexist:
/// 1. Phase-1 triad (`chatModeHead`, `kbEntryHead`, `toolHead`) drives the
///    existing 4-mode chat dispatch.
/// 2. ADR-015 Phase-2 (9 heads in `adr015Heads`, optional) produces the
///    structured `TelcoLaneDecision` over five lanes.
///
/// Preferred path: one shared classification LoRA (`telco-shared-clf-v1`)
/// drives every head from one embedding. Transitional fallback: existing
/// paired classification adapters run each Phase-1 head against its
/// trained hidden-state space; ADR-015 heads stay `.unavailable` in that
/// mode because they require the shared adapter.
public final class TelcoMultiHeadClassifier: TelcoMultiHeadClassifying, @unchecked Sendable {
    private let backend: LlamaBackend
    private let heads: HeadInventory
    private let adapters: AdapterInventory
    private let logger = Logger(subsystem: "ai.liquid.demos.telcotriage", category: "TelcoMultiHead")

    public struct HeadInventory: Sendable {
        // Phase-1 triad
        public let chatModeHead: ClassifierHead
        public let kbEntryHead: ClassifierHead
        public let toolHead: ClassifierHead
        // ADR-015 Phase-2 heads (optional; if any nil, ADR-015 lane is disabled)
        public let adr015: ADR015HeadInventory?

        public init(
            chatModeHead: ClassifierHead,
            kbEntryHead: ClassifierHead,
            toolHead: ClassifierHead,
            adr015: ADR015HeadInventory? = nil
        ) {
            self.chatModeHead = chatModeHead
            self.kbEntryHead = kbEntryHead
            self.toolHead = toolHead
            self.adr015 = adr015
        }
    }

    public struct ADR015HeadInventory: Sendable {
        public let supportIntent: ClassifierHead
        public let issueComplexity: ClassifierHead
        public let routingLane: ClassifierHead
        public let cloudRequirements: ClassifierHead
        public let requiredTool: ClassifierHead
        public let customerEscalationRisk: ClassifierHead
        public let piiRisk: ClassifierHead
        public let transcriptQuality: ClassifierHead
        public let slotCompleteness: ClassifierHead

        public init(
            supportIntent: ClassifierHead,
            issueComplexity: ClassifierHead,
            routingLane: ClassifierHead,
            cloudRequirements: ClassifierHead,
            requiredTool: ClassifierHead,
            customerEscalationRisk: ClassifierHead,
            piiRisk: ClassifierHead,
            transcriptQuality: ClassifierHead,
            slotCompleteness: ClassifierHead
        ) {
            self.supportIntent = supportIntent
            self.issueComplexity = issueComplexity
            self.routingLane = routingLane
            self.cloudRequirements = cloudRequirements
            self.requiredTool = requiredTool
            self.customerEscalationRisk = customerEscalationRisk
            self.piiRisk = piiRisk
            self.transcriptQuality = transcriptQuality
            self.slotCompleteness = slotCompleteness
        }
    }

    public struct AdapterInventory: Sendable {
        public let sharedAdapterPath: String?
        public let chatModeAdapterPath: String?
        public let kbExtractAdapterPath: String?
        public let toolSelectorAdapterPath: String?

        public init(
            sharedAdapterPath: String?,
            chatModeAdapterPath: String?,
            kbExtractAdapterPath: String?,
            toolSelectorAdapterPath: String?
        ) {
            self.sharedAdapterPath = sharedAdapterPath
            self.chatModeAdapterPath = chatModeAdapterPath
            self.kbExtractAdapterPath = kbExtractAdapterPath
            self.toolSelectorAdapterPath = toolSelectorAdapterPath
        }
    }

    public init(
        backend: LlamaBackend,
        heads: HeadInventory,
        adapters: AdapterInventory
    ) {
        self.backend = backend
        self.heads = heads
        self.adapters = adapters
    }

    public var inferenceMode: TelcoClassifierInferenceMode {
        adapters.sharedAdapterPath == nil ? .pairedAdapters : .sharedAdapter
    }

    public var hasADR015Heads: Bool { heads.adr015 != nil }

    public func classifyAll(query: String) async throws -> TelcoDecisionVector {
        if let sharedAdapterPath = adapters.sharedAdapterPath {
            return try await classifyWithSharedAdapter(query: query, adapterPath: sharedAdapterPath)
        }
        return try await classifyWithPairedAdapters(query: query)
    }

    private func classifyWithSharedAdapter(
        query: String,
        adapterPath: String
    ) async throws -> TelcoDecisionVector {
        let forwardStart = CFAbsoluteTimeGetCurrent()
        try await backend.setAdapter(path: adapterPath, scale: 1.0)
        let embedding = try await backend.embeddings(prompt: query, clearCache: true)
        let forwardMs = (CFAbsoluteTimeGetCurrent() - forwardStart) * 1000

        let headStart = CFAbsoluteTimeGetCurrent()

        // Phase-1 triad
        let mode = TelcoHeadResult.from(heads.chatModeHead.classify(embedding))
        let kbEntry = TelcoHeadResult.from(heads.kbEntryHead.classify(embedding))
        let tool = TelcoHeadResult.from(heads.toolHead.classify(embedding))

        // ADR-015 Phase-2 (optional)
        var supportIntent: TelcoHeadResult = .unavailable
        var issueComplexity: TelcoHeadResult = .unavailable
        var routingLane: TelcoHeadResult = .unavailable
        var cloudRequirements: TelcoMultiLabelHeadResult = .unavailable
        var requiredTool: TelcoHeadResult = .unavailable
        var customerEscalationRisk: TelcoHeadResult = .unavailable
        var piiRisk: TelcoHeadResult = .unavailable
        var transcriptQuality: TelcoHeadResult = .unavailable
        var slotCompleteness: TelcoMultiLabelHeadResult = .unavailable

        if let adr = heads.adr015 {
            supportIntent = .from(adr.supportIntent.classify(embedding))
            issueComplexity = .from(adr.issueComplexity.classify(embedding))
            routingLane = .from(adr.routingLane.classify(embedding))
            cloudRequirements = .from(adr.cloudRequirements.classifyMultiLabel(embedding))
            requiredTool = .from(adr.requiredTool.classify(embedding))
            customerEscalationRisk = .from(adr.customerEscalationRisk.classify(embedding))
            piiRisk = .from(adr.piiRisk.classify(embedding))
            transcriptQuality = .from(adr.transcriptQuality.classify(embedding))
            slotCompleteness = .from(adr.slotCompleteness.classifyMultiLabel(embedding))
        }

        let vector = TelcoDecisionVector(
            inputHash: query.hashValue,
            mode: mode,
            kbEntry: kbEntry,
            tool: tool,
            supportIntent: supportIntent,
            issueComplexity: issueComplexity,
            routingLane: routingLane,
            cloudRequirements: cloudRequirements,
            requiredTool: requiredTool,
            customerEscalationRisk: customerEscalationRisk,
            piiRisk: piiRisk,
            transcriptQuality: transcriptQuality,
            slotCompleteness: slotCompleteness,
            inferenceMode: .sharedAdapter,
            forwardPassMs: forwardMs,
            headProjectionMs: (CFAbsoluteTimeGetCurrent() - headStart) * 1000
        )
        log(vector)
        return vector
    }

    private func classifyWithPairedAdapters(query: String) async throws -> TelcoDecisionVector {
        guard let chatModeAdapterPath = adapters.chatModeAdapterPath,
              let kbExtractAdapterPath = adapters.kbExtractAdapterPath,
              let toolSelectorAdapterPath = adapters.toolSelectorAdapterPath
        else {
            throw TelcoMultiHeadError.missingAdapters
        }

        var forwardMs = 0.0
        var headMs = 0.0

        let mode = try await classify(
            query: query,
            adapterPath: chatModeAdapterPath,
            head: heads.chatModeHead,
            forwardMs: &forwardMs,
            headMs: &headMs
        )
        let kbEntry = try await classify(
            query: query,
            adapterPath: kbExtractAdapterPath,
            head: heads.kbEntryHead,
            forwardMs: &forwardMs,
            headMs: &headMs
        )
        let tool = try await classify(
            query: query,
            adapterPath: toolSelectorAdapterPath,
            head: heads.toolHead,
            forwardMs: &forwardMs,
            headMs: &headMs
        )

        // Paired-adapter mode predates ADR-015 — every Phase-2 head's
        // classifier was trained against the shared adapter's hidden-state
        // distribution, not the per-task adapter's. Producing a value here
        // would be silently wrong, so we surface `.unavailable` and let
        // the lane router degrade gracefully (deterministic fallback).
        let vector = TelcoDecisionVector(
            inputHash: query.hashValue,
            mode: mode,
            kbEntry: kbEntry,
            tool: tool,
            inferenceMode: .pairedAdapters,
            forwardPassMs: forwardMs,
            headProjectionMs: headMs
        )
        log(vector)
        return vector
    }

    private func classify(
        query: String,
        adapterPath: String,
        head: ClassifierHead,
        forwardMs: inout Double,
        headMs: inout Double
    ) async throws -> TelcoHeadResult {
        let forwardStart = CFAbsoluteTimeGetCurrent()
        try await backend.setAdapter(path: adapterPath, scale: 1.0)
        let embedding = try await backend.embeddings(prompt: query, clearCache: true)
        forwardMs += (CFAbsoluteTimeGetCurrent() - forwardStart) * 1000

        let headStart = CFAbsoluteTimeGetCurrent()
        let result = TelcoHeadResult.from(head.classify(embedding))
        headMs += (CFAbsoluteTimeGetCurrent() - headStart) * 1000
        return result
    }

    private func log(_ vector: TelcoDecisionVector) {
        let adr015 = vector.hasADR015Heads
            ? " lane=\(vector.routingLane.label) sint=\(vector.supportIntent.label)"
            : ""
        logger.info("classifyAll mode=\(vector.mode.label, privacy: .public) tool=\(vector.tool.label, privacy: .public) kb=\(vector.kbEntry.label, privacy: .public)\(adr015, privacy: .public) total=\(String(format: "%.0f", vector.totalMs), privacy: .public)ms inference=\(vector.inferenceMode.rawValue, privacy: .public)")
    }
}

public final class MultiHeadTelcoDecisionEngine: TelcoDecisionEngine, @unchecked Sendable {
    private let classifier: TelcoMultiHeadClassifying

    public init(classifier: TelcoMultiHeadClassifying) {
        self.classifier = classifier
    }

    public func decide(
        query: String,
        kb: [KBEntry],
        extraction: ExtractionResult,
        availableTools: [Tool]
    ) async -> TelcoDecisionResult {
        do {
            let vector = try await classifier.classifyAll(query: query)

            // ADR-015 Phase 2: compute the lane decision when the 9-head
            // stack is loaded. Chat dispatch still uses the legacy 4-mode
            // TelcoRoutingDecision; the lane is surfaced through the
            // engineering-mode pipeline trace.
            var lane: TelcoLaneDecision?
            if vector.hasADR015Heads {
                let resolved = TelcoLaneRouter.route(vector)
                lane = resolved
                AppLog.intelligence.info("ADR-015 lane=\(Self.laneLabel(resolved), privacy: .public) intent=\(vector.supportIntent.label, privacy: .public)/\(String(format: "%.2f", vector.supportIntent.confidence), privacy: .public) escalation=\(vector.customerEscalationRisk.label, privacy: .public) pii=\(vector.piiRisk.label, privacy: .public)")
            }

            let decision = TelcoDecisionRouter.route(
                vector,
                kb: kb,
                availableTools: availableTools,
                extraction: extraction
            )
            return TelcoDecisionResult(decision: decision, vector: vector, lane: lane)
        } catch {
            AppLog.intelligence.error("telco decision engine failed: \(error.localizedDescription, privacy: .public)")
            return TelcoDecisionResult(
                decision: .outOfScope(modePrediction: ChatModePrediction(
                    mode: .outOfScope,
                    confidence: 0,
                    reasoning: "multi-head inference error",
                    runtimeMS: 0
                )),
                vector: nil,
                lane: nil
            )
        }
    }

    private static func laneLabel(_ lane: TelcoLaneDecision) -> String {
        switch lane {
        case .localAnswer: return "local_answer"
        case .localTool(let tool, _, _): return "local_tool/\(tool.rawValue)"
        case .cloudAssist(_, let reqs, _, _, _): return "cloud_assist/\(reqs.count)reqs"
        case .humanEscalation: return "human_escalation"
        case .blocked: return "blocked"
        case .degraded: return "degraded"
        }
    }
}

public enum TelcoMultiHeadError: Error, LocalizedError {
    case missingAdapters

    public var errorDescription: String? {
        switch self {
        case .missingAdapters:
            return "Telco multi-head classifier has no shared adapter and incomplete paired adapters"
        }
    }
}
