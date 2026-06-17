import Foundation
import os.log

/// Async-Sendable protocol the classifier exposes. Lives behind a
/// protocol so tests can substitute stubs without spinning up llama.cpp.
///
/// **Single-turn** — multi-turn dependencies are handled by the
/// pairwise relational heads (ADR-024 `RelationalHeadsStrategy`), not
/// by stuffing history into this entry point. ADR-023 Phase 1's
/// `classify(query:history:)` overload was retired by ADR-024 after
/// the on-device parse-failure surface was diagnosed.
public protocol QueryUnderstandingClassifying: Sendable {
    func classify(query: String) async throws -> QueryUnderstanding
}

/// Errors specific to v2 understanding-layer classification.
public enum QueryUnderstandingError: Error, LocalizedError {
    case missingArtifact(name: String)
    case backendFailure(underlying: Error)
    case headLoadFailure(task: String, underlying: Error)

    public var errorDescription: String? {
        switch self {
        case .missingArtifact(let name):
            return "Understanding layer artifact missing: \(name)"
        case .backendFailure(let err):
            return "Understanding backend failure: \(err.localizedDescription)"
        case .headLoadFailure(let task, let err):
            return "Understanding head '\(task)' failed to load: \(err.localizedDescription)"
        }
    }
}

/// Layer 1 of the ADR-022 architecture: one classifier producing the
/// full `QueryUnderstanding` vector per query. Hides the choice of
/// strategy behind a single `classify(query:)` entry point so the
/// caller (ChatViewModel) doesn't branch on whether v2 is bundled.
///
/// **Strategy pattern**:
///
///  - `SharedBackboneStrategy` — one `setAdapter(telco-shared-clf-v2)`
///    swap + one `embeddings()` call + N head projections from the same
///    hidden state. The architectural target per ADR-022 §3 ("the
///    shared-backbone multi-head pattern is the most underutilized
///    leverage in our stack").
///  - `CompositeFallbackStrategy` — wraps the existing
///    `LFMChatModeRouter` (generative chat_mode) + `TelcoStageAClassifier`
///    (topic_gate + refusal_flags, private adapters). Used when v2
///    isn't bundled. Costs 1 generative call + 2 forward passes.
///    `emotionalState` and `slotCompleteness` are nil in this mode —
///    the workflow handles the absence gracefully.
///  - `UnavailableStrategy` — neither path is bundled (degraded build).
///    Returns an empty `QueryUnderstanding` with `strategy = .unavailable`.
///    The router's safe defaults handle this case explicitly.
///
/// The factory `bundled(...)` picks the best available strategy at
/// boot time. The choice is stable for the app lifetime — we never
/// flip strategies mid-session.
public final class QueryUnderstandingClassifier: QueryUnderstandingClassifying, @unchecked Sendable {

    private let strategy: any UnderstandingStrategy
    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "QueryUnderstanding"
    )

    public init(strategy: any UnderstandingStrategy) {
        self.strategy = strategy
    }

    public func classify(query: String) async throws -> QueryUnderstanding {
        let t0 = CFAbsoluteTimeGetCurrent()
        let understanding = try await strategy.run(query: query)
        let totalMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        // Re-stamp totalMs at the OUTER boundary so the trace surfaces
        // the wall-clock the chat saw, not the strategy's internal
        // estimate (which may exclude actor-hop overhead).
        let rebound = QueryUnderstanding(
            chatMode: understanding.chatMode,
            topicGate: understanding.topicGate,
            refusalFlags: understanding.refusalFlags,
            emotionalState: understanding.emotionalState,
            slotCompleteness: understanding.slotCompleteness,
            // ADR-024 relational signals propagate through the
            // re-stamp boundary alongside the single-input vector.
            turnRelationship: understanding.turnRelationship,
            slotAlignment: understanding.slotAlignment,
            stanceChange: understanding.stanceChange,
            totalMs: totalMs,
            strategy: understanding.strategy
        )

        logger.info(
            "understanding strategy=\(rebound.strategy.rawValue, privacy: .public) mode=\(rebound.chatMode?.mode.rawValue ?? "-", privacy: .public) topic=\(self.topicDescription(rebound.topicGate), privacy: .public) flags=\(self.flagsDescription(rebound.refusalFlags), privacy: .public) emo=\(rebound.emotionalState?.value.wireName ?? "-", privacy: .public) total=\(String(format: "%.0f", rebound.totalMs), privacy: .public)ms"
        )
        return rebound
    }

    private func topicDescription(_ outcome: TopicGateOutcome?) -> String {
        guard let outcome else { return "-" }
        switch outcome.value {
        case .inScope:      return "in_scope"
        case .outOfScope:   return "out_of_scope"
        case .greeting:     return "greeting"
        }
    }

    private func flagsDescription(_ outcome: RefusalFlagsOutcome?) -> String {
        guard let outcome else { return "-" }
        var parts: [String] = []
        if outcome.value.hasRagAnswer    { parts.append("rag") }
        if outcome.value.navigationOnly  { parts.append("nav") }
        if outcome.value.liveAgentTrigger { parts.append("agent") }
        return parts.isEmpty ? "none" : parts.joined(separator: ",")
    }

    // MARK: - Factory

    /// Build the best-available classifier given the current bundle.
    ///
    /// Selection order:
    ///  1. v2 shared backbone if `telco-shared-clf-v2` + at least one
    ///     head are bundled.
    ///  2. Composite (LFMChatModeRouter + TelcoStageAClassifier) if
    ///     PR #30 artifacts are bundled.
    ///  3. Unavailable (caller falls back to non-classified routing).
    public static func bundled(
        backend: LlamaBackend,
        chatModeRouter: ChatModeRouter,
        stageA: TelcoStageAClassifying?,
        bundle: Bundle = .main
    ) -> QueryUnderstandingClassifier {
        if let shared = try? SharedBackboneStrategy.bundled(backend: backend, bundle: bundle) {
            return QueryUnderstandingClassifier(strategy: shared)
        }
        if let stageA {
            let composite = CompositeFallbackStrategy(
                chatModeRouter: chatModeRouter,
                stageA: stageA
            )
            return QueryUnderstandingClassifier(strategy: composite)
        }
        return QueryUnderstandingClassifier(strategy: UnavailableStrategy())
    }
}

// MARK: - Strategy contract

/// Internal contract every strategy satisfies. Hidden from callers —
/// they only see `QueryUnderstandingClassifier.classify(query:)`.
///
/// Single-turn only — ADR-024 retired the history-aware overload
/// that ADR-023 Phase 1 once added (relational signal flows through
/// `RelationalHeadsStrategy` on cached hidden states instead).
public protocol UnderstandingStrategy: Sendable {
    func run(query: String) async throws -> QueryUnderstanding
}

// MARK: - Shared-backbone (v2)

/// **The ADR-022 §4.3 architectural target.** One adapter swap. One
/// forward pass. N head projections.
///
/// At runtime: `setAdapter(v2)` → `embeddings(query)` → for each
/// loaded head, project the hidden state into logits, decode to the
/// typed outcome. Total wall-clock ≈ 150 ms (dominated by the forward
/// pass; each head is <1 ms via cblas_sgemv per `ClassifierHead`).
///
/// **Partial-head tolerance**: we ship heads in waves. The strategy
/// loads whatever heads ARE bundled at construction time and reports
/// the rest as nil in `QueryUnderstanding`. Routing handles nil
/// fields per §4.3 design principle #1.
public final class SharedBackboneStrategy: UnderstandingStrategy, @unchecked Sendable {
    private let backend: LlamaBackend
    private let adapterPath: String
    private let chatModeHead: ClassifierHead?
    private let topicGateHead: ClassifierHead?
    private let refusalFlagsHead: ClassifierHead?
    private let emotionalStateHead: ClassifierHead?
    private let slotCompletenessHead: ClassifierHead?

    public init(
        backend: LlamaBackend,
        adapterPath: String,
        chatModeHead: ClassifierHead? = nil,
        topicGateHead: ClassifierHead? = nil,
        refusalFlagsHead: ClassifierHead? = nil,
        emotionalStateHead: ClassifierHead? = nil,
        slotCompletenessHead: ClassifierHead? = nil
    ) {
        self.backend = backend
        self.adapterPath = adapterPath
        self.chatModeHead = chatModeHead
        self.topicGateHead = topicGateHead
        self.refusalFlagsHead = refusalFlagsHead
        self.emotionalStateHead = emotionalStateHead
        self.slotCompletenessHead = slotCompletenessHead
    }

    /// Factory that wires bundled artifacts. Throws when the shared
    /// adapter is missing OR no heads can be loaded — both indicate a
    /// degraded bundle where the v2 strategy can't operate.
    public static func bundled(
        backend: LlamaBackend,
        bundle: Bundle = .main
    ) throws -> SharedBackboneStrategy {
        guard let adapterPath = TelcoModelBundle.understandingV2AdapterPath(in: bundle) else {
            throw QueryUnderstandingError.missingArtifact(name: TelcoModelBundle.understandingV2AdapterName)
        }

        let headPaths = TelcoModelBundle.understandingV2HeadPaths(in: bundle)
        guard !headPaths.isEmpty else {
            throw QueryUnderstandingError.missingArtifact(name: "telco-shared-clf-v2 heads")
        }

        func load(_ task: String) throws -> ClassifierHead? {
            guard let paths = headPaths[task] else { return nil }
            do {
                return try ClassifierHead(
                    weightsURL: paths.weightsURL,
                    biasURL: paths.biasURL,
                    metaURL: paths.metaURL
                )
            } catch {
                throw QueryUnderstandingError.headLoadFailure(task: task, underlying: error)
            }
        }

        return SharedBackboneStrategy(
            backend: backend,
            adapterPath: adapterPath,
            chatModeHead:        try load(TelcoModelBundle.understandingV2ChatModeHeadTask),
            topicGateHead:       try load(TelcoModelBundle.understandingV2TopicGateHeadTask),
            refusalFlagsHead:    try load(TelcoModelBundle.understandingV2RefusalFlagsHeadTask),
            emotionalStateHead:  try load(TelcoModelBundle.understandingV2EmotionalStateHeadTask),
            slotCompletenessHead: try load(TelcoModelBundle.understandingV2SlotCompletenessHeadTask)
        )
    }

    public func run(query: String) async throws -> QueryUnderstanding {
        // One adapter swap, one forward pass — the entire architectural
        // point of v2. Adapter cache makes the swap ~1 ms after first hit.
        let hidden: [Float]
        do {
            hidden = try await backend.embeddingsWithAdapter(
                prompt: query,
                adapterPath: adapterPath,
                clearCache: true
            )
        } catch {
            throw QueryUnderstandingError.backendFailure(underlying: error)
        }

        // Each head projects the SAME hidden state. Order doesn't
        // matter — they're independent matmuls.
        let chatModeOutcome = chatModeHead.flatMap { head -> ChatModePrediction? in
            let pred = head.classify(hidden)
            guard let mode = ChatMode(rawValue: pred.label) else { return nil }
            return ChatModePrediction(
                mode: mode,
                confidence: Double(pred.confidence),
                reasoning: "v2 head softmax (\(String(format: "%.1f", pred.confidence * 100))%)",
                runtimeMS: 0  // attributed to the shared pass; not double-counted
            )
        }

        let topicGateOutcome: TopicGateOutcome? = topicGateHead.map { head in
            let pred = head.classify(hidden)
            return TopicGateOutcome(
                value: TelcoStageAClassifier.topicGateFromLabel(pred.label),
                confidence: Double(pred.confidence)
            )
        }

        let refusalFlagsOutcome: RefusalFlagsOutcome? = refusalFlagsHead.map { head in
            let pred = head.classifyMultiLabel(hidden)
            return RefusalFlagsOutcome(
                value: TelcoStageAClassifier.refusalFlagsFromBinaryVector(pred.binaryVector),
                probabilities: pred.probabilities.map(Double.init)
            )
        }

        let emotionalStateOutcome: EmotionalStateOutcome? = emotionalStateHead.map { head in
            let pred = head.classify(hidden)
            return EmotionalStateOutcome(
                value: EmotionalState.from(wireName: pred.label),
                confidence: Double(pred.confidence)
            )
        }

        let slotCompletenessOutcome: SlotCompletenessOutcome? = slotCompletenessHead.map { head in
            let pred = head.classifyMultiLabel(hidden)
            return SlotCompletenessOutcome(
                value: SlotCompleteness.from(binaryVector: pred.binaryVector),
                probabilities: pred.probabilities.map(Double.init)
            )
        }

        // ADR-024 — relational fields (turnRelationship, slotAlignment,
        // stanceChange) default to nil here. Phase 8c integration point:
        // when `telco-relational-v1` adapter + heads are bundled,
        // construct a `RelationalHeadsStrategy` and run it as a second
        // pass on this same backbone; merge its outcomes into the
        // returned vector. Until then, the router treats nil as
        // INDEPENDENT — single-turn baseline.
        return QueryUnderstanding(
            chatMode: chatModeOutcome,
            topicGate: topicGateOutcome,
            refusalFlags: refusalFlagsOutcome,
            emotionalState: emotionalStateOutcome,
            slotCompleteness: slotCompletenessOutcome,
            totalMs: 0,  // re-stamped by the outer classifier
            strategy: .shared
        )
    }
}

// MARK: - Composite fallback (PR #30 path)

/// Wraps the existing `LFMChatModeRouter` + (optional) `TelcoStageAClassifier`
/// into the v2 contract. Pays 1 generative call (chat_mode, ~600 ms
/// on simulator, ~1.2 s on device) + 2 forward passes (Stage A, ~150 ms
/// each) when Stage A is reachable. Two new heads (`emotionalState`,
/// `slotCompleteness`) are reported as nil — no heuristic fallback
/// per ADR-022 §4.3 design principle #2 ("every head serves a concrete
/// product need").
///
/// This is the path the app runs TODAY. Once the Phase 2 H100 retrain
/// ships the shared adapter, the factory promotes to `SharedBackboneStrategy`
/// automatically — no code changes in ChatViewModel.
///
/// **Stage A is optional**: a degraded build without the Stage A heads
/// still produces a `chat_mode`-only QueryUnderstanding. The router
/// handles the absence per §4.3 — topic_gate=nil falls through to
/// chat_mode's coarse scope.
public struct CompositeFallbackStrategy: UnderstandingStrategy {
    private let chatModeRouter: ChatModeRouter
    private let stageA: TelcoStageAClassifying?

    public init(chatModeRouter: ChatModeRouter, stageA: TelcoStageAClassifying? = nil) {
        self.chatModeRouter = chatModeRouter
        self.stageA = stageA
    }

    public func run(query: String) async throws -> QueryUnderstanding {
        // Run chat_mode + Stage A concurrently when both are available
        // — they're independent and both hit the backend with different
        // adapters. The backend serialises adapter swaps internally, so
        // concurrency here doesn't deadlock; it just lets the runtime
        // pipeline whatever it can.
        //
        // Single-turn: ADR-023 Phase 1 once threaded history into the
        // chat_mode router here. ADR-024 retired that — multi-turn
        // signal flows through the relational heads, not through the
        // chat_mode prompt.
        async let modePredictionTask = chatModeRouter.classify(query: query)

        let stageADecision: TelcoStageADecision?
        if let stageA {
            async let stageATask: TelcoStageADecision = stageA.classify(query: query)
            do {
                stageADecision = try await stageATask
            } catch {
                // Stage A failure shouldn't kill the whole understanding
                // pass — chat_mode alone is still enough for the router
                // to make a decision (degraded mode, but live). Engineering
                // mode will see topicGate = nil and know.
                stageADecision = nil
            }
        } else {
            stageADecision = nil
        }

        let modePrediction = await modePredictionTask

        let topicGateOutcome = stageADecision.map { decision in
            TopicGateOutcome(
                value: decision.topicGate,
                confidence: decision.topicGateConfidence
            )
        }
        let refusalFlagsOutcome = stageADecision.map { decision in
            RefusalFlagsOutcome(
                value: decision.refusalFlags,
                probabilities: decision.refusalFlagsProbabilities.map(Double.init)
            )
        }

        return QueryUnderstanding(
            chatMode: modePrediction,
            topicGate: topicGateOutcome,
            refusalFlags: refusalFlagsOutcome,
            emotionalState: nil,
            slotCompleteness: nil,
            totalMs: 0,  // re-stamped by the outer classifier
            strategy: .composite
        )
    }
}

// MARK: - Unavailable (degraded build)

/// No-op strategy used when neither v2 nor v1 artifacts are bundled —
/// e.g., a fresh CI clone or an LFMEngine-disabled simulator build.
/// Returns an empty `QueryUnderstanding` with `strategy = .unavailable`.
/// The router falls back to OOS refusal per its degraded-build
/// safety default.
public struct UnavailableStrategy: UnderstandingStrategy {
    public init() {}
    public func run(query: String) async throws -> QueryUnderstanding {
        return QueryUnderstanding(
            chatMode: nil,
            topicGate: nil,
            refusalFlags: nil,
            emotionalState: nil,
            slotCompleteness: nil,
            totalMs: 0,
            strategy: .unavailable
        )
    }
}
