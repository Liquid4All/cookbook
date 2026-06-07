import Foundation
import os.log

/// Surface-area events the chat UI observes as the dispatcher runs.
///
/// Customer mode only acts on `.response` and `.failed`; engineering mode
/// renders every event as a row in the pipeline trace. This is the
/// same pattern the Banking POC `InferencePipeline` uses, just narrower
/// (one turn, no multi-step chain) — keeps the trace UI surface
/// consistent across both POCs for the inevitable future merge.
public enum VerizonDispatchEvent: Sendable, Equatable {
    case stageAStarted
    case stageAComplete(stageA: VerizonStageADecision)
    case laneSelected(VerizonLane)
    case retrievalStarted
    case retrievalComplete(result: ColBERTRetrievalResult)
    case stageBStarted
    case stageBComplete(StageBResponse)
    case faithfulnessChecked(score: FaithfulnessScore)
    case fallbackInvoked(reason: String)
    case response(VerizonDispatchResult)
    case failed(message: String)
}

/// Final response delivered to the chat view.
public struct VerizonDispatchResult: Sendable, Equatable {
    public enum Source: String, Sendable, Equatable {
        case composer  // Step 6 — deterministic composer on top of lexical retrieval
        case dialogueRepair  // V4 response-only verbalizer over composer evidence/policy
        case stageB    // Legacy Stage B path (disabled on v1 per Step 5 decision)
        case kbFallback
        case template
        case toolAction
        case personalSummary
    }

    /// User-facing response text. May contain Markdown links.
    public let text: String

    /// The lane the router picked. Surfaced for engineering trace +
    /// session telemetry.
    public let lane: VerizonLane

    /// Where the text came from (Stage B, deterministic KB fallback, a
    /// static template, etc.). Critical for the engineering-mode trace
    /// because Stage B vs fallback have very different latency profiles.
    public let source: Source

    /// vzhome:// deep link the user can tap, when the response carries
    /// one. Surfaced separately so the UI can render an actionable
    /// button below the message.
    public let deepLink: String?

    /// The RAG chunk Stage B was grounded against, when the response
    /// came from `.stageB` with a successful ColBERT retrieval.
    /// Drives the "Source: Network § Wi-Fi Management" citation chip
    /// per ADR §11.7. Nil when source is template, kbFallback, or
    /// when Stage B ran ungrounded (degraded mode).
    public let retrievedChunk: ColBERTChunk?

    /// The KB entry whose `.answer` was returned verbatim by the
    /// `kbFallback` source. Populated when ColBERT/Stage B failed and
    /// `handleUnknownFeature` matched a KeywordKBExtractor entry. The
    /// citation chip uses this when `retrievedChunk` is nil so the
    /// user sees "Read full article" regardless of which RAG path
    /// produced the text.
    public let kbEntry: KBEntry?

    /// Total wall-clock from query received to response ready. The
    /// engineering-mode latency badge reads this.
    public let totalMs: Double

    // -- Step 6 composer-path fields --

    /// Explicit timing breakdown for the deterministic composer path.
    /// These are nil for the legacy Stage B / KB fallback paths.
    public let retrievalMs: Double?
    public let routePolicyMs: Double?
    public let composerMs: Double?

    /// The composer route the dispatcher derived for this turn — when
    /// the composer was the answer source. Nil for legacy Stage B /
    /// kbFallback / template paths. Telemetry consumer.
    public let composerRoute: ComposerRoute?

    /// True iff the rendered text requires a real, executable tool
    /// confirmation (a registered `ToolIntent` exists for the unit's
    /// `linkID`). Surfaced so `ChatViewModel.runVerizonDispatch` can
    /// push a `PendingToolConfirmation` for the existing affirmative-
    /// recovery flow. NEVER true for view-only RAG answers (guardrail
    /// #3 in the Step 6 plan — no confirmation theatre).
    public let requiresConfirmation: Bool?

    /// Registered local tool behind this answer, when one exists.
    /// The UI/state layer uses this typed value to create pending
    /// confirmation state. It must never infer executable behavior
    /// from rendered text such as "Reply yes".
    public let executableToolIntent: ToolIntent?

    /// The canonical RAG unit the composer rendered from. Drives the
    /// citation chip on the composer path the same way `retrievedChunk`
    /// drove it on the legacy ColBERT path.
    public let citedRAGUnit: RAGUnit?

    /// ADR-028 blackboard facts from lexical retrieval. Populated on the
    /// composer path with the ranked top-k `BM25HierarchyRetriever` hits
    /// that policy considered before selecting `citedRAGUnit`.
    public let retrievalCandidates: [TelcoRetrievalCandidate]

    /// ADR-029 §6 measurement only: the policy engine's own `reuseActiveEvidence`
    /// flag and `reason` string, surfaced verbatim so the phone-flow harness can
    /// score the route decision from ground truth instead of inferring repair
    /// reuse from page-id continuity. No product behavior depends on these.
    public let reuseActiveEvidence: Bool
    public let policyReason: String?

    /// ADR-029 §7: the explicit dialogue-state operation (raw value) and its
    /// audit reason, serialized onto the harness report so `situation_eval` scores
    /// the state decision directly instead of inferring it from page continuity.
    public let stateOperation: String?
    public let stateOperationReason: String?

    public init(
        text: String,
        lane: VerizonLane,
        source: Source,
        deepLink: String? = nil,
        retrievedChunk: ColBERTChunk? = nil,
        kbEntry: KBEntry? = nil,
        totalMs: Double,
        retrievalMs: Double? = nil,
        routePolicyMs: Double? = nil,
        composerMs: Double? = nil,
        composerRoute: ComposerRoute? = nil,
        requiresConfirmation: Bool? = nil,
        executableToolIntent: ToolIntent? = nil,
        citedRAGUnit: RAGUnit? = nil,
        retrievalCandidates: [TelcoRetrievalCandidate] = [],
        reuseActiveEvidence: Bool = false,
        policyReason: String? = nil,
        stateOperation: String? = nil,
        stateOperationReason: String? = nil
    ) {
        self.text = text
        self.lane = lane
        self.source = source
        self.deepLink = deepLink
        self.retrievedChunk = retrievedChunk
        self.kbEntry = kbEntry
        self.totalMs = totalMs
        self.retrievalMs = retrievalMs
        self.routePolicyMs = routePolicyMs
        self.composerMs = composerMs
        self.composerRoute = composerRoute
        self.requiresConfirmation = requiresConfirmation
        self.executableToolIntent = executableToolIntent
        self.citedRAGUnit = citedRAGUnit
        self.retrievalCandidates = retrievalCandidates
        self.reuseActiveEvidence = reuseActiveEvidence
        self.policyReason = policyReason
        self.stateOperation = stateOperation
        self.stateOperationReason = stateOperationReason
    }
}

/// Spine of the merged chat flow.
///
/// Single entry point — `dispatch(query:)` returns an `AsyncStream` so
/// the UI can render progressive trace events as each stage completes.
/// Customer-mode views ignore intermediate events; engineering-mode
/// views render each as a trace row.
///
/// Lane handlers:
///   - `.liveAgentEscalation`     → static template + ETA
///   - `.navOnlyDeeplink`         → static template + Account deep link
///   - `.outOfScopeRefusal`       → Verizon canonical refusal
///   - `.greeting`                → friendly hello (no LLM)
///   - `.unknownFeature`          → KB fallback if it has a hit, else refusal
///   - `.clarification`           → ask user to pick (stub for now —
///                                  full multi-turn lands with Stage C)
///   - `.ragStepByStep`           → Stage B + LinkResolver, falls back to
///                                  KeywordKBExtractor on format break
///
/// Tool execution + personal summary are surfaced as `.toolAction` /
/// `.personalSummary` lane responses with placeholder bodies — the
/// dispatcher signals the lane and lets the existing ChatViewModel
/// scaffolding (ToolExecutor, LFMChatProvider + CustomerContext)
/// fulfill the user-visible response. That integration lands in the
/// next phase (ChatView rewire).
public actor VerizonChatDispatcher {
    private let stageA: VerizonStageAClassifying?
    private let stageB: StageBGenerating?
    private let kbFallback: KBExtractor
    private let kb: [KBEntry]
    private let retriever: ColBERTRetriever?
    private let modelHost: SwappingModelHost?
    // Step 6 composer-path dependencies. Optional for backwards
    // compatibility with existing test fixtures, but when all four are
    // wired (the production path), the dispatcher uses the composer
    // as the answer layer and Stage B / ColBERT / KeywordKBExtractor
    // become unreachable on .ragStepByStep.
    private let composer: AnswerComposing?
    private let corpus: RAGUnitCorpus?
    private let lexicalRetriever: BM25HierarchyRetriever?
    private let toolRegistry: ToolRegistry?
    private let toolAliasMap: ToolAliasMap?
    private let dialogueRepairVerbalizer: DialogueRepairVerbalizing?
    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "VerizonDispatcher"
    )

    public init(
        stageA: VerizonStageAClassifying?,
        stageB: StageBGenerating?,
        kbFallback: KBExtractor,
        kb: [KBEntry],
        retriever: ColBERTRetriever? = nil,
        modelHost: SwappingModelHost? = nil,
        composer: AnswerComposing? = nil,
        corpus: RAGUnitCorpus? = nil,
        lexicalRetriever: BM25HierarchyRetriever? = nil,
        toolRegistry: ToolRegistry? = nil,
        toolAliasMap: ToolAliasMap? = nil,
        dialogueRepairVerbalizer: DialogueRepairVerbalizing? = nil
    ) {
        self.stageA = stageA
        self.stageB = stageB
        self.kbFallback = kbFallback
        self.kb = kb
        self.retriever = retriever
        self.modelHost = modelHost
        self.composer = composer
        self.corpus = corpus
        self.lexicalRetriever = lexicalRetriever
        self.toolRegistry = toolRegistry
        self.toolAliasMap = toolAliasMap
        self.dialogueRepairVerbalizer = dialogueRepairVerbalizer
    }

    /// True when all Step 6 composer-path dependencies are present.
    /// The dispatcher takes the composer path iff this is true.
    private var composerPathEnabled: Bool {
        composer != nil && corpus != nil && lexicalRetriever != nil && toolRegistry != nil
    }

    /// First-principles online path for the Step 4/5 architecture:
    /// explicit conversation state + BM25Hierarchy retrieval +
    /// ToolRegistry policy + deterministic composer. This path never
    /// invokes Stage A, chat-mode-router, refusal flags, relational LoRA,
    /// Stage B, ColBERT, or KeywordKBExtractor.
    public nonisolated func dispatchComposer(
        query: String,
        retrievalContext: RetrievalContext = .empty,
        telcoUnderstanding: TelcoSharedUnderstanding? = nil,
        dialogueState: DialogueRepairConversationState = .empty,
        turnRelation: TelcoTurnRelation? = nil,
        policyState: TelcoDialogueStateSnapshot = .empty
    ) -> AsyncStream<VerizonDispatchEvent> {
        AsyncStream { continuation in
            Task {
                await self.runComposerOnly(
                    query: query,
                    retrievalContext: retrievalContext,
                    telcoUnderstanding: telcoUnderstanding,
                    dialogueState: dialogueState,
                    turnRelation: turnRelation,
                    policyState: policyState,
                    continuation: continuation
                )
            }
        }
    }

    public nonisolated func dispatch(
        query: String,
        retrievalContext: RetrievalContext = .empty
    ) -> AsyncStream<VerizonDispatchEvent> {
        AsyncStream { continuation in
            Task {
                await self.run(
                    query: query,
                    prebuiltStageA: nil,
                    prebuiltLane: nil,
                    retrievalContext: retrievalContext,
                    continuation: continuation
                )
            }
        }
    }

    /// ADR-022 §4.3 Layer 1 entry point — caller supplies a pre-built
    /// Stage A decision (from the v2 understanding pass) and the
    /// already-resolved `VerizonLane` (from `VerizonUnderstandingRouter.decide`).
    /// The dispatcher skips its own Stage A classifier call (avoiding
    /// the duplicate ~150 ms forward pass) and goes straight to lane
    /// handling. The progressive trace stream still emits
    /// `.stageAStarted` / `.stageAComplete` / `.laneSelected` so the
    /// existing trace UI doesn't break.
    ///
    /// ADR-024 follow-up 2026-05-27 — `retrievalContext` carries
    /// optional cross-turn signal (prior assistant text) to augment
    /// ColBERT retrieval for anaphoric / negative-continuation
    /// follow-ups. Default `.empty` for back-compat.
    public nonisolated func dispatch(
        query: String,
        prebuiltStageA: VerizonStageADecision,
        prebuiltLane: VerizonLane,
        retrievalContext: RetrievalContext = .empty
    ) -> AsyncStream<VerizonDispatchEvent> {
        AsyncStream { continuation in
            Task {
                await self.run(
                    query: query,
                    prebuiltStageA: prebuiltStageA,
                    prebuiltLane: prebuiltLane,
                    retrievalContext: retrievalContext,
                    continuation: continuation
                )
            }
        }
    }

    private func run(
        query: String,
        prebuiltStageA: VerizonStageADecision?,
        prebuiltLane: VerizonLane?,
        retrievalContext: RetrievalContext = .empty,
        continuation: AsyncStream<VerizonDispatchEvent>.Continuation
    ) async {
        let t0 = CFAbsoluteTimeGetCurrent()

        continuation.yield(.stageAStarted)
        let stageADecision: VerizonStageADecision
        if let prebuiltStageA {
            // Caller already paid for Stage A — re-yield it through the
            // event stream so the trace UI renders the same chain.
            stageADecision = prebuiltStageA
        } else {
            guard let stageA else {
                continuation.yield(.failed(message: "Stage A classifier unavailable on legacy dispatch path"))
                continuation.finish()
                return
            }
            do {
                stageADecision = try await stageA.classify(query: query)
            } catch {
                logger.error("Stage A failed: \(error.localizedDescription, privacy: .public)")
                continuation.yield(.failed(message: "Stage A classifier failed: \(error.localizedDescription)"))
                continuation.finish()
                return
            }
        }
        continuation.yield(.stageAComplete(stageA: stageADecision))

        // Trust the caller's lane decision when supplied; the v2 router
        // includes signals (chat_mode, emotional_state) that this
        // dispatcher's local `VerizonRagRouter` doesn't know about, so
        // re-routing here would discard them.
        let lane = prebuiltLane ?? VerizonRagRouter.route(stageA: stageADecision)
        continuation.yield(.laneSelected(lane))

        let result: VerizonDispatchResult
        switch lane {
        case .liveAgentEscalation:
            result = handleLiveAgent(query: query, lane: lane, t0: t0)
        case .navOnlyDeeplink:
            result = handleNavOnly(query: query, lane: lane, t0: t0)
        case .oosRefusal:
            result = handleOutOfScope(query: query, lane: lane, t0: t0)
        case .greeting:
            result = handleGreeting(query: query, lane: lane, t0: t0)
        case .unknownFeature:
            result = await handleUnknownFeature(query: query, lane: lane, t0: t0)
        case .clarification:
            result = handleClarification(query: query, lane: lane, t0: t0)
        case .ragStepByStep:
            result = await handleRagStepByStep(
                query: query,
                stageADecision: stageADecision,
                lane: lane,
                retrievalContext: retrievalContext,
                continuation: continuation,
                t0: t0
            )
        }

        continuation.yield(.response(result))
        continuation.finish()
    }

    private func runComposerOnly(
        query: String,
        retrievalContext: RetrievalContext,
        telcoUnderstanding: TelcoSharedUnderstanding?,
        dialogueState: DialogueRepairConversationState,
        turnRelation: TelcoTurnRelation?,
        policyState: TelcoDialogueStateSnapshot,
        continuation: AsyncStream<VerizonDispatchEvent>.Continuation
    ) async {
        let t0 = CFAbsoluteTimeGetCurrent()
        let effectivePolicyState = Self.effectivePolicyState(
            provided: policyState,
            retrievalContext: retrievalContext,
            dialogueState: dialogueState
        )
        let effectiveTurnRelation = Self.effectiveTurnRelation(
            provided: turnRelation,
            query: query,
            retrievalContext: retrievalContext,
            dialogueState: dialogueState,
            policyState: effectivePolicyState
        )
        let result = await runComposerPipeline(
            query: query,
            retrievalContext: retrievalContext,
            telcoUnderstanding: telcoUnderstanding,
            dialogueState: dialogueState,
            turnRelation: effectiveTurnRelation,
            policyState: effectivePolicyState,
            continuation: continuation,
            t0: t0
        )
        continuation.yield(.laneSelected(result.lane))
        continuation.yield(.response(result))
        continuation.finish()
    }

    // MARK: - Lane handlers
    //
    // Step 6.5 lane migration: when the composer is wired (production
    // path), every lane handler delegates to the composer so all
    // user-facing text comes from a single source of truth. The
    // composer's per-route templates are documented at
    // `scripts/vz/answer_composer.py::ROUTE_TEMPLATES` and mirrored in
    // `Core/Composer/AnswerComposer.swift`.
    //
    // The legacy inline templates remain as the dev / engineering
    // fallback when `composerPathEnabled == false` — same words as
    // before so legacy behaviour is unchanged for that path.

    private func handleLiveAgent(query: String, lane: VerizonLane, t0: CFAbsoluteTime) -> VerizonDispatchResult {
        composeTemplateResponse(
            route: .liveAgent,
            query: query,
            lane: lane,
            legacyText: "Connecting you to a Verizon support agent. Estimated wait: a few minutes.",
            legacyDeepLink: VerizonLinkResolver.levelOneFallback(for: .liveAgent),
            t0: t0
        )
    }

    private func handleNavOnly(query: String, lane: VerizonLane, t0: CFAbsoluteTime) -> VerizonDispatchResult {
        // Account / billing pivots — no LLM near financial data.
        // Composer renders the canonical My Verizon URL; legacy
        // fallback uses the in-app Account tab deep link.
        let legacyLink = VerizonLinkResolver.levelOneFallback(for: .accountOOS)
        let legacyText = "I can't help with that here. Go to [Account](\(legacyLink)) > Bills to manage your account."
        return composeTemplateResponse(
            route: .accountNav,
            query: query,
            lane: lane,
            legacyText: legacyText,
            legacyDeepLink: legacyLink,
            t0: t0
        )
    }

    private func handleOutOfScope(query: String, lane: VerizonLane, t0: CFAbsoluteTime) -> VerizonDispatchResult {
        composeTemplateResponse(
            route: .outOfScope,
            query: query,
            lane: lane,
            legacyText: "I'm here to help with Verizon Home Internet — router, network, devices, parental controls, equipment, or Digital Secure Home. Please try asking a different question.",
            legacyDeepLink: nil,
            t0: t0
        )
    }

    private func handleGreeting(query: String, lane: VerizonLane, t0: CFAbsoluteTime) -> VerizonDispatchResult {
        composeTemplateResponse(
            route: .greeting,
            query: query,
            lane: lane,
            legacyText: "Hi — I'm your Verizon Home Internet assistant. Ask me about your Wi-Fi, router, devices, parental controls, or equipment.",
            legacyDeepLink: nil,
            t0: t0
        )
    }

    private func handleClarification(query: String, lane: VerizonLane, t0: CFAbsoluteTime) -> VerizonDispatchResult {
        // Composer's `.clarify` template fills the "are you asking
        // about X?" slot from the user's query.
        composeTemplateResponse(
            route: .clarify,
            query: query,
            lane: lane,
            legacyText: "Could you tell me more about what you're trying to do? For example: change your Wi-Fi password, restart your router, or check connected devices.",
            legacyDeepLink: nil,
            t0: t0
        )
    }

    private func handleUnknownFeature(
        query: String,
        lane: VerizonLane,
        t0: CFAbsoluteTime
    ) async -> VerizonDispatchResult {
        // **Guardrail #2**: on the composer path, `KeywordKBExtractor`
        // is NOT consulted. We render the canonical no_rag_answer
        // template instead — composer's safe fallback URL only, no
        // alternative-answer corpus.
        if composerPathEnabled {
            return composeTemplateResponse(
                route: .noRagAnswer,
                query: query,
                lane: lane,
                legacyText: "It looks like I don't have information about that. Would you like me to connect you with a Verizon support agent?",
                legacyDeepLink: nil,
                t0: t0
            )
        }
        // ---- Legacy dev/engineering path: KB fallback ----
        let citation = await kbFallback.extract(query: query, kb: kb)
        if citation.isMatch, let entry = kb.first(where: { $0.id == citation.entryId }) {
            return VerizonDispatchResult(
                text: entry.answer,
                lane: lane,
                source: .kbFallback,
                deepLink: entry.deepLinks.first?.url,
                kbEntry: entry,
                totalMs: elapsed(t0)
            )
        }
        return VerizonDispatchResult(
            text: "It looks like I don't have information about that. Would you like me to connect you with a Verizon support agent?",
            lane: lane,
            source: .template,
            deepLink: nil,
            totalMs: elapsed(t0)
        )
    }

    /// Single source of truth for non-RAG template responses. When the
    /// composer is wired, delegates to it so route-text-link rendering
    /// stays consistent with the composer's contract (and with the
    /// Python `answer_composer.py` reference). When not wired, returns
    /// the legacy inline template as before.
    private func composeTemplateResponse(
        route: ComposerRoute,
        query: String,
        lane: VerizonLane,
        legacyText: String,
        legacyDeepLink: String?,
        t0: CFAbsoluteTime
    ) -> VerizonDispatchResult {
        if composerPathEnabled, let composer = composer {
            let composerStart = CFAbsoluteTimeGetCurrent()
            let answer = composer.compose(
                query: query,
                route: route,
                evidence: nil,
                requiresConfirmation: nil,
                history: [],
                expectedPolicyLinkID: nil
            )
            let composerMs = elapsed(composerStart)
            return VerizonDispatchResult(
                text: answer.text,
                lane: lane,
                source: .composer,
                deepLink: answer.renderedLinks.first,
                totalMs: elapsed(t0),
                retrievalMs: 0,
                routePolicyMs: 0,
                composerMs: composerMs,
                composerRoute: route,
                requiresConfirmation: false,
                executableToolIntent: nil,
                citedRAGUnit: nil
            )
        }
        return VerizonDispatchResult(
            text: legacyText,
            lane: lane,
            source: .template,
            deepLink: legacyDeepLink,
            totalMs: elapsed(t0)
        )
    }

    /// The .ragStepByStep handler.
    ///
    /// **Step 6 composer path** (production, taken when `composerPathEnabled`):
    ///   1. Lexical retrieve via `BM25HierarchyRetriever` (post-4c
    ///      alias-improved baseline, the retrieval source of truth
    ///      per the Step 5 decision record).
    ///   2. Derive composer route via the `ToolRegistry` gate
    ///      (guardrail #3 — confirmation iff a real executable
    ///      `ToolIntent` is registered for the unit's `linkID`).
    ///   3. `AnswerComposer.compose(...)` renders the user-facing text.
    ///      Cannot hallucinate a link by construction.
    ///   4. Emit. No faithfulness check needed — the composer can't
    ///      paraphrase off the evidence.
    ///
    /// **Legacy Stage B path** (taken only when composer dependencies
    /// are NOT wired — preserved for dev / engineering builds, not
    /// reachable in normal demo). The original 5-stage Ottoguard
    /// shape: ColBERT retrieve → Stage B generate → faithfulness gate →
    /// KeywordKBExtractor fallback on any failure.
    private func handleRagStepByStep(
        query: String,
        stageADecision: VerizonStageADecision,
        lane: VerizonLane,
        retrievalContext: RetrievalContext,
        continuation: AsyncStream<VerizonDispatchEvent>.Continuation,
        t0: CFAbsoluteTime
    ) async -> VerizonDispatchResult {
        if composerPathEnabled {
            let policyState = Self.effectivePolicyState(
                provided: .empty,
                retrievalContext: retrievalContext,
                dialogueState: .empty
            )
            return await runComposerPipeline(
                query: query,
                retrievalContext: retrievalContext,
                telcoUnderstanding: nil,
                dialogueState: .empty,
                turnRelation: Self.effectiveTurnRelation(
                    provided: nil,
                    query: query,
                    retrievalContext: retrievalContext,
                    dialogueState: .empty,
                    policyState: policyState
                ),
                policyState: policyState,
                continuation: continuation,
                t0: t0
            )
        }
        // ---- Legacy Stage B path (dev / engineering only) ----
        guard let stageB = stageB else {
            continuation.yield(.fallbackInvoked(reason: "Stage B adapter not bundled"))
            return await handleUnknownFeature(query: query, lane: lane, t0: t0)
        }

        // ---- Stage 2: Retrieve ----
        // Skip if retriever or model host isn't bundled — degraded
        // mode generates Stage B without grounding, which is what
        // the §10 implementation did. Logged loudly so engineering
        // mode shows the missing signal.
        var retrieval: ColBERTRetrievalResult?
        if let retriever, let modelHost {
            continuation.yield(.retrievalStarted)
            do {
                let result: ColBERTRetrievalResult = try await modelHost.withColBERT { backend in
                    // ADR-024 follow-up 2026-05-27 — `retrievalContext`
                    // carries prior-assistant text when the router
                    // emitted `.augmentRetrievalWithPriorAssistant`.
                    // `.empty` for single-turn / first-turn paths.
                    try await retriever.retrieve(
                        query: query,
                        context: retrievalContext,
                        via: backend
                    )
                }
                continuation.yield(.retrievalComplete(result: result))
                retrieval = result
            } catch {
                continuation.yield(.fallbackInvoked(
                    reason: "ColBERT retrieval threw: \(error.localizedDescription)"
                ))
                logger.error(
                    "ColBERT retrieval failed: \(error.localizedDescription, privacy: .public)"
                )
                // Fall through with retrieval == nil → ungrounded Stage B
            }
        } else {
            continuation.yield(.fallbackInvoked(reason: "ColBERT not bundled"))
        }

        // ---- Re-route on retrieval signal ----
        // If retrieval came back ambiguous or below the floor, the
        // router may now pick .clarification or .unknownFeature
        // instead of staying on .ragStepByStep. Honor that.
        let refinedLane = VerizonRagRouter.route(
            stageA: stageADecision,
            retrieval: retrieval
        )
        if refinedLane != lane {
            // Router downgraded the lane after seeing retrieval. Yield
            // the refined lane so engineering trace shows the
            // re-routing, then dispatch to the appropriate handler.
            continuation.yield(.laneSelected(refinedLane))
            logger.info(
                "lane refined post-retrieval: \(lane.wireName, privacy: .public) → \(refinedLane.wireName, privacy: .public)"
            )
            switch refinedLane {
            case .unknownFeature:
                return await handleUnknownFeature(query: query, lane: refinedLane, t0: t0)
            case .clarification:
                return handleClarification(query: query, lane: refinedLane, t0: t0)
            case .ragStepByStep:
                break  // shouldn't happen given refinedLane != lane, but defensive
            default:
                // Any other lane swap is unexpected here (Stage A
                // already picked .ragStepByStep, retrieval can only
                // downgrade). Surface via fallback.
                continuation.yield(.fallbackInvoked(
                    reason: "unexpected lane swap: \(lane.wireName) → \(refinedLane.wireName)"
                ))
                return await handleUnknownFeature(query: query, lane: refinedLane, t0: t0)
            }
        }

        // ---- Stage 3: Generate (with grounding when retrieval succeeded) ----
        let topChunk = retrieval?.hits.first?.chunk
        continuation.yield(.stageBStarted)
        do {
            let response = try await stageB.generate(
                query: query,
                retrievedChunk: topChunk
            )
            continuation.yield(.stageBComplete(response))

            // ---- Stage 4: Validate (format compliance + faithfulness) ----
            // Format compliance gate — fast deterministic check
            // (extracted URL is known, intro present, no bangs).
            // The GBNF grammar (Phase 2) eliminates most shape
            // failures by construction; this catches escape hatches
            // like hallucinated-but-shaped URLs that pass GBNF but
            // fail isKnownDeepLink.
            guard response.isFormatCompliant else {
                continuation.yield(.fallbackInvoked(reason: "Stage B format check failed"))
                logger.warning(
                    "Stage B fallback (format): \(response.rawText.prefix(120), privacy: .public)"
                )
                return await handleUnknownFeature(query: query, lane: refinedLane, t0: t0)
            }

            // Faithfulness gate — did Stage B actually use the
            // retrieved chunk, or did it paraphrase from training?
            // Token-overlap v1 per ADR §11.4.2; reverse retrieval
            // is v2 (needs dual-backend topology to be cheap).
            // Skipped when topChunk is nil (degraded mode without
            // ColBERT) — nothing to be faithful to.
            let faithfulness = FaithfulnessScorer.score(
                response: response.rawText,
                chunk: topChunk
            )
            continuation.yield(.faithfulnessChecked(score: faithfulness))
            if !faithfulness.isFaithful {
                continuation.yield(.fallbackInvoked(
                    reason: "Stage B faithfulness \(String(format: "%.2f", faithfulness.bigramJaccard)) below floor \(String(format: "%.2f", faithfulness.floor))"
                ))
                logger.warning(
                    "Stage B fallback (faithfulness): jaccard=\(String(format: "%.3f", faithfulness.bigramJaccard), privacy: .public) below floor"
                )
                return await handleUnknownFeature(query: query, lane: refinedLane, t0: t0)
            }

            return VerizonDispatchResult(
                text: response.rawText,
                lane: refinedLane,
                source: .stageB,
                deepLink: response.extractedDeepLink,
                retrievedChunk: topChunk,
                totalMs: elapsed(t0)
            )
        } catch {
            continuation.yield(.fallbackInvoked(reason: "Stage B threw: \(error.localizedDescription)"))
            logger.error("Stage B threw: \(error.localizedDescription, privacy: .public)")
            return await handleUnknownFeature(query: query, lane: refinedLane, t0: t0)
        }
    }

    private nonisolated func elapsed(_ t0: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - t0) * 1000
    }

    // MARK: - Step 6 composer path

    /// Step 6 production answer path: lexical retrieve + ToolRegistry-
    /// gated route derivation + deterministic composer. No LLM call,
    /// no model swap, no GBNF, no faithfulness scorer. Sub-millisecond.
    ///
    /// Guarantees by construction:
    ///   * Every rendered `vzhome://` URL is the selected unit's
    ///     `canonicalURL` (or one of the safe external fallbacks).
    ///   * `requiresConfirmation = true` ONLY when `ToolRegistry`
    ///     has a real `ToolIntent` registered for the unit's `linkID`
    ///     (guardrail #3 — no confirmation theatre on view-only pages).
    ///   * Composer never auto-fires a tool — the dispatcher only
    ///     surfaces `requiresConfirmation` on the result envelope; the
    ///     ChatViewModel's existing affirmative-recovery flow handles
    ///     the actual tool execution after user confirmation.
    /// ADR-029 §1 production answer path. Pure, sub-millisecond:
    ///
    ///   1. Lexical retrieve via `BM25HierarchyRetriever` (state-conditioned).
    ///   2. `TelcoPolicyEngine.decide(...)` — the single authoritative route
    ///      owner — consumes the turn relation, shared understanding, the
    ///      dialogue-state snapshot, the retrieval candidates, the selected
    ///      unit, and the tool registry/alias map, and emits one
    ///      `TelcoPolicyResolution`.
    ///   3. `AnswerComposer.compose(...)` renders the decided route. The
    ///      composer never re-derives the route.
    ///   4. Optional response-only V4 dialogue-repair verbalizer.
    ///
    /// Guarantees by construction:
    ///   * Every rendered `vzhome://` URL is the selected unit's
    ///     `canonicalURL` (or a safe external fallback).
    ///   * Confirmation is required iff the policy engine resolved a real
    ///     executable `ToolIntent` for the unit (guardrail #3).
    ///   * For non-grounded routes (live-agent, account, clarify, greeting,
    ///     out-of-scope, no-answer) the cited unit is nil — the route never
    ///     leaks an unrelated local page into the citation chip.
    private func runComposerPipeline(
        query: String,
        retrievalContext: RetrievalContext,
        telcoUnderstanding: TelcoSharedUnderstanding?,
        dialogueState: DialogueRepairConversationState,
        turnRelation: TelcoTurnRelation?,
        policyState: TelcoDialogueStateSnapshot,
        continuation: AsyncStream<VerizonDispatchEvent>.Continuation,
        t0: CFAbsoluteTime
    ) async -> VerizonDispatchResult {
        guard
            let composer = composer,
            let corpus = corpus,
            let lexicalRetriever = lexicalRetriever,
            let toolRegistry = toolRegistry
        else {
            // composerPathEnabled gated this call, so all four must be
            // present. Defensive fallback in case a future caller
            // bypasses the gate.
            continuation.yield(.fallbackInvoked(reason: "composer dependencies missing"))
            return VerizonDispatchResult(
                text: "It looks like I don't have specific information about that. You can check [Verizon Home Internet](https://www.verizon.com/home/internet) for more details.",
                lane: .ragStepByStep,
                source: .composer,
                deepLink: nil,
                totalMs: elapsed(t0),
                retrievalMs: 0,
                routePolicyMs: 0,
                composerMs: 0,
                composerRoute: .noRagAnswer,
                requiresConfirmation: false,
                executableToolIntent: nil,
                citedRAGUnit: nil
            )
        }

        // ---- Retrieve (BM25 hierarchy, state-conditioned) ----
        // Retrieval runs on every turn — it is sub-millisecond and
        // side-effect-free — so the policy owner can see evidence
        // availability before deciding. This unifies what used to be a
        // pre-retrieval and a post-retrieval policy split (ADR-029 §1).
        continuation.yield(.retrievalStarted)
        let history: [ConversationTurnSnippet] = retrievalContext.priorAssistantText.map {
            [ConversationTurnSnippet(role: "ASSISTANT", body: $0)]
        } ?? []
        let retrievalStart = CFAbsoluteTimeGetCurrent()
        let hits = lexicalRetriever.rank(query: query, context: retrievalContext, k: 3)
        let retrievalMs = elapsed(retrievalStart)
        let topUnit = hits.first.flatMap { corpus.unit(forPageID: $0.pageID) }
        let retrievalCandidates = hits.map {
            TelcoRetrievalCandidate(pageID: $0.pageID, linkID: $0.linkID, score: $0.score)
        }

        // ---- Resolve the explicit dialogue-state operation (ADR-029 §7) ----
        // Computed before the route policy so it is authoritative AND serialized
        // onto the harness report. Retrieval already ran above (sub-ms,
        // side-effect-free); the operation's retrieval strategy governs whether
        // the policy reuses the active task's prior evidence vs the fresh top hit.
        let routePolicyStart = CFAbsoluteTimeGetCurrent()
        let prior = TelcoDeterministicPrior.derive(query: query)
        let stateResolution = TelcoStateOperationResolver.resolve(
            query: query,
            relation: turnRelation,
            prior: prior,
            state: policyState
        )
        let signals = TelcoPolicySignals(
            query: query,
            relation: turnRelation,
            understanding: telcoUnderstanding,
            state: policyState,
            prior: prior,
            stateResolution: stateResolution
        )
        let decision = TelcoPolicyEngine.decide(
            signals: signals,
            candidates: retrievalCandidates,
            selectedUnit: topUnit,
            toolRegistry: toolRegistry,
            aliasMap: toolAliasMap
        )
        let routePolicyMs = elapsed(routePolicyStart)

        // ---- Resolve composer evidence from the decision ----
        // Grounded routes render from a RAG unit; a repair continuation
        // reuses the active task's prior unit so page/link state does not
        // drift. Non-grounded routes intentionally carry no cited unit.
        let composerEvidence = evidence(
            for: decision,
            topUnit: topUnit,
            policyState: policyState,
            corpus: corpus
        )

        // ---- Compose (renders the decided route) ----
        let composerStart = CFAbsoluteTimeGetCurrent()
        let answer = composer.compose(
            query: query,
            route: decision.route,
            evidence: composerEvidence,
            requiresConfirmation: decision.requiresConfirmation,
            history: history,
            expectedPolicyLinkID: nil
        )
        let composerMs = elapsed(composerStart)

        var finalText = answer.text
        var finalSource: VerizonDispatchResult.Source = .composer
        var finalComposerMs = composerMs
        if let dialogueRepairVerbalizer,
           let act = DialogueRepairActDeriver.derive(
                query: query,
                route: decision.route,
                evidence: composerEvidence,
                retrievalContext: retrievalContext,
                understanding: telcoUnderstanding
           ),
           Self.shouldUseDialogueRepairVerbalizer(
                act: act,
                route: decision.route,
                retrievalContext: retrievalContext,
                dialogueState: dialogueState
           ) {
            let verbalizerInput = DialogueRepairVerbalizerInput(
                currentUserTurn: query,
                priorAssistantText: retrievalContext.priorAssistantText,
                conversationState: dialogueState,
                understanding: telcoUnderstanding,
                evidence: composerEvidence,
                route: decision.route,
                act: act,
                handoff: DialogueRepairActDeriver.handoff(for: decision.route),
                requiresConfirmation: decision.requiresConfirmation
            )
            let verbalized = await dialogueRepairVerbalizer.verbalize(verbalizerInput)
            finalText = Self.withRuntimeOwnedConfirmation(
                verbalized.text,
                requiresConfirmation: decision.requiresConfirmation
            )
            finalSource = .dialogueRepair
            finalComposerMs += verbalized.latencyMs
            logger.info(
                "dialogue_repair_v4_used act=\(act.rawValue, privacy: .public) fallback=\(verbalized.usedFallback, privacy: .public) extraction=\(verbalized.extractionMode, privacy: .public)"
            )
        }

        logger.info(
            "policy op=\(decision.stateOperation.rawValue, privacy: .public)/\(stateResolution.retrieval.rawValue, privacy: .public) route=\(decision.route.wireName, privacy: .public) reason=\(decision.reason, privacy: .public) pid=\(composerEvidence?.pageID ?? "<none>", privacy: .public) link=\(composerEvidence?.linkID ?? "<none>", privacy: .public) confirm=\(decision.requiresConfirmation, privacy: .public) tool=\(decision.executableToolIntent?.toolID ?? "<none>", privacy: .public) source=\(finalSource.rawValue, privacy: .public)"
        )

        return VerizonDispatchResult(
            text: finalText,
            lane: Self.lane(for: decision.route),
            source: finalSource,
            deepLink: answer.renderedLinks.first,
            retrievedChunk: nil,
            kbEntry: nil,
            totalMs: elapsed(t0),
            retrievalMs: retrievalMs,
            routePolicyMs: routePolicyMs,
            composerMs: finalComposerMs,
            composerRoute: decision.route,
            requiresConfirmation: decision.requiresConfirmation,
            executableToolIntent: decision.executableToolIntent,
            citedRAGUnit: composerEvidence,
            retrievalCandidates: retrievalCandidates,
            reuseActiveEvidence: decision.reuseActiveEvidence,
            policyReason: decision.reason,
            stateOperation: decision.stateOperation.rawValue,
            stateOperationReason: decision.stateOperationReason
        )
    }

    /// Resolve the RAG unit the composer renders from, given the policy
    /// decision. Only evidence-bearing routes carry a unit; a repair
    /// continuation reuses the active task's prior page.
    private nonisolated func evidence(
        for decision: TelcoPolicyResolution,
        topUnit: RAGUnit?,
        policyState: TelcoDialogueStateSnapshot,
        corpus: RAGUnitCorpus
    ) -> RAGUnit? {
        guard decision.route.requiresEvidence else { return nil }
        if decision.reuseActiveEvidence,
           let priorPageID = policyState.priorPageID,
           let priorUnit = corpus.unit(forPageID: priorPageID) {
            return priorUnit
        }
        return topUnit
    }

    /// Bridge older dispatcher callers that only carry `RetrievalContext` into
    /// the current policy-engine state contract. The live ChatViewModel passes a
    /// full blackboard snapshot; tests and compatibility callers often only know
    /// the prior page/link and optional pending tool.
    private nonisolated static func effectivePolicyState(
        provided: TelcoDialogueStateSnapshot,
        retrievalContext: RetrievalContext,
        dialogueState: DialogueRepairConversationState
    ) -> TelcoDialogueStateSnapshot {
        guard provided == .empty else { return provided }

        let priorText = retrievalContext.priorAssistantText?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let hasPriorAssistant = priorText.map { !$0.isEmpty } ?? false
        let hasState =
            retrievalContext.priorPageID != nil
            || retrievalContext.priorLinkID != nil
            || hasPriorAssistant
            || dialogueState.pendingTool != nil
            || dialogueState.pendingConfirmation
            || dialogueState.frustrationCount > 0
        guard hasState else { return provided }

        return TelcoDialogueStateSnapshot(
            hasActiveTask: retrievalContext.priorPageID != nil,
            priorPageID: retrievalContext.priorPageID,
            priorLinkID: retrievalContext.priorLinkID,
            pendingToolID: dialogueState.pendingTool,
            repairAttemptsOnActiveTask: dialogueState.frustrationCount,
            frustrationCount: dialogueState.frustrationCount,
            hasPriorAssistantTurn: hasPriorAssistant,
            priorRouteWasClarify: false
        )
    }

    /// Compatibility relation fallback for callers that have not run the
    /// relation head but do carry prior-page context. Uses the same multi-turn
    /// primitives as the production blackboard path: confirmations, repair,
    /// topic switch, step focus, and short anaphoric follow-ups.
    private nonisolated static func effectiveTurnRelation(
        provided: TelcoTurnRelation?,
        query: String,
        retrievalContext: RetrievalContext,
        dialogueState: DialogueRepairConversationState,
        policyState: TelcoDialogueStateSnapshot
    ) -> TelcoTurnRelation? {
        if let provided { return provided }

        if ConversationStateRecorder.isLiveAgentRequest(query) {
            return .escalationRequest
        }
        if ConversationStateRecorder.isBareAffirmative(query) {
            return policyState.pendingToolID == nil ? .ambiguousShortTurn : .confirmationYes
        }
        if ConversationStateRecorder.isBareNegative(query) {
            return policyState.pendingToolID == nil ? .ambiguousShortTurn : .confirmationNo
        }
        if ConversationStateRecorder.isDidntWorkContinuation(query) {
            return .repairFailed
        }

        let normalized = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized.contains("can't find") || normalized.contains("cannot find") {
            return .repairCannotFind
        }
        if hasTopicSwitchPrefix(query) {
            return .topicSwitch
        }
        if policyState.priorPageID != nil, normalized.hasPrefix("where") {
            return .stepFocus
        }
        if policyState.priorPageID != nil || retrievalContext.priorPageID != nil,
           isShortFollowup(query) {
            return .ambiguousShortTurn
        }
        return nil
    }

    /// Map an authoritative route to the legacy `VerizonLane` the trace UI
    /// and session telemetry consume. The route is the source of truth;
    /// the lane is a presentation projection.
    private nonisolated static func lane(for route: ComposerRoute) -> VerizonLane {
        switch route {
        case .liveAgent: return .liveAgentEscalation
        case .accountNav: return .navOnlyDeeplink
        case .outOfScope: return .oosRefusal
        case .greeting: return .greeting
        case .clarify: return .clarification
        case .noRagAnswer: return .unknownFeature
        case .ragAnswer, .answerPlusAction, .toolAction: return .ragStepByStep
        }
    }

    private nonisolated static func withRuntimeOwnedConfirmation(
        _ text: String,
        requiresConfirmation: Bool?
    ) -> String {
        guard requiresConfirmation == true else { return text }
        let lower = text.lowercased()
        guard !lower.contains("reply 'yes'") && !lower.contains("reply yes") else {
            return text
        }
        return "\(text) Reply 'yes' to confirm."
    }

    private nonisolated static func shouldUseDialogueRepairVerbalizer(
        act: DialogueRepairAct,
        route: ComposerRoute,
        retrievalContext: RetrievalContext,
        dialogueState: DialogueRepairConversationState
    ) -> Bool {
        let priorAssistant = retrievalContext.priorAssistantText?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let hasPriorTurn = priorAssistant.map { !$0.isEmpty } ?? false
        let hasRepairState = hasPriorTurn ||
            dialogueState.pendingConfirmation ||
            dialogueState.frustrationCount > 0

        switch act {
        case .navigationFollowup, .stepFocus, .failedAttempt, .repairCannotFind:
            return hasRepairState
        case .clarification:
            return route == .clarify
        case .noAnswerRepair:
            return route == .noRagAnswer ||
                route == .outOfScope ||
                route == .accountNav ||
                route == .liveAgent
        case .topicPivotRepair:
            return hasPriorTurn
        }
    }
}
