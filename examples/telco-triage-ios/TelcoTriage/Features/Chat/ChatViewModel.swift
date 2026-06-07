import Foundation
import Combine
import UIKit

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var inputText: String = ""
    @Published var attachedImage: UIImage?
    @Published var isProcessing: Bool = false
    @Published var routingStage: RoutingStage?
    @Published var privacyShieldQuery: PrivacyShieldState?
    /// Binding driving the `KBArticleView` sheet. Non-nil → sheet is
    /// presented for that entry. Set by `openKBArticle(_:)` from the
    /// "Read full article" chip.
    @Published var readingArticle: KBEntry?

    // Dependency access level is `internal` rather than `private` so the
    // vision helper can live in a separate extension file
    // (ChatViewModel+Vision.swift) and keep this file lean.
    let provider: LFMChatProvider
    let piiAnalyzer: PIIAnalyzer
    let kb: KnowledgeBase
    let tokenLedger: TokenLedger
    let sessionStats: SessionStats
    let toolRegistry: ToolRegistry
    let visionAnalyzer: VisionAnalyzer
    let customerContext: CustomerContext
    let nbaEngine: NextBestActionEngine

    // Intelligence layer. The Liquid Telco composer dispatcher owns the
    // normal support Q&A path. ChatModeRouter and the unified
    // understanding classifier remain injectable for degraded builds,
    // explicit experiments, and non-composer features.
    let chatModeRouter: ChatModeRouter
    let kbExtractor: KBExtractor
    let queryExtractor: QueryExtractor
    let toolSelector: ToolSelector

    /// Liquid Telco composer dispatcher. When non-nil, owns the normal
    /// support answer path from explicit state + BM25HierarchyRetriever
    /// evidence + deterministic composition. The legacy composite
    /// understanding stack is bypassed unless this dispatcher is absent.
    let verizonDispatcher: VerizonChatDispatcher?

    /// ADR-026 semantic control plane for normal Telco Triage turns:
    /// one LFM2.5-350M shared-adapter pass over 9 classifier heads.
    /// Nil only in degraded bundles or tests that do not need model
    /// semantics.
    let telcoUnderstandingClassifier: TelcoSharedUnderstandingClassifying?

    /// ADR-022 §4.3 Layer 1 — the unified understanding classifier.
    /// Always present (`UnavailableStrategy` covers degraded builds).
    /// Replaces the bare `chatModeRouter.classify(query:)` call that
    /// formerly opened `processTextQuery`; the resulting
    /// `QueryUnderstanding` vector drives routing AND populates the
    /// engineering trace card.
    let understandingClassifier: QueryUnderstandingClassifying

    /// ADR-028 turn-relation classifier/fallback. On current bundles this
    /// prefers `telco-turn-relation-v4`; degraded builds return `.none`
    /// and the blackboard reducer applies deterministic safety fallbacks.
    /// Always non-nil — no optional chaining needed.
    let relationalStrategy: RelationalHeadsStrategy

    /// Latest understanding vector for the current turn. Published so
    /// the (future) engineering-mode trace view can render it live as
    /// each stage completes. Reset on every new turn.
    @Published var lastUnderstanding: QueryUnderstanding?

    /// ADR-023 Phase 2 — session-scoped conversation state. Owns
    /// pendingClarification, pendingToolConfirmation, and the
    /// frustration counters. Lives for the chat session. Drives the
    /// pre-classifier clarification recovery + the NBA layer's
    /// counter-based escalation.
    let conversationState: ConversationState

    /// ADR-028 typed blackboard for turn relation, retrieval evidence,
    /// policy decisions, pending actions, and response audit events.
    /// `ConversationState` remains the UI/tool-recovery state; this
    /// reducer-owned state is the architecture-facing control plane.
    @Published private(set) var dialogueBlackboard: TelcoDialogueBlackboard

    /// Latest Stage A + dispatcher result for the current turn. Captured
    /// from the AsyncStream events so the engineering-mode trace UI can
    /// render the new pipeline rows. Reset on every new turn.
    @Published var lastVerizonStageA: VerizonStageADecision?
    @Published var lastVerizonLane: VerizonLane?
    @Published var lastVerizonStageBResponse: StageBResponse?
    @Published var lastVerizonResult: VerizonDispatchResult?

    // Tool execution + result synthesis
    let toolExecutor: ToolExecutor
    private let useSimulatorFastGroundedQA: Bool

    /// Per-message pipeline-card expand state. Lifted out of the card's
    /// `@State` because `LazyVStack` recycles cells aggressively — local
    /// `@State` on a child of a `ForEach` is reset on scroll, which made
    /// collapsed cards re-expand themselves after the user collapsed
    /// them. Keying by `ChatMessage.id` (UUID) survives recycling.
    @Published var expandedTraceMessageIDs: Set<UUID> = []
    @Published var expandedTelcoUnderstandingMessageIDs: Set<UUID> = []

    func isTraceExpanded(messageID: UUID) -> Bool {
        // Default-expanded for the most recent assistant message,
        // collapsed for older ones — keeps the latest result surfaced
        // without piling up cards as the conversation grows.
        if expandedTraceMessageIDs.contains(messageID) { return true }
        guard let latestAssistant = messages.last(where: { $0.role == .assistant })?.id else {
            return true
        }
        return latestAssistant == messageID
    }

    func toggleTraceExpanded(messageID: UUID) {
        if expandedTraceMessageIDs.contains(messageID) {
            expandedTraceMessageIDs.remove(messageID)
        } else {
            expandedTraceMessageIDs.insert(messageID)
        }
    }

    func isTelcoUnderstandingExpanded(messageID: UUID) -> Bool {
        expandedTelcoUnderstandingMessageIDs.contains(messageID)
    }

    func setTelcoUnderstandingExpanded(messageID: UUID, isExpanded: Bool) {
        if isExpanded {
            expandedTelcoUnderstandingMessageIDs.insert(messageID)
        } else {
            expandedTelcoUnderstandingMessageIDs.remove(messageID)
        }
    }

    /// Resolves the current brand's welcome greeting at call time. Passed
    /// as a closure so a mid-session brand flip (via Settings) propagates
    /// without ChatViewModel needing to observe BrandRegistry.
    private let welcomeGreetingProvider: @MainActor () -> String

    init(
        chatModeRouter: ChatModeRouter,
        kbExtractor: KBExtractor,
        provider: LFMChatProvider,
        piiAnalyzer: PIIAnalyzer,
        kb: KnowledgeBase,
        tokenLedger: TokenLedger,
        sessionStats: SessionStats,
        toolRegistry: ToolRegistry,
        visionAnalyzer: VisionAnalyzer,
        customerContext: CustomerContext,
        nbaEngine: NextBestActionEngine,
        queryExtractor: QueryExtractor = RegexQueryExtractor(),
        toolSelector: ToolSelector,
        toolExecutor: ToolExecutor,
        verizonDispatcher: VerizonChatDispatcher? = nil,
        telcoUnderstandingClassifier: TelcoSharedUnderstandingClassifying? = nil,
        understandingClassifier: QueryUnderstandingClassifying? = nil,
        relationalStrategy: RelationalHeadsStrategy? = nil,
        conversationState: ConversationState? = nil,
        useSimulatorFastGroundedQA: Bool = ChatViewModel.shouldUseSimulatorFastGroundedQA,
        welcomeGreetingProvider: @escaping @MainActor () -> String
    ) {
        self.chatModeRouter = chatModeRouter
        self.kbExtractor = kbExtractor
        self.provider = provider
        self.piiAnalyzer = piiAnalyzer
        self.kb = kb
        self.tokenLedger = tokenLedger
        self.sessionStats = sessionStats
        self.toolRegistry = toolRegistry
        self.visionAnalyzer = visionAnalyzer
        self.customerContext = customerContext
        self.nbaEngine = nbaEngine
        self.queryExtractor = queryExtractor
        self.toolSelector = toolSelector
        self.toolExecutor = toolExecutor
        self.verizonDispatcher = verizonDispatcher
        self.telcoUnderstandingClassifier = telcoUnderstandingClassifier
        // ADR-022 §4.3 — fall back to a composite classifier wrapping
        // the caller-supplied chatModeRouter when the host doesn't
        // pre-build one (tests, integration harnesses). The composite
        // degrades to chatMode-only when Stage A isn't reachable —
        // same behaviour as PR #30.
        self.understandingClassifier = understandingClassifier
            ?? QueryUnderstandingClassifier(
                strategy: CompositeFallbackStrategy(
                    chatModeRouter: chatModeRouter,
                    stageA: nil
                )
            )
        // ADR-028 — default to UnavailableRelationalStrategy when tests or
        // degraded integrations do not supply the packaged v4 head. The
        // blackboard reducer then applies deterministic fallback labels.
        self.relationalStrategy = relationalStrategy ?? UnavailableRelationalStrategy()
        // ADR-023 Phase 2 — default to a fresh ConversationState so the
        // session starts clean. Tests that want to pre-seed state (e.g.
        // simulate a pending clarification) inject their own instance.
        self.conversationState = conversationState ?? ConversationState()
        self.dialogueBlackboard = TelcoDialogueBlackboard()
        self.useSimulatorFastGroundedQA = useSimulatorFastGroundedQA
        self.welcomeGreetingProvider = welcomeGreetingProvider
        seedWelcomeMessage()
    }

    // MARK: - User input

    func send() {
        let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let image = attachedImage

        guard (!trimmed.isEmpty || image != nil), !isProcessing else { return }

        let userMessage = ChatMessage(
            role: .user,
            text: trimmed.isEmpty ? "[image attached]" : trimmed,
            attachedImage: image
        )
        messages.append(userMessage)
        inputText = ""
        attachedImage = nil

        Task {
            if let image {
                await processVisionQuery(query: trimmed, image: image)
            } else {
                await processTextQuery(query: trimmed)
            }
        }
    }

    func sendVoiceTranscript(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isProcessing else { return }
        inputText = ""
        let userMessage = ChatMessage(role: .user, text: trimmed, voiceInput: true)
        messages.append(userMessage)
        Task { await processTextQuery(query: trimmed) }
    }

    func attachImage(_ image: UIImage) {
        self.attachedImage = image
    }

    func clearAttachment() {
        self.attachedImage = nil
    }

    func clear() {
        messages.removeAll()
        expandedTraceMessageIDs.removeAll()
        expandedTelcoUnderstandingMessageIDs.removeAll()
        conversationState.reset()
        dialogueBlackboard = TelcoDialogueBlackboard()
        seedWelcomeMessage()
    }

    func openPrivacyShield(for message: ChatMessage) {
        let spans = message.piiSpans
        let sanitized = piiAnalyzer.redact(message.text, spans: spans)
        privacyShieldQuery = PrivacyShieldState(
            original: message.text,
            sanitized: sanitized,
            spans: spans
        )
    }

    func dismissPrivacyShield() {
        privacyShieldQuery = nil
    }

    /// "Read full article" handler — opens the `KBArticleView` sheet
    /// for the KB entry the LFM grounded its answer on. The tapped
    /// `DeepLink` is used only to resolve which message the chip
    /// belongs to; the article comes from `ChatMessage.sourceEntry`.
    func openKBArticle(_ entry: KBEntry) {
        readingArticle = entry
    }

    func dismissKBArticle() {
        readingArticle = nil
    }

    // MARK: - Pipeline

    private func seedWelcomeMessage() {
        messages.append(ChatMessage(
            role: .assistant,
            text: welcomeGreetingProvider(),
            routing: RoutingSummary(path: .answerWithRAG, toolIntent: nil, containsPII: false)
        ))
    }

    private func processTextQuery(query: String) async {
        isProcessing = true
        routingStage = .understanding
        defer {
            isProcessing = false
            routingStage = nil
        }

        let extraction = queryExtractor.extract(from: query)

        // 1) PII scan is independent of the routing decision — it
        //    always runs so the badge on the user bubble is accurate.
        //    Nothing here gates cloud egress; that lives in the
        //    out-of-scope path alone (and even then, scrubs PII).
        let piiSpans = piiAnalyzer.scan(query)
        let containsPII = !piiSpans.isEmpty
        if containsPII,
           let lastIndex = messages.lastIndex(where: { $0.role == .user }) {
            messages[lastIndex].piiSpans = piiSpans
            sessionStats.recordPII(piiSpans.count)
        }

        // ADR-023 Phase 2 — pre-classifier clarification recovery.
        // When the previous assistant turn asked a clarification
        // question (Verizon .clarification lane, or a tool-action turn
        // with a missing required slot), the user's reply should be
        // tested as the answer to that question BEFORE running the
        // full understanding classifier. Two recovery paths:
        //
        //   (a) Pending tool confirmation + bare affirmative ("yes",
        //       "go ahead") → fire the pending tool directly. The
        //       confirmation button is still primary; this path
        //       handles users who type the answer instead of tapping.
        //
        //   (b) Pending clarification with a known intent + missing
        //       slot → use RegexQueryExtractor biased toward the
        //       missing slot. If we lift a value, fire the intent
        //       with combined args.
        //
        // On either success: clear pending and return. On failure:
        // clear pending (the user changed topics) and fall through
        // to normal classification. Mis-fires cost one turn of
        // friction — missing the clarification entirely costs the
        // whole flow.
        if let pendingConfirmation = conversationState.pendingToolConfirmation,
           ConversationStateRecorder.isBareAffirmative(query) {
            let priorUserText = previousUserTurnText()
            let (turnRelation, _) = await classifyTelcoTurnRelation(
                query,
                priorUserText: priorUserText
            )
            dialogueBlackboard = TelcoDialogueBlackboardReducer.reduce(
                dialogueBlackboard,
                userTurn: query,
                observedRelation: turnRelation
            )
            guard let tool = toolRegistry.tool(id: pendingConfirmation.toolID) else {
                AppLog.intelligence.error(
                    "pending-tool-confirmation missing registered tool: \(pendingConfirmation.toolID, privacy: .public)"
                )
                conversationState.clearPendingToolConfirmation()
                appendInferenceFailure(
                    error: ToolError.notFound(pendingConfirmation.toolID),
                    mode: .toolAction,
                    containsPII: containsPII
                )
                return
            }
            AppLog.intelligence.info(
                "pending-tool-confirmation recovered: \(pendingConfirmation.toolID, privacy: .public)"
            )
            conversationState.clearPendingToolConfirmation()
            await executeConfirmedTool(
                tool: tool,
                decision: pendingConfirmation,
                triggerQuery: query
            )
            return
        }

        if conversationState.pendingToolConfirmation != nil,
           ConversationStateRecorder.isBareNegative(query) {
            let priorUserText = previousUserTurnText()
            let (turnRelation, turnRelationMS) = await classifyTelcoTurnRelation(
                query,
                priorUserText: priorUserText
            )
            dialogueBlackboard = TelcoDialogueBlackboardReducer.reduce(
                dialogueBlackboard,
                userTurn: query,
                observedRelation: turnRelation
            )
            conversationState.clearPendingToolConfirmation()

            let activePageID = dialogueBlackboard.priorPageID
            let activeLinkID = dialogueBlackboard.priorLinkID
            let text = "Okay, I won't do that."
            let message = ChatMessage(
                role: .assistant,
                text: text,
                routing: RoutingSummary(
                    path: .toolCall,
                    toolIntent: nil,
                    containsPII: containsPII
                ),
                latencyMS: turnRelationMS,
                trace: CallTrace(
                    surface: .tool,
                    inferenceMS: turnRelationMS,
                    inputTokens: 0,
                    outputTokens: 0,
                    extraction: extraction
                )
            )
            messages.append(message)
            sessionStats.recordLatency(turnRelationMS)
            dialogueBlackboard = TelcoDialogueBlackboardReducer.recordResponse(
                text,
                on: dialogueBlackboard
            )
            recordTurnSideEffects(
                query: query,
                lane: .toolAction,
                toolDecision: nil,
                pendingIntent: nil,
                missingSlots: [],
                assistantText: text,
                citedPageID: activePageID,
                citedLinkID: activeLinkID
            )
            return
        }

        if let pending = conversationState.pendingClarification,
           let recovery = tryFulfillPendingClarification(
               userReply: query,
               extraction: extraction,
               pending: pending
           ) {
            AppLog.intelligence.info(
                "pending-clarification recovered: intent=\(recovery.intent.rawValue, privacy: .public) added=\(recovery.recoveredSlotKey ?? "-", privacy: .public)"
            )
            conversationState.clearPendingClarification()
            routingStage = .preparingAction
            await runToolProposal(
                query: query,
                modePrediction: ChatModePrediction(
                    mode: .toolAction,
                    confidence: 0.95,
                    reasoning: "pending-clarification: slot recovered (\(recovery.recoveredSlotKey ?? "no slot"))",
                    runtimeMS: 0
                ),
                extraction: extraction,
                containsPII: containsPII,
                preselectedToolSelection: ToolSelection(
                    intent: recovery.intent,
                    confidence: 0.95,
                    arguments: recovery.arguments,
                    reasoning: "Recovered slot from clarification answer.",
                    runtimeMS: 0
                ),
                understanding: nil
            )
            // The post-turn recordTurn for runToolProposal will fire
            // below in the tail of the dispatch path — but since we're
            // returning early, do it explicitly here so frustration
            // counters update on this turn too.
            conversationState.recordTurn(
                userMessage: query,
                assistantLane: .toolAction,
                toolDecision: nil
            )
            return
        }

        // Clarification was pending but the reply didn't fulfil it —
        // the user changed topics. Clear and fall through.
        if conversationState.pendingClarification != nil {
            AppLog.intelligence.info("pending-clarification cleared (no match) — falling through to classifier")
            conversationState.clearPendingClarification()
        }

        // ADR-022 §4.3 Layer 0 — deterministic fast-paths that skip
        // Layer 1 entirely when the trigger is unambiguous. Only one
        // such fast-path exists today (PersonalSummaryDetector); the
        // ImperativeToolDetector lives INSIDE the .toolAction lane,
        // not before it.
        if PersonalSummaryDetector.detect(query) {
            AppLog.intelligence.info("personal-summary fast-path matched — skipping understanding layer")
            routingStage = .composing
            await runPersonalizedSummary(
                query: query,
                modePrediction: ChatModePrediction(
                    mode: .personalSummary,
                    confidence: 0.95,
                    reasoning: "deterministic personal-summary pattern match",
                    runtimeMS: 0
                ),
                extraction: extraction,
                understanding: nil
            )
            return
        }

        // ADR-026 runtime split — normal Liquid Telco support turns run
        // exactly one Liquid model pass for semantic understanding, then
        // enter the composer dispatcher. This path intentionally bypasses
        // QueryUnderstandingClassifier, chat-mode-router, Stage A LoRAs,
        // relational LoRA, and Stage B.
        if let dispatcher = verizonDispatcher {
            let priorUserText = previousUserTurnText()
            // ADR-029 §5 selective probing. An explicit, unambiguous request for
            // a human is a hard, safe policy decision on its own — neither the
            // turn-relation head nor the shared classifier can change it — so
            // both forwards are skipped on these control turns. This is
            // correctness-neutral: `TelcoPolicyEngine` escalates on the same
            // deterministic prior. (The relation head additionally self-gates on
            // turns with no prior dialogue state; see
            // `TelcoTurnRelationV4Strategy.classifyFromText`.)
            let hardControlTurn = TelcoDeterministicPrior.derive(query: query).explicitHumanRequest
            let turnRelation: TelcoTurnRelation?
            let turnRelationMS: Int
            let telcoUnderstanding: TelcoSharedUnderstanding?
            let telcoUnderstandingMS: Int
            if hardControlTurn {
                turnRelation = nil
                turnRelationMS = 0
                telcoUnderstanding = nil
                telcoUnderstandingMS = 0
            } else {
                (turnRelation, turnRelationMS) = await classifyTelcoTurnRelation(
                    query,
                    priorUserText: priorUserText
                )
                (telcoUnderstanding, telcoUnderstandingMS) = await classifyTelcoTurn(query)
            }
            dialogueBlackboard = TelcoDialogueBlackboardReducer.reduce(
                dialogueBlackboard,
                userTurn: query,
                observedRelation: turnRelation,
                understanding: telcoUnderstanding
            )
            AppLog.intelligence.info("composer runtime path: telco shared understanding + deterministic composer")
            lastUnderstanding = nil
            routingStage = .searching
            await runVerizonDispatch(
                query: query,
                modePrediction: ChatModePrediction(
                    mode: .kbQuestion,
                    confidence: 1.0,
                    reasoning: "telco-shared-clf-v1 control plane",
                    runtimeMS: telcoUnderstandingMS
                ),
                extraction: extraction,
                containsPII: containsPII,
                dispatcher: dispatcher,
                understanding: nil,
                telcoUnderstanding: telcoUnderstanding,
                preDispatchMS: telcoUnderstandingMS + turnRelationMS,
                telcoUnderstandingMS: telcoUnderstandingMS,
                retrievalContext: dialogueBlackboard.retrievalContext,
                composerOnly: true
            )
            return
        }

        // ADR-022 §4.3 Layer 1 — one shared-backbone forward pass
        // produces the full QueryUnderstanding vector (chat_mode,
        // topic_gate, refusal_flags, emotional_state, slot_completeness).
        // Strategy is fixed at boot: shared backbone when v2 is bundled,
        // composite fallback today, unavailable in degraded builds.
        //
        // ADR-024 — pairwise relational heads run as a second pass on
        // the same backbone (separate adapter), consuming cached prior
        // hidden states from `ConversationState`. The classifier
        // returns the FULL vector (single-input + relational fields).
        // Relational fields nil when prior hidden states unavailable
        // (first turn) or the relational adapter isn't bundled —
        // router handles nil per ADR-022 §4.3 principle #1.
        let understanding: QueryUnderstanding
        do {
            understanding = try await understandingClassifier.classify(query: query)
        } catch {
            AppLog.intelligence.error("understanding layer failed: \(error.localizedDescription, privacy: .public) — falling back to OOS")
            await runOutOfScope(
                query: query,
                modePrediction: ChatModePrediction(
                    mode: .outOfScope,
                    confidence: 0.0,
                    reasoning: "understanding inference error",
                    runtimeMS: 0
                ),
                extraction: extraction,
                understanding: nil
            )
            return
        }
        lastUnderstanding = understanding

        // ADR-024 Phase δ — Layer 1' relational pass. Runs AFTER the
        // single-input classifier so the shared backbone adapter is
        // already loaded (the relational adapter swap on top of it is
        // sub-millisecond from the adapter cache). On the first turn
        // `priorAssistantText` is nil → `.none` outcomes, zero cost.
        //
        // priorUserText: the user message that drove the prior assistant
        // reply. We need it for the stance_change head input format
        // (`[USER_PRIOR]` sentinel). Pull it from `messages` — the
        // current message is the last one (appended before this call),
        // so the prior user message is the second-to-last user role entry.
        let priorUserText = previousUserTurnText()

        let relationalOutcomes: RelationalOutcomes
        do {
            relationalOutcomes = try await relationalStrategy.classifyFromText(
                currentUserQuery: query,
                priorAssistantText: conversationState.priorAssistantText,
                priorUserText: priorUserText,
                runtimeState: relationalRuntimeState()
            )
        } catch {
            // Relational pass failure must never block the main turn.
            // Log and proceed with .none (single-turn baseline).
            AppLog.intelligence.error(
                "relational-pass failed: \(error.localizedDescription, privacy: .public) — falling through to single-turn baseline"
            )
            relationalOutcomes = .none
        }

        // Merge the relational outcomes into the QueryUnderstanding vector.
        // The relational pass runs on a separate adapter so its timing
        // adds to the Layer 1 total. If .none, the merge is a no-op.
        let enrichedUnderstanding: QueryUnderstanding
        if relationalOutcomes.turnRelationship != nil {
            enrichedUnderstanding = QueryUnderstanding(
                chatMode: understanding.chatMode,
                topicGate: understanding.topicGate,
                refusalFlags: understanding.refusalFlags,
                emotionalState: understanding.emotionalState,
                slotCompleteness: understanding.slotCompleteness,
                turnRelationship: relationalOutcomes.turnRelationship,
                slotAlignment: relationalOutcomes.slotAlignment,
                stanceChange: relationalOutcomes.stanceChange,
                totalMs: understanding.totalMs + relationalOutcomes.runtimeMs,
                strategy: understanding.strategy
            )
            lastUnderstanding = enrichedUnderstanding
            AppLog.intelligence.info(
                "relational-pass: \(relationalOutcomes.turnRelationship?.value.wireName ?? "none", privacy: .public) conf=\(String(format: "%.2f", relationalOutcomes.turnRelationship?.confidence ?? 0), privacy: .public) \(String(format: "%.0f", relationalOutcomes.runtimeMs), privacy: .public)ms"
            )
        } else {
            enrichedUnderstanding = understanding
        }

        // ADR-022 §4.3 Layer 2 + ADR-024 §4.6 — pure-function fused
        // router. Returns a `RoutingDecision` containing both the
        // lane AND the side-effects the workflow must perform
        // (fire pending tool, accumulate slot, augment retrieval).
        // When relational fields on `understanding` are nil (degraded
        // build OR first turn), the fusion router falls through to
        // the single-turn `decide(understanding:retrieval:)` path
        // with `actions: []` — behaviour identical to pre-ADR-024.
        let decision = VerizonUnderstandingRouter.decideMultiTurn(
            understanding: enrichedUnderstanding,
            conversation: conversationState.snapshot,
            pendingClarification: conversationState.pendingClarification,
            pendingToolConfirmation: conversationState.pendingToolConfirmation,
            retrieval: nil
        )
        let lane = decision.lane
        if !decision.actions.isEmpty {
            AppLog.intelligence.info(
                "multi-turn fusion: \(decision.reasoning, privacy: .public) actions=\(decision.actions.count, privacy: .public)"
            )
        }
        AppLog.intelligence.info(
            "understanding lane=\(lane.wireName, privacy: .public)"
        )

        // Apply each post-decision action BEFORE dispatch. Some
        // actions short-circuit the dispatch entirely (firePendingTool)
        // by returning from processTextQuery; others mutate
        // conversation state in preparation for the lane handler.
        //
        // ADR-024 follow-up 2026-05-27 — the result now carries an
        // optional RetrievalContext that the lane handler threads into
        // its ColBERT call. Was a Bool short-circuit signal; now a
        // typed envelope so `.augmentRetrievalWithPriorAssistant`
        // actually propagates instead of just logging.
        let postActions = await applyPostDecisionActions(
            decision.actions,
            query: query,
            extraction: extraction,
            containsPII: containsPII,
            understanding: enrichedUnderstanding
        )
        if postActions.shortCircuited {
            // A short-circuiting action handled the dispatch
            // (e.g. fired a pending tool). Don't run the lane handler.
            return
        }

        // ADR-022 §4.3 Layer 3 — workflow. One handler per lane;
        // the carriers (.toolAction, .personalSummary) keep their
        // existing ChatViewModel implementations so the migration is
        // structural-only, not behavioural.
        let modePrediction = enrichedUnderstanding.chatMode
            ?? Self.modePredictionFromLane(lane)

        switch lane {
        case .verizon(let verizonLane):
            // The dispatcher takes over for every Verizon lane. The
            // pre-built Stage A overload tells it not to re-run the
            // classifier (we already paid that cost in Layer 1). If
            // Stage A wasn't available (degraded build), the dispatcher
            // can't run — fall back to the legacy KB grounded-QA path.
            guard let dispatcher = verizonDispatcher,
                  let stageA = Self.stageADecisionFrom(understanding) else {
                if verizonLane == .oosRefusal {
                    await runOutOfScope(
                        query: query,
                        modePrediction: modePrediction,
                        extraction: extraction,
                        understanding: enrichedUnderstanding
                    )
                    return
                }
                routingStage = .searching
                let retrievalStart = Date()
                let citation = await kbExtractor.extract(query: query, kb: kb.entries)
                let retrievalMS = Int(Date().timeIntervalSince(retrievalStart) * 1000)
                routingStage = .composing
                await runGroundedQA(
                    query: query,
                    citation: citation,
                    modePrediction: modePrediction,
                    extraction: extraction,
                    containsPII: containsPII,
                    retrievalMS: retrievalMS,
                    understanding: enrichedUnderstanding
                )
                return
            }
            routingStage = .composing
            await runVerizonDispatch(
                query: query,
                modePrediction: modePrediction,
                extraction: extraction,
                containsPII: containsPII,
                dispatcher: dispatcher,
                understanding: enrichedUnderstanding,
                prebuiltStageA: stageA,
                prebuiltLane: verizonLane,
                retrievalContext: postActions.retrievalContext
            )

        case .toolAction:
            routingStage = .preparingAction
            await runToolProposal(
                query: query,
                modePrediction: modePrediction,
                extraction: extraction,
                containsPII: containsPII,
                understanding: enrichedUnderstanding
            )

        case .personalSummary:
            routingStage = .composing
            await runPersonalizedSummary(
                query: query,
                modePrediction: modePrediction,
                extraction: extraction,
                understanding: enrichedUnderstanding
            )
        }
    }

    /// Synthesise a `ChatModePrediction` from a resolved lane when
    /// the understanding vector didn't carry one (degraded
    /// `UnavailableStrategy` path). Keeps the downstream trace card +
    /// routing-summary surfaces populated with a meaningful mode label.
    private static func modePredictionFromLane(_ lane: UnderstandingLane) -> ChatModePrediction {
        let mode: ChatMode
        switch lane {
        case .verizon(let verizonLane):
            switch verizonLane {
            case .greeting, .ragStepByStep, .navOnlyDeeplink,
                 .unknownFeature, .clarification, .liveAgentEscalation:
                mode = .kbQuestion
            case .oosRefusal:
                mode = .outOfScope
            }
        case .toolAction:       mode = .toolAction
        case .personalSummary:  mode = .personalSummary
        }
        return ChatModePrediction(
            mode: mode,
            confidence: 0,
            reasoning: "lane-derived (no chat_mode head signal)",
            runtimeMS: 0
        )
    }

    private func dialogueRepairState() -> DialogueRepairConversationState {
        let priorText = conversationState.priorAssistantText?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let hasPriorAssistant = priorText.map { !$0.isEmpty } ?? false
        return DialogueRepairConversationState(
            priorPageID: hasPriorAssistant ? conversationState.priorPageID : nil,
            priorLinkID: hasPriorAssistant ? conversationState.priorLinkID : nil,
            pendingTool: conversationState.pendingToolConfirmation?.toolID,
            frustrationCount: conversationState.didntWorkCount,
            pendingConfirmation: conversationState.pendingToolConfirmation != nil
        )
    }

    /// Compact, deterministic view of the dialogue blackboard for the
    /// authoritative `TelcoPolicyEngine`. Built *after* this turn's relation
    /// has been reduced into the blackboard, so the remediation counters
    /// already reflect the current turn (a `repair_failed` /
    /// `repair_cannot_find` turn has already incremented its task's attempt
    /// count by the time the engine sees the snapshot).
    private func makePolicyStateSnapshot() -> TelcoDialogueStateSnapshot {
        let activeTask = dialogueBlackboard.activeTaskID
        let repairAttempts = activeTask
            .map { dialogueBlackboard.failedAttemptCountByTask[$0] ?? 0 }
            ?? dialogueBlackboard.frustrationCount
        return TelcoDialogueStateSnapshot(
            hasActiveTask: dialogueBlackboard.priorPageID != nil,
            priorPageID: dialogueBlackboard.priorPageID,
            priorLinkID: dialogueBlackboard.priorLinkID,
            pendingToolID: dialogueBlackboard.pendingToolConfirmation?.toolID,
            repairAttemptsOnActiveTask: repairAttempts,
            frustrationCount: dialogueBlackboard.frustrationCount,
            hasPriorAssistantTurn: dialogueBlackboard.lastAssistantSummary != nil,
            // ADR-029 §7: the prior turn's recorded route. Lets the state-operation
            // resolver treat a short reply to our own clarification as a
            // `clarification_answer` (grounds) rather than a fresh
            // `ambiguous_short_turn` (re-asks).
            priorRouteWasClarify: dialogueBlackboard.lastPolicyDecision?.route == .clarify
        )
    }

    private func relationalRuntimeState() -> RelationalRuntimeState {
        let pendingClarification = conversationState.pendingClarification.map { pending in
            let slots = pending.missingSlots.map(\.rawValue).sorted()
            return slots.isEmpty ? pending.source.rawValue : slots.joined(separator: ",")
        }
        let pendingTool = dialogueBlackboard.pendingToolConfirmation?.toolID
            ?? conversationState.pendingToolConfirmation?.toolID
        return RelationalRuntimeState(
            priorRoute: conversationState.priorLane?.wireName,
            priorPageID: dialogueBlackboard.priorPageID ?? conversationState.priorPageID,
            priorLinkID: dialogueBlackboard.priorLinkID ?? conversationState.priorLinkID,
            pendingTool: pendingTool,
            pendingConfirmation: pendingTool != nil,
            pendingClarification: pendingClarification,
            frustrationCount: max(dialogueBlackboard.frustrationCount, conversationState.didntWorkCount)
        )
    }

    private static func routingPath(
        for result: VerizonDispatchResult,
        fallback: RoutingPath
    ) -> RoutingPath {
        switch result.composerRoute {
        case .some(.toolAction):
            return .toolCall
        case .some(.outOfScope), .some(.noRagAnswer):
            return .outOfScope
        case .some(.ragAnswer), .some(.answerPlusAction), .some(.accountNav), .some(.liveAgent),
             .some(.clarify), .some(.greeting):
            return .answerWithRAG
        case .none:
            return fallback
        }
    }

    /// Re-shape the `QueryUnderstanding` Stage A signals into the
    /// dispatcher's `VerizonStageADecision` so we can pre-supply it
    /// and skip the dispatcher's own Stage A classifier call. Returns
    /// nil when topic_gate or refusal_flags are missing (degraded
    /// build) — the caller falls back to the legacy KB grounded-QA
    /// path.
    private static func stageADecisionFrom(_ understanding: QueryUnderstanding) -> VerizonStageADecision? {
        guard let topic = understanding.topicGate,
              let flags = understanding.refusalFlags else {
            return nil
        }
        return VerizonStageADecision(
            topicGate: topic.value,
            topicGateConfidence: topic.confidence,
            topicGateProbabilities: [],  // unused by the dispatcher; trace shows the v2 card instead
            refusalFlags: flags.value,
            refusalFlagsProbabilities: flags.probabilities.map(Float.init),
            totalMs: understanding.totalMs
        )
    }

    private func classifyTelcoTurn(_ query: String) async -> (TelcoSharedUnderstanding?, Int) {
        guard let telcoUnderstandingClassifier else {
            AppLog.intelligence.warning("telco shared understanding unavailable; composer will use retrieval evidence only")
            return (nil, 0)
        }
        do {
            let understanding = try await telcoUnderstandingClassifier.classify(query: query)
            return (understanding, Int(understanding.totalMs.rounded()))
        } catch {
            AppLog.intelligence.error("telco shared understanding failed: \(error.localizedDescription, privacy: .public)")
            return (nil, 0)
        }
    }

    private func classifyTelcoTurnRelation(
        _ query: String,
        priorUserText: String?
    ) async -> (TelcoTurnRelation?, Int) {
        do {
            let outcomes = try await relationalStrategy.classifyFromText(
                currentUserQuery: query,
                priorAssistantText: dialogueBlackboard.lastAssistantSummary,
                priorUserText: priorUserText,
                runtimeState: relationalRuntimeState()
            )
            guard let relation = outcomes.telcoTurnRelation else {
                return (nil, Int(outcomes.runtimeMs.rounded()))
            }
            AppLog.intelligence.info(
                "turn-relation: \(relation.value.rawValue, privacy: .public) conf=\(String(format: "%.2f", relation.confidence), privacy: .public) \(String(format: "%.0f", outcomes.runtimeMs), privacy: .public)ms"
            )
            return (relation.value, Int(outcomes.runtimeMs.rounded()))
        } catch {
            AppLog.intelligence.error(
                "turn-relation failed: \(error.localizedDescription, privacy: .public) — blackboard fallback will classify"
            )
            return (nil, 0)
        }
    }

    private func previousUserTurnText() -> String? {
        messages
            .dropLast()
            .last(where: { $0.role == .user })
            .map { $0.text }
    }

    private static func blackboardSummaryMatches(_ summary: String?, _ text: String) -> Bool {
        let expected = String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(240))
        return summary == expected
    }

    /// Verizon RAG dispatch — Stage A (probe-validated heads) →
    /// VerizonRagRouter → branches (Stage B for ragStepByStep, templates
    /// for the refusal / nav / live-agent lanes, KeywordKBExtractor for
    /// the unknownFeature fallback). Subscribes to the dispatcher's
    /// AsyncStream so the engineering-mode trace can render Stage A
    /// outputs + lane decision + Stage B latency live as each step
    /// completes. Appends exactly ONE ChatMessage at the end —
    /// progressive UI re-renders happen via the @Published Verizon
    /// state setters, not by appending then editing a placeholder.
    private func runVerizonDispatch(
        query: String,
        modePrediction: ChatModePrediction,
        extraction: ExtractionResult,
        containsPII: Bool,
        dispatcher: VerizonChatDispatcher,
        understanding: QueryUnderstanding? = nil,
        telcoUnderstanding: TelcoSharedUnderstanding? = nil,
        preDispatchMS: Int = 0,
        telcoUnderstandingMS: Int? = nil,
        prebuiltStageA: VerizonStageADecision? = nil,
        prebuiltLane: VerizonLane? = nil,
        retrievalContext: RetrievalContext = .empty,
        composerOnly: Bool = false
    ) async {
        // Reset trace state for this turn so a stale Stage B response
        // from the previous turn doesn't render under this bubble while
        // the new dispatch is still in flight.
        lastVerizonStageA = nil
        lastVerizonLane = nil
        lastVerizonStageBResponse = nil
        lastVerizonResult = nil

        let dispatchStart = Date()
        var finalResult: VerizonDispatchResult?
        var finalErrorMessage: String?

        // ADR-022 §4.3 Layer 1 → Layer 3: when caller pre-built the
        // Stage A signal + lane (the v2 understanding path), hand them
        // to the dispatcher so it doesn't re-run Stage A. The
        // dispatcher's progressive trace still emits .stageAComplete
        // and .laneSelected so the existing UI doesn't break.
        let dispatchStream: AsyncStream<VerizonDispatchEvent> = {
            if composerOnly {
                return dispatcher.dispatchComposer(
                    query: query,
                    retrievalContext: retrievalContext,
                    telcoUnderstanding: telcoUnderstanding,
                    dialogueState: dialogueRepairState(),
                    turnRelation: dialogueBlackboard.lastTurnRelation,
                    policyState: makePolicyStateSnapshot()
                )
            } else if let prebuiltStageA, let prebuiltLane {
                return dispatcher.dispatch(
                    query: query,
                    prebuiltStageA: prebuiltStageA,
                    prebuiltLane: prebuiltLane,
                    retrievalContext: retrievalContext
                )
            }
            return dispatcher.dispatch(
                query: query,
                retrievalContext: retrievalContext
            )
        }()

        for await event in dispatchStream {
            switch event {
            case .stageAStarted:
                routingStage = .understanding
            case .stageAComplete(let stageA):
                lastVerizonStageA = stageA
            case .laneSelected(let lane):
                lastVerizonLane = lane
                routingStage = lane.requiresGeneration ? .composing : .searching
            case .retrievalStarted:
                routingStage = .searching
            case .retrievalComplete(let result):
                // Latest retrieval result observable for the
                // engineering trace (Phase 4 will render it).
                AppLog.intelligence.info(
                    "retrieval top=\(result.hits.first?.chunk.chunkID ?? "<none>", privacy: .public) conf=\(String(format: "%.3f", result.topConfidence), privacy: .public) gap=\(String(format: "%.3f", result.topGap), privacy: .public) elapsed=\(String(format: "%.0f", result.elapsedMs), privacy: .public)ms"
                )
            case .stageBStarted:
                routingStage = .composing
            case .stageBComplete(let response):
                lastVerizonStageBResponse = response
            case .faithfulnessChecked(let score):
                AppLog.intelligence.info(
                    "faithfulness jaccard=\(String(format: "%.3f", score.bigramJaccard), privacy: .public) floor=\(String(format: "%.2f", score.floor), privacy: .public) faithful=\(score.isFaithful, privacy: .public)"
                )
            case .fallbackInvoked(let reason):
                AppLog.intelligence.info("Verizon dispatcher fallback: \(reason, privacy: .public)")
            case .response(let result):
                finalResult = result
            case .failed(let message):
                finalErrorMessage = message
            }
        }

        let dispatchLatencyMS = Int(Date().timeIntervalSince(dispatchStart) * 1000)

        if let finalResult {
            lastVerizonResult = finalResult
            let dispatchWallMS = finalResult.totalMs > 0 ? finalResult.totalMs : Double(dispatchLatencyMS)
            let totalWallMS = Int((dispatchWallMS + Double(preDispatchMS)).rounded())
            let retrievalMS = finalResult.retrievalMs.map { Int($0.rounded()) }
            let routePolicyMS = finalResult.routePolicyMs.map { Int($0.rounded()) }
            let composerMS = finalResult.composerMs.map { Int($0.rounded()) }
            let answerMS = composerMS ?? dispatchLatencyMS
            let displayText = finalResult.text
            // Citation chip — composer, Stage B, and KB fallback all
            // produce evidence, just in different envelopes. The UI
            // wants one KBEntry-shaped value so the source chip and
            // article sheet stay path-agnostic.
            let citationEntry: KBEntry?
            if let chunk = finalResult.retrievedChunk {
                citationEntry = Self.makeCitationEntry(from: chunk)
            } else if let unit = finalResult.citedRAGUnit {
                citationEntry = Self.makeCitationEntry(from: unit)
            } else if let entry = finalResult.kbEntry {
                citationEntry = entry
            } else {
                citationEntry = nil
            }
            let resolvedLane = lastVerizonLane.map { UnderstandingLane.verizon($0) }
                ?? UnderstandingLane.verizon(.ragStepByStep)
            var message = ChatMessage(
                role: .assistant,
                text: displayText,
                routing: RoutingSummary(
                    path: Self.routingPath(for: finalResult, fallback: modePrediction.mode.routingPath),
                    toolIntent: nil,
                    containsPII: containsPII,
                    confidence: modePrediction.confidence
                ),
                sourceEntry: citationEntry,
                deepLinks: finalResult.deepLink.map {
                    [DeepLink(
                        label: finalResult.citedRAGUnit?.displayLabel ?? "Open in app",
                        url: $0
                    )]
                } ?? [],
                latencyMS: totalWallMS,
                trace: CallTrace(
                    surface: .onDeviceRAG,
                    retrievalMS: retrievalMS,
                    inferenceMS: answerMS,
                    topKBMatchID: finalResult.citedRAGUnit?.pageID,
                    topKBScore: finalResult.citedRAGUnit == nil ? nil : 1.0,
                    kbEntriesScanned: kb.entries.count,
                    inputTokens: 0,
                    outputTokens: lastVerizonStageBResponse?.outputTokens ?? 0,
                    chatMode: modePrediction.mode,
                    chatModeConfidence: modePrediction.confidence,
                    chatModeRuntimeMS: modePrediction.runtimeMS,
                    extraction: extraction,
                    understanding: understanding,
                    telcoUnderstanding: telcoUnderstanding,
                    telcoUnderstandingMS: telcoUnderstandingMS ?? (preDispatchMS > 0 ? preDispatchMS : nil),
                    routePolicyMS: routePolicyMS,
                    composerMS: composerMS,
                    totalWallMS: totalWallMS,
                    // Step 6.6 composer telemetry — nil on legacy
                    // Stage B / kbFallback paths, populated on the
                    // composer path so engineering-mode trace can
                    // surface the route + cited page + confirmation flag.
                    composerRoute: finalResult.composerRoute?.wireName,
                    composerCitedPageID: finalResult.citedRAGUnit?.pageID,
                    composerRenderedLinkID: finalResult.citedRAGUnit?.linkID,
                    composerConfirmationShown: finalResult.requiresConfirmation,
                    reuseActiveEvidence: finalResult.reuseActiveEvidence,
                    policyReason: finalResult.policyReason,
                    stateOperation: finalResult.stateOperation,
                    stateOperationReason: finalResult.stateOperationReason
                )
            )
            let composerToolDecision = composerExecutableToolDecision(
                from: finalResult,
                extraction: extraction
            )
            if finalResult.composerRoute == .toolAction {
                message.toolDecision = composerToolDecision
            }
            recordDialogueBlackboardDispatch(
                result: finalResult,
                assistantText: message.text,
                pendingToolConfirmation: composerToolDecision
            )
            // Compound RAG + tool — attach a "Want me to do this?" card
            // when the Verizon lane carries content (RAG/unknown/clarification)
            // and the user's query is an unambiguous imperative.
            maybeAttachCompoundTool(
                to: &message,
                query: query,
                extraction: extraction,
                lane: resolvedLane
            )
            attachNBAIfAvailable(
                to: &message,
                query: query,
                understanding: understanding,
                lane: resolvedLane,
                toolIntent: nil
            )
            messages.append(message)
            sessionStats.recordLatency(totalWallMS)
            recordTurnSideEffects(
                query: query,
                lane: resolvedLane,
                toolDecision: message.toolDecision,
                pendingToolConfirmation: composerToolDecision,
                pendingIntent: nil,
                missingSlots: [],
                assistantText: message.text,
                // Step 5b Pre-flight Fix C iOS-integration follow-up —
                // carry the cited RAG unit forward so the next turn's
                // short-followup override can reuse it. Both nil when
                // the composer produced no citation (greeting, OOS,
                // live-agent, clarify, ambiguous-yes-ignored).
                citedPageID: finalResult.citedRAGUnit?.pageID,
                citedLinkID: finalResult.citedRAGUnit?.linkID
            )
            return
        }

        // No final result + a failure message → dispatcher errored out
        // early. Use the existing inference-failure path so the user
        // sees a consistent error bubble across the legacy and new
        // routes.
        let error = NSError(
            domain: "VerizonChatDispatcher",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: finalErrorMessage ?? "Verizon dispatcher failed without a result"]
        )
        appendInferenceFailure(
            error: error,
            mode: modePrediction.mode,
            containsPII: containsPII
        )
    }

    private func recordDialogueBlackboardDispatch(
        result: VerizonDispatchResult,
        assistantText: String,
        pendingToolConfirmation: ToolDecision?
    ) {
        let citedUnit = result.citedRAGUnit
        let policyDecision = result.composerRoute.map {
            TelcoPolicyDecision(
                route: $0,
                requiresConfirmation: result.requiresConfirmation ?? false,
                handoff: $0 == .liveAgent ? "live_agent" : nil
            )
        }

        dialogueBlackboard = TelcoDialogueBlackboardReducer.recordRetrievalAndPolicy(
            on: dialogueBlackboard,
            retrievalCandidates: result.retrievalCandidates,
            selectedPageID: citedUnit?.pageID,
            selectedLinkID: citedUnit?.linkID,
            selectedTitle: citedUnit?.displayLabel,
            policyDecision: policyDecision
        )

        if let pendingToolConfirmation,
           pendingToolConfirmation.requiresConfirmation {
            let pending = TelcoPendingTool(
                toolID: pendingToolConfirmation.toolID,
                intent: pendingToolConfirmation.intent,
                pageID: citedUnit?.pageID,
                linkID: citedUnit?.linkID
            )
            dialogueBlackboard = TelcoDialogueBlackboardReducer.setPendingTool(
                pending,
                on: dialogueBlackboard
            )
        }

        dialogueBlackboard = TelcoDialogueBlackboardReducer.recordResponse(
            assistantText,
            on: dialogueBlackboard
        )
    }

    private func runGroundedQA(
        query: String,
        citation: KBCitation,
        modePrediction: ChatModePrediction,
        extraction: ExtractionResult,
        containsPII: Bool,
        retrievalMS: Int,
        understanding: QueryUnderstanding? = nil
    ) async {
        // Resolve the cited KB entry if the extractor returned a
        // match. A `.noMatch` citation or a hallucinated id (already
        // guarded by `LFMKBExtractor`) falls back to the synthetic
        // "no match" stub so the grounded-QA prompt still runs —
        // the prompt tells the model to say "no matching article"
        // when the reference doesn't fit.
        let topEntry: KBEntry
        if citation.entryId == EmbeddingKBExtractor.loadingEntryID {
            // KB embedding index is still building (cold-start window,
            // ~5s on first install). Surface this as a transient
            // "warming up" message instead of a flat noMatch — the
            // user retries and gets a real answer seconds later.
            topEntry = warmingUpFallbackEntry()
        } else if citation.isMatch, let entry = kb.entries.first(where: { $0.id == citation.entryId }) {
            topEntry = entry
        } else {
            topEntry = fallbackEntryForEmptyKB()
        }
        if useSimulatorFastGroundedQA {
            let displayText = Self.compactGroundedAnswer(topEntry.answer)
            let inputTokens = TokenEstimator.estimate(query)
            let outputTokens = TokenEstimator.estimate(displayText)
            let visibleMS = modePrediction.runtimeMS + retrievalMS
            tokenLedger.recordOnDevice(inputTokens: inputTokens, outputTokens: outputTokens)

            var message = ChatMessage(
                role: .assistant,
                text: displayText,
                routing: RoutingSummary(
                    path: modePrediction.mode.routingPath,
                    toolIntent: nil,
                    containsPII: containsPII,
                    confidence: modePrediction.confidence
                ),
                sourceEntry: citation.isMatch ? topEntry : nil,
                deepLinks: topEntry.deepLinks,
                latencyMS: visibleMS,
                trace: CallTrace(
                    surface: .onDeviceRAG,
                    retrievalMS: retrievalMS,
                    inferenceMS: 0,
                    topKBMatchID: citation.isMatch ? citation.entryId : nil,
                    topKBScore: citation.isMatch ? citation.confidence : nil,
                    kbEntriesScanned: kb.entries.count,
                    inputTokens: inputTokens,
                    outputTokens: outputTokens,
                    chatMode: modePrediction.mode,
                    chatModeConfidence: modePrediction.confidence,
                    chatModeRuntimeMS: modePrediction.runtimeMS,
                    extraction: extraction,
                    understanding: understanding
                )
            )
            maybeAttachCompoundTool(
                to: &message,
                query: query,
                extraction: extraction,
                lane: .verizon(.ragStepByStep)
            )
            attachNBAIfAvailable(
                to: &message,
                query: query,
                understanding: understanding,
                lane: .verizon(.ragStepByStep),
                toolIntent: nil
            )
            messages.append(message)
            sessionStats.recordLatency(visibleMS)
            recordTurnSideEffects(
                query: query,
                lane: .verizon(.ragStepByStep),
                toolDecision: message.toolDecision,
                pendingIntent: nil,
                missingSlots: [],
                assistantText: message.text
            )
            return
        }
        do {
            let response = try await provider.generate(
                query: query,
                mode: .groundedQA(topEntry: topEntry)
            )
            tokenLedger.recordOnDevice(
                inputTokens: response.inputTokens,
                outputTokens: response.outputTokens
            )
            // Some queries ("How do I restart?") cause the base model
            // to emit a 2-word echo of the KB topic and stop. That
            // renders worse than showing the KB entry verbatim, which
            // is the same content the user can read in the full
            // article. Fall back when the generation is visibly too
            // terse to be a real answer.
            let displayText = Self.isTerseGeneration(response.text)
                ? Self.firstParagraph(of: topEntry.answer)
                : response.text
            var message = ChatMessage(
                role: .assistant,
                text: displayText,
                routing: RoutingSummary(
                    path: modePrediction.mode.routingPath,
                    toolIntent: nil,
                    containsPII: containsPII,
                    confidence: modePrediction.confidence
                ),
                sourceEntry: citation.isMatch ? topEntry : nil,
                deepLinks: response.deepLinks,
                latencyMS: modePrediction.runtimeMS + retrievalMS + response.latencyMS,
                trace: CallTrace(
                    surface: .onDeviceRAG,
                    retrievalMS: retrievalMS,
                    inferenceMS: response.latencyMS,
                    topKBMatchID: citation.isMatch ? citation.entryId : nil,
                    topKBScore: citation.isMatch ? citation.confidence : nil,
                    kbEntriesScanned: kb.entries.count,
                    inputTokens: response.inputTokens,
                    outputTokens: response.outputTokens,
                    chatMode: modePrediction.mode,
                    chatModeConfidence: modePrediction.confidence,
                    chatModeRuntimeMS: modePrediction.runtimeMS,
                    extraction: extraction,
                    understanding: understanding
                )
            )
            maybeAttachCompoundTool(
                to: &message,
                query: query,
                extraction: extraction,
                lane: .verizon(.ragStepByStep)
            )
            attachNBAIfAvailable(
                to: &message,
                query: query,
                understanding: understanding,
                lane: .verizon(.ragStepByStep),
                toolIntent: nil
            )
            messages.append(message)
            sessionStats.recordLatency(message.trace?.customerVisibleMS ?? response.latencyMS)
            recordTurnSideEffects(
                query: query,
                lane: .verizon(.ragStepByStep),
                toolDecision: message.toolDecision,
                pendingIntent: nil,
                missingSlots: [],
                assistantText: message.text
            )
        } catch {
            appendInferenceFailure(error: error, mode: modePrediction.mode, containsPII: containsPII)
        }
    }

    private func runToolProposal(
        query: String,
        modePrediction: ChatModePrediction,
        extraction: ExtractionResult,
        containsPII: Bool,
        preselectedToolSelection: ToolSelection? = nil,
        understanding: QueryUnderstanding? = nil
    ) async {
        // Tool selection routes on the query alone. The generative-
        // retrieval architecture doesn't pre-fetch a KB entry for the
        // action branch, and the tool-selector prompt never used one.
        //
        // Fast-path: when the imperative literally names a tool ("run
        // diagnostics", "restart my router", "pause my son's tablet"),
        // ImperativeToolDetector returns a tool intent in O(μs) and we
        // skip the ~1.7 s LFMToolSelector LFM call. Question forms are
        // explicitly rejected by the detector so KB lookups aren't
        // hijacked. Ambiguous phrasings fall through to the LFM
        // selector for full argument extraction + confidence.
        let toolSelection: ToolSelection
        if let preselectedToolSelection {
            toolSelection = preselectedToolSelection
        } else if let imperativeIntent = ImperativeToolDetector.detect(query) {
            AppLog.intelligence.info("imperative tool fast-path matched: \(imperativeIntent.rawValue, privacy: .public) — skipping LFMToolSelector")
            toolSelection = ToolSelection(
                intent: imperativeIntent,
                confidence: 0.95,
                arguments: imperativeArguments(intent: imperativeIntent, extraction: extraction),
                reasoning: "deterministic imperative pattern match",
                runtimeMS: 0
            )
        } else {
            toolSelection = await toolSelector.select(
                query: query,
                extraction: extraction,
                availableTools: toolRegistry.all
            )
        }

        guard let intent = toolSelection.intent,
              let tool = toolRegistry.tool(for: intent) else {
            // Mode router said "action" but the tool selector
            // didn't lock in a tool. Fall through to the question
            // branch — run KB extraction and ground an answer
            // instead. Graceful handoff between LFM-backed
            // primitives, no lexical fallback.
            let retrievalStart = Date()
            let citation = await kbExtractor.extract(query: query, kb: kb.entries)
            let retrievalMS = Int(Date().timeIntervalSince(retrievalStart) * 1000)
            await runGroundedQA(
                query: query,
                citation: citation,
                modePrediction: modePrediction,
                extraction: extraction,
                containsPII: containsPII,
                retrievalMS: retrievalMS,
                understanding: understanding
            )
            return
        }

        let args = toolSelection.arguments
        // Deterministic one-liner instead of calling the 350M base for
        // framing. The tool card + sheet already carry all the info the
        // customer needs; asking the base to paraphrase the prompt
        // template leaked the scaffolding on TestFlight build 14
        // ("Arguments: (no arguments)\n\nOne-sentence confirmation
        // prompt:"). A hand-written framing is faster, trustworthy, and
        // exec-safe.
        let framingText = Self.toolProposalFraming(tool: tool, arguments: args.values)
        tokenLedger.recordDeflection()

        let decisionPayload = ToolDecision(
            intent: intent,
            toolID: tool.id,
            displayName: tool.displayName,
            icon: tool.icon,
            description: tool.description,
            arguments: Self.formatArguments(args),
            confidence: toolSelection.confidence,
            reasoning: toolSelection.reasoning.isEmpty ? nil : toolSelection.reasoning,
            requiresConfirmation: tool.requiresConfirmation,
            isDestructive: tool.isDestructive
        )

        // Real on-device LFM time that the user waited on: ChatModeRouter
        // (classify the 4-way gate) + ToolSelector (pick tool + extract
        // args). Both are LFM calls with LoRA adapter swaps. The final
        // framing sentence is deterministic, so no third inference to add.
        let onDeviceMS = modePrediction.runtimeMS + toolSelection.runtimeMS
        var message = ChatMessage(
            role: .assistant,
            text: framingText,
            routing: RoutingSummary(
                path: .toolCall,
                toolIntent: intent,
                containsPII: containsPII,
                confidence: modePrediction.confidence
            ),
            sourceEntry: nil,
            deepLinks: tool.deepLink.map { [$0] } ?? [],
            latencyMS: onDeviceMS,
            toolDecision: decisionPayload,
            trace: CallTrace(
                surface: .tool,
                retrievalMS: nil,
                inferenceMS: onDeviceMS,
                topKBMatchID: nil,
                topKBScore: nil,
                kbEntriesScanned: kb.entries.count,
                inputTokens: 0,
                outputTokens: 0,
                chatMode: modePrediction.mode,
                chatModeConfidence: modePrediction.confidence,
                chatModeRuntimeMS: modePrediction.runtimeMS,
                extraction: extraction,
                toolSelectionReasoning: toolSelection.reasoning.isEmpty ? nil : toolSelection.reasoning,
                toolSelectionConfidence: toolSelection.confidence,
                understanding: understanding
            )
        )
        attachNBAIfAvailable(
            to: &message,
            query: query,
            understanding: understanding,
            lane: .toolAction,
            toolIntent: intent
        )
        messages.append(message)
        sessionStats.recordLatency(onDeviceMS)

        // ADR-023 Phase 2 — record turn so the missing-slot case sets
        // `pendingClarification` for the NEXT turn (the user typing
        // the device name back is the recovery path). When the
        // slot_completeness head is bundled, this drives the
        // multi-turn slot-fill flow without needing a separate state
        // machine.
        let missingSlots = ToolSlotRequirements.missingSlots(
            for: intent,
            given: understanding?.slotCompleteness?.value
        )
        let supportContext = Self.supportPageContext(for: intent)
        recordTurnSideEffects(
            query: query,
            lane: .toolAction,
            toolDecision: decisionPayload,
            pendingIntent: intent,
            missingSlots: missingSlots,
            assistantText: message.text,
            citedPageID: supportContext?.pageID,
            citedLinkID: supportContext?.linkID
        )
    }

    /// RAG page context associated with executable tools. Tool proposal
    /// turns are not composed from `RAGUnit`, but the UI still shows a
    /// support link and the next turn may ask "how do I do it?". Cache
    /// the corresponding canonical page so the dispatcher can reuse it
    /// for short/anaphoric follow-ups instead of treating the next turn
    /// as an unrelated retrieval.
    nonisolated static func supportPageContext(
        for intent: ToolIntent
    ) -> (pageID: String, linkID: String)? {
        switch intent {
        case .restartRouter:
            return ("02.07", "restart-router")
        case .runSpeedTest:
            return ("01.02", "speed-test")
        case .checkConnection, .runDiagnostics:
            return ("01.01", "troubleshoot")
        case .wpsPair:
            return ("02.08", "equipment-wps")
        case .toggleParentalControls:
            return ("13.00", "home")
        case .rebootExtender:
            return ("02.00", "equipment")
        case .scheduleTechnician:
            return nil
        }
    }

    private static func toolProposalFraming(tool: Tool, arguments: [String: String]) -> String {
        func arg(_ key: String) -> String? {
            guard let v = arguments[key], !v.isEmpty, v != "all" else { return nil }
            return v
        }
        switch tool.id {
        case "restart-router":
            return "I'll restart your router — connection drops for about 45 seconds."
        case "run-speed-test":
            return "I'll run a speed test now."
        case "check-connection":
            return "I'll check your connection status."
        case "enable-wps":
            return "I'll open a WPS pairing window for 2 minutes."
        case "run-diagnostics":
            return "I'll run diagnostics on your home network."
        case "schedule-technician":
            let when = arg("preferred_date") ?? "the next available slot"
            return "I'll schedule a technician for \(when)."
        case "toggle-parental-controls":
            let device = arg("target_device") ?? "the selected device"
            let action = arg("action") ?? "pause_internet"
            switch action {
            case "pause_internet": return "I'll pause internet for \(device)."
            case "enable":         return "I'll turn on parental controls for \(device)."
            case "disable":        return "I'll turn off parental controls for \(device)."
            default:               return "I'll update parental controls for \(device)."
            }
        case "reboot-extender":
            // Model extracts bare location like "upstairs" / "basement";
            // insert "the … extender" so the sentence reads naturally.
            if let location = arg("extender_name") {
                return "I'll reboot the \(location) extender."
            }
            return "I'll reboot your extender."
        default:
            return "I'll \(tool.displayName.lowercased()) now."
        }
    }

    private func runPersonalizedSummary(
        query: String,
        modePrediction: ChatModePrediction,
        extraction: ExtractionResult,
        understanding: QueryUnderstanding? = nil
    ) async {
        let profile = customerContext.profile

        // Billing is a distinct shape of "personal summary" that the
        // generic profileSummary prompt doesn't cover — the template
        // frames the task as "state of their home network" and doesn't
        // expose monthlyPrice. On the 350M base the likely outputs are
        // (a) a network summary that ignores the bill question or
        // (b) a hallucinated amount. Both are pitch-breaking for the
        // #1 call-driver category.
        //
        // Deterministic billing answer populated from real profile
        // fields until the F7 get-bill tool lands. See FEATURES.yaml
        // F7 for the v2 agentic replacement.
        // Every personal_summary query goes through a deterministic
        // responder. Billing short-circuits to billingResponse;
        // everything else gets personalSummaryResponse. The 350M base
        // cannot reliably summarize raw profile fields — it echoes them
        // (verified via scripts/test_telco_chat_pipeline_local.py). See F8 in
        // FEATURES.yaml for the v2 plan that reintroduces LFM-generated
        // summaries once we have a summarizer adapter.
        let text: String =
            Self.isBillingQuery(query)
                ? Self.billingResponse(profile: profile)
                : Self.personalSummaryResponse(query: query, profile: profile)

        // The ChatModeRouter is a real on-device LFM inference — its
        // latency is what the user actually waited on, and it's what
        // should show in the "On-device · …ms" badge. The final text
        // is composed in Swift from the profile data (see F8 for the
        // v2 summarizer that replaces the Swift-side composer).
        let message = ChatMessage(
            role: .assistant,
            text: text,
            routing: RoutingSummary(
                path: .personalized,
                toolIntent: nil,
                containsPII: false,
                confidence: modePrediction.confidence
            ),
            latencyMS: modePrediction.runtimeMS,
            trace: CallTrace(
                surface: .onDeviceRAG,
                retrievalMS: nil,
                inferenceMS: modePrediction.runtimeMS,
                inputTokens: 0,
                outputTokens: 0,
                chatMode: modePrediction.mode,
                chatModeConfidence: modePrediction.confidence,
                chatModeRuntimeMS: modePrediction.runtimeMS,
                extraction: extraction,
                understanding: understanding
            )
        )
        tokenLedger.recordDeflection()
        messages.append(message)
        sessionStats.recordLatency(modePrediction.runtimeMS)
        recordTurnSideEffects(
            query: query,
            lane: .personalSummary,
            toolDecision: nil,
            pendingIntent: nil,
            missingSlots: [],
            assistantText: message.text
        )
    }

    private func runOutOfScope(
        query: String,
        modePrediction: ChatModePrediction,
        extraction: ExtractionResult,
        understanding: QueryUnderstanding? = nil
    ) async {
        // ChatModeRouter already classified this as out_of_scope with
        // high confidence. Asking the 350M base to compose a refusal
        // sometimes made it answer the question anyway ("what's the
        // weather today" → "The weather is good." — observed on
        // TestFlight build 14). That breaks the privacy/safety story
        // the pitch is built on.
        //
        // Deterministic refusal instead. No model call, no risk of
        // the base model slipping into a helpful-assistant pattern,
        // sub-50ms latency, and the boundary is auditable in code.
        // The ChatModeRouter inference is what the user actually waited
        // on — it's the on-device LFM call that decided this was
        // out-of-scope. Showing its runtime in the badge (rather than
        // 0ms) is honest about what the model did.
        let message = ChatMessage(
            role: .assistant,
            text: Self.outOfScopeRefusal,
            routing: RoutingSummary(
                path: .outOfScope,
                toolIntent: nil,
                containsPII: false,
                confidence: modePrediction.confidence
            ),
            latencyMS: modePrediction.runtimeMS,
            trace: CallTrace(
                surface: .onDeviceRAG,
                retrievalMS: nil,
                inferenceMS: modePrediction.runtimeMS,
                inputTokens: 0,
                outputTokens: 0,
                chatMode: modePrediction.mode,
                chatModeConfidence: modePrediction.confidence,
                chatModeRuntimeMS: modePrediction.runtimeMS,
                extraction: extraction,
                understanding: understanding
            )
        )
        messages.append(message)
        sessionStats.recordLatency(modePrediction.runtimeMS)
        recordTurnSideEffects(
            query: query,
            lane: .verizon(.oosRefusal),
            toolDecision: nil,
            pendingIntent: nil,
            missingSlots: [],
            assistantText: message.text
        )
    }

    private static let outOfScopeRefusal =
        "That's outside what I handle - I only cover home internet support. " +
        "Your question stayed on this phone; nothing was sent to the cloud."

    /// Argument extraction for the ImperativeToolDetector fast-path.
    /// When the detector returns a tool intent in O(μs), we still need
    /// arguments (target device for parental controls, location hint
    /// for extender reboot, etc.) — pull them from the already-computed
    /// `ExtractionResult` rather than re-running an LFM.
    private func imperativeArguments(
        intent: ToolIntent,
        extraction: ExtractionResult
    ) -> ToolArguments {
        switch intent {
        case .toggleParentalControls:
            var values: [String: String] = ["action": "pause_internet"]
            if let target = extraction.targetDevice {
                values["target_device"] = target
            }
            return ToolArguments(values)
        case .rebootExtender:
            var values: [String: String] = [:]
            if let location = extraction.locationHint {
                values["extender_name"] = location
            }
            return ToolArguments(values)
        default:
            // restartRouter / runDiagnostics / runSpeedTest /
            // checkConnection / wpsPair / scheduleTechnician all take
            // no required args in the iOS tool registry — the tool sheet
            // handles any missing inputs interactively.
            return .empty
        }
    }

    /// Synthesize a `KBEntry`-shaped citation from a ColBERT chunk so
    /// the existing "Read full article" chip in `ChatMessageRow` renders
    /// for Verizon RAG bubbles. The chunk's section name becomes the
    /// citation topic ("Network > Wi-Fi password"), its body becomes
    /// the article text, and its canonical deep link (if present)
    /// rides along as the in-app deep link.
    private static func makeCitationEntry(from chunk: ColBERTChunk) -> KBEntry {
        let section = chunk.section.isEmpty ? "Verizon Home Internet" : chunk.section
        let category = section.components(separatedBy: " > ").first ?? "Verizon"
        let deepLinks: [DeepLink] = chunk.deepLink.map {
            [DeepLink(label: chunk.deepLinkLabel ?? "Open in app", url: $0)]
        } ?? []
        return KBEntry(
            id: chunk.pageID,
            topic: section,
            aliases: [],
            category: category,
            answer: chunk.body,
            deepLinks: deepLinks,
            tags: [],
            requiresToolExecution: false
        )
    }

    /// Synthesize a `KBEntry`-shaped citation from the canonical
    /// composer RAG unit. This is the current production path: the
    /// deterministic composer renders from `RAGUnit`, while the chat UI
    /// already knows how to display source chips and article sheets for
    /// `KBEntry`.
    private static func makeCitationEntry(from unit: RAGUnit) -> KBEntry {
        let category = unit.section.isEmpty ? "Liquid Telco" : unit.section
        let stepText: String
        if unit.steps.isEmpty {
            stepText = ""
        } else {
            stepText = unit.steps.enumerated()
                .map { "\($0.offset + 1). \($0.element)" }
                .joined(separator: "\n")
        }
        let answer = [stepText, unit.body]
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .joined(separator: "\n\n")
        return KBEntry(
            id: unit.pageID,
            topic: unit.displayLabel,
            aliases: unit.aliases,
            category: category,
            answer: answer,
            deepLinks: [DeepLink(label: unit.displayLabel, url: unit.canonicalURL)],
            tags: [unit.section, unit.linkID, unit.pageID].filter { !$0.isEmpty },
            requiresToolExecution: unit.actionAffordance == "tool_action"
        )
    }

    /// True when the base model's grounded-QA generation is visibly too
    /// terse to have answered the customer meaningfully. The threshold
    /// is intentionally conservative — we only reject responses that
    /// are unambiguously degenerate (fewer than 8 words AND fewer than
    /// 40 characters). A legit short response like "Your router is
    /// offline — restart it." (6 words but > 40 chars) still passes.
    static func isTerseGeneration(_ text: String) -> Bool {
        let wordCount = text.split(whereSeparator: { $0.isWhitespace }).count
        return wordCount < 8 && text.count < 40
    }

    /// First paragraph (or first ~400 chars) of a KB entry's answer,
    /// trimmed at the last full sentence so we don't render a hanging
    /// clause. Used as the grounded-QA fallback when the base model
    /// produces a degenerate response.
    static func firstParagraph(of answer: String, maxChars: Int = 400) -> String {
        if let blank = answer.range(of: "\n\n") {
            let para = String(answer[..<blank.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !para.isEmpty { return para }
        }
        if answer.count <= maxChars {
            return answer.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let head = String(answer.prefix(maxChars))
        if let lastPeriod = head.lastIndex(of: ".") {
            return String(head[...lastPeriod])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return head.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// True when the query is asking about billing / balance / payment.
    /// ChatModeRouter already routed us to personal_summary; this
    /// second-level gate picks the billing sub-shape out so we can
    /// answer with real profile data instead of letting the generic
    /// network-health prompt swallow it.
    ///
    /// Scoped tight to avoid collisions: "bill" / "billing" only
    /// (not "billy"), "charge/charged" only as whole words, and
    /// whole-phrase "amount due" / "due date". "Pay" alone is
    /// intentionally excluded — too ambiguous ("pay attention",
    /// "paying a visit").
    static func isBillingQuery(_ query: String) -> Bool {
        return Self.billingKeywords.firstMatch(
            in: query,
            range: NSRange(query.startIndex..., in: query)
        ) != nil
    }

    private static let billingKeywords: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(
            pattern: #"\b(bill|billing|balance|invoice|owe|owed|payment|payments|charged?|amount\s+due|due\s+date|how\s+much\s+do\s+i\s+owe)\b"#,
            options: [.caseInsensitive]
        )
    }()

    /// Deterministic household summary populated from the profile.
    /// The 350M base, given raw profile fields + "write a summary,"
    /// pattern-locks on the key:value list and echoes it. Rather than
    /// risk that on stage, render prose in code. Covered by the local
    /// harness (scripts/test_telco_chat_pipeline_local.py).
    ///
    /// Query parameter is reserved for future angle-switching (devices
    /// vs wifi-health vs plan). Today all three converge on one
    /// comprehensive summary.
    static func personalSummaryResponse(query: String, profile: CustomerProfile) -> String {
        _ = query
        let plan = profile.plan
        let first = profile.firstName

        let devices = profile.usage.connectedDeviceCount
        let peak = profile.usage.peakDeviceCount

        let unhealthy = profile.equipment.filter { $0.status == .unhealthy }
        let online = profile.equipment.filter { $0.status == .online }

        var sentences: [String] = []
        sentences.append(
            "\(first), you're on the \(plan.name) plan at " +
            "\(plan.downSpeedMbps)/\(plan.upSpeedMbps) Mbps."
        )
        if !unhealthy.isEmpty {
            let names = unhealthy.map(\.model).joined(separator: " and ")
            let onlineCount = online.count
            let noun = onlineCount == 1 ? "other piece of equipment is" : "rest of your equipment is"
            sentences.append(
                "Your \(names) is showing as unhealthy — worth a reboot. The \(noun) online."
            )
        } else {
            sentences.append("All of your equipment is online and healthy.")
        }
        sentences.append(
            "You have \(devices) devices connected right now, peaking at \(peak) over the last 30 days."
        )
        return sentences.joined(separator: " ")
    }

    /// Deterministic billing response populated from the profile's
    /// real pricing fields. Transparent about what this surface
    /// knows vs what lives in the full carrier statement -
    /// honest > fake-precise for an exec pitch.
    static func billingResponse(profile: CustomerProfile) -> String {
        let plan = profile.plan
        let price = String(format: "$%.2f", plan.monthlyPrice)
        let activeBoltOns = profile.usage.activeBoltOns.filter { !$0.isEmpty }

        var sentences: [String] = [
            "Your \(plan.name) plan is \(price)/mo before taxes and fees."
        ]
        switch activeBoltOns.count {
        case 0: break
        case 1:
            sentences.append("\(activeBoltOns[0]) is active as an add-on.")
        default:
            sentences.append("\(activeBoltOns.joined(separator: ", ")) are active as add-ons.")
        }
        sentences.append(
            "For your current statement and payment options, open your carrier app."
        )
        return sentences.joined(separator: " ")
    }

    /// ADR-023 Phase 2 — single side-effect call after every assistant
    /// turn. Centralises three concerns so each `run*` handler stays
    /// declarative about WHAT it did, not WHAT to record:
    ///
    ///  1. Frustration counter updates (live-agent / didn't-work).
    ///  2. `pendingClarification` set/clear based on lane + missing
    ///     slots.
    ///  3. `pendingToolConfirmation` set/clear based on toolDecision.
    ///
    /// Skipping this call breaks the multi-turn state machine — every
    /// `run*` handler that appends an assistant message MUST call it
    /// before returning.
    private func recordTurnSideEffects(
        query: String,
        lane: UnderstandingLane,
        toolDecision: ToolDecision?,
        pendingToolConfirmation: ToolDecision? = nil,
        pendingIntent: ToolIntent?,
        missingSlots: Set<Slot>,
        assistantText: String? = nil,
        citedPageID: String? = nil,
        citedLinkID: String? = nil
    ) {
        conversationState.recordTurn(
            userMessage: query,
            assistantLane: lane,
            toolDecision: toolDecision,
            pendingToolConfirmation: pendingToolConfirmation,
            missingSlots: missingSlots,
            pendingIntent: pendingIntent,
            originalQuery: query
        )

        // ADR-024 §4.5 — record the resolved prior-turn context so the
        // NEXT turn's relational router can read it. Without this call,
        // `priorLane` + `priorIntent` stay nil and the STANCE_REVERT /
        // NEGATIVE_CONTINUATION paths in `decideMultiTurn` never engage
        // (silent dead code). Caught by the 2026-05-27 code review.
        //
        // Intent provenance: prefer the just-attached toolDecision's
        // intent (the .toolAction lane path); fall back to the
        // pendingIntent the workflow surfaced (clarification path);
        // nil when neither — keeps the next-turn router strictly
        // single-turn for non-action lanes.
        conversationState.recordPriorTurnContext(
            lane: lane,
            intent: toolDecision?.intent ?? pendingToolConfirmation?.intent ?? pendingIntent
        )

        // ADR-024 §4.5 — hidden-state cache for the NEXT turn's
        // relational pass. Phase 8a ships this call-site so the
        // contract is wired; both args are nil today because the
        // relational adapter isn't bundled yet (the SharedBackboneStrategy
        // doesn't extract per-turn hiddens). Phase 8d will pass through
        // `understanding.userHidden` / `assistantHidden` once the
        // strategy populates them. Today's no-op IS the architectural
        // substrate — the call is the contract.
        conversationState.cacheTurnHiddenStates(
            user: nil,
            assistant: nil
        )

        // ADR-024 follow-up 2026-05-27 — cache the literal assistant
        // text for next-turn `RetrievalContext` augmentation. Distinct
        // from the hidden-state cache above: TEXT is available
        // immediately and feeds the retriever; HIDDENS require a
        // forward pass we don't perform yet. Decoupling them means
        // retrieval augmentation works today without waiting for
        // Phase 8d.
        conversationState.cacheTurnText(assistant: assistantText)

        // Step 5b Pre-flight Fix C iOS-integration follow-up — cache
        // the cited RAG unit's `pageID` / `linkID` so the NEXT turn's
        // `RetrievalContext` can carry them into the dispatcher's
        // short-followup override. Both args are nil today on every
        // path EXCEPT the Verizon composer path, which passes
        // `finalResult.citedRAGUnit?.pageID` / `.linkID`. Nil ON THIS
        // turn is the explicit "no prior page" signal — greeting, OOS
        // refusal, live-agent, clarify, and ambiguous-yes-ignored
        // turns all clear it so the next turn doesn't anchor to a
        // stale page.
        conversationState.recordPriorPage(pageID: citedPageID, linkID: citedLinkID)

        let blackboardPendingDecision = pendingToolConfirmation
            ?? ((toolDecision?.isCompoundAttachment == false) ? toolDecision : nil)
        if let decision = blackboardPendingDecision,
           decision.requiresConfirmation,
           dialogueBlackboard.pendingToolConfirmation?.toolID != decision.toolID {
            dialogueBlackboard = TelcoDialogueBlackboardReducer.setPendingTool(
                TelcoPendingTool(
                    toolID: decision.toolID,
                    intent: decision.intent,
                    pageID: citedPageID,
                    linkID: citedLinkID
                ),
                on: dialogueBlackboard
            )
        }

        if let assistantText,
           !Self.blackboardSummaryMatches(dialogueBlackboard.lastAssistantSummary, assistantText) {
            dialogueBlackboard = TelcoDialogueBlackboardReducer.recordResponse(
                assistantText,
                on: dialogueBlackboard
            )
        }
    }

    // MARK: - Tool confirmation (invoked from ToolDecisionCard)

    /// Called from the Confirm button on a `ToolDecisionCard`. Runs the
    /// tool via `ToolExecutor`, which calls `tool.execute(...)` and
    /// then runs a second LFM generation to summarize the structured
    /// result. Both latencies roll into the trace row on the
    /// confirmation message.
    func confirmTool(messageID: UUID) {
        guard let idx = messages.firstIndex(where: { $0.id == messageID }),
              let decision = messages[idx].toolDecision,
              let tool = toolRegistry.tool(id: decision.toolID) else {
            return
        }

        // Mark the proposal consumed so it stops being a tappable card.
        messages[idx].toolDecision = nil

        // ADR-023 Phase 2 — the tool fired; any pending pointer is no
        // longer valid. Clear so a future "yes" doesn't re-fire it.
        conversationState.clearPendingToolConfirmation()

        Task { @MainActor in
            isProcessing = true
            defer { isProcessing = false }
            await executeConfirmedTool(tool: tool, decision: decision, triggerQuery: nil)
        }
    }

    func declineTool(messageID: UUID) {
        guard let idx = messages.firstIndex(where: { $0.id == messageID }) else { return }
        messages[idx].toolDecision = nil
        // ADR-023 Phase 2 — user dismissed the proposal; clear pending
        // so a follow-up "yes" doesn't resurrect it.
        conversationState.clearPendingToolConfirmation()
        dialogueBlackboard = TelcoDialogueBlackboardReducer.recordToolCancelled(
            on: dialogueBlackboard
        )
    }

    /// Execute a confirmed tool decision. This is the single side-effect
    /// path for both confirmation surfaces:
    ///   - tapping Confirm on the customer sheet / engineering card
    ///   - typing a bare affirmative while `ConversationState` has a
    ///     pending tool confirmation
    ///
    /// Keeping these together prevents the UI from showing an actionable
    /// promise that is not backed by the same execution logic.
    private func executeConfirmedTool(
        tool: Tool,
        decision: ToolDecision,
        triggerQuery: String?
    ) async {
        do {
            let outcome = try await toolExecutor.execute(tool: tool, decision: decision)
            tokenLedger.recordOnDevice(
                inputTokens: outcome.inputTokens,
                outputTokens: outcome.outputTokens
            )
            let total = outcome.toolLatencyMS + outcome.summaryLatencyMS
            let message = ChatMessage(
                role: .assistant,
                text: outcome.assistantText,
                routing: RoutingSummary(
                    path: .toolCall,
                    toolIntent: tool.intent,
                    containsPII: false
                ),
                deepLinks: tool.deepLink.map { [$0] } ?? [],
                latencyMS: total,
                trace: CallTrace(
                    surface: .tool,
                    inferenceMS: total,
                    inputTokens: outcome.inputTokens,
                    outputTokens: outcome.outputTokens
                )
            )
            messages.append(message)
            sessionStats.recordLatency(total)
            sessionStats.recordToolExecution(
                toolID: tool.id,
                status: outcome.toolResult.status
            )
            dialogueBlackboard = TelcoDialogueBlackboardReducer.recordToolExecuted(
                on: dialogueBlackboard
            )

            if let triggerQuery {
                let supportContext = Self.supportPageContext(for: tool.intent)
                recordTurnSideEffects(
                    query: triggerQuery,
                    lane: .toolAction,
                    toolDecision: nil,
                    pendingIntent: nil,
                    missingSlots: [],
                    assistantText: message.text,
                    citedPageID: supportContext?.pageID,
                    citedLinkID: supportContext?.linkID
                )
            }
        } catch {
            // Tool execution failed — the original mode was .toolAction;
            // pass it through so the trace row is accurate.
            appendInferenceFailure(error: error, mode: .toolAction, containsPII: false)
        }
    }

    // MARK: - NBA

    func nba(for id: String) -> (any NextBestAction)? {
        nbaEngine.topActions.first { $0.id == id }
            ?? NextBestActionRegistry.default.all.first { $0.id == id }
    }

    func acceptNBA(_ id: String) {
        nbaEngine.record(outcome: NBAOutcome(actionID: id, verdict: .accepted))
    }

    func declineNBA(_ id: String) {
        nbaEngine.record(outcome: NBAOutcome(actionID: id, verdict: .declined))
    }

    // MARK: - Helpers

    static func formatArguments(_ arguments: ToolArguments) -> [ToolDecisionArgument] {
        arguments.values
            .sorted { $0.key < $1.key }
            .map { key, value in
                let label = key
                    .replacingOccurrences(of: "_", with: " ")
                    .capitalized
                return ToolDecisionArgument(label: label, value: value)
            }
    }

    /// Convert the composer control-plane's typed executable intent into
    /// the shared `ToolDecision` envelope. This is the bridge between
    /// RAG answer rendering and the action state machine: the dispatcher
    /// decides "this selected unit maps to this registered tool"; the
    /// view model formats arguments and lets `ConversationState` persist
    /// the pending confirmation.
    private func composerExecutableToolDecision(
        from result: VerizonDispatchResult,
        extraction: ExtractionResult
    ) -> ToolDecision? {
        guard let intent = result.executableToolIntent,
              let tool = toolRegistry.tool(for: intent) else {
            return nil
        }
        let arguments = imperativeArguments(intent: intent, extraction: extraction)
        return ToolDecision(
            intent: intent,
            toolID: tool.id,
            displayName: tool.displayName,
            icon: tool.icon,
            description: tool.description,
            arguments: Self.formatArguments(arguments),
            confidence: result.requiresConfirmation == true ? 0.99 : 0.95,
            reasoning: tool.description,
            requiresConfirmation: tool.requiresConfirmation,
            isDestructive: tool.isDestructive,
            isCompoundAttachment: false
        )
    }

    /// ADR-022 compound-response review (2026-05-26) — when the primary
    /// lane produces a how-to or fallback response AND the user's query
    /// is an unambiguous imperative ("pause my son's tablet"), attach
    /// a SECONDARY ToolDecision card to the same message. RAG remains
    /// the primary content; the tool card is a one-tap shortcut beneath
    /// it. The user gets transparency (the article) + action (the card)
    /// in the same turn — neither is hidden behind the other.
    ///
    /// **Policy matrix** (see ADR-022 compound-response review):
    ///   .verizon(.ragStepByStep)        → ATTACH (the regression case)
    ///   .verizon(.unknownFeature)       → ATTACH (KB miss but user named a tool)
    ///   .verizon(.clarification)        → ATTACH (retrieval ambiguous but verb clear)
    ///   .verizon(.oosRefusal)           → SUPPRESS (refusal + tool = contradictory)
    ///   .verizon(.liveAgentEscalation)  → SUPPRESS (don't muddy escape hatch)
    ///   .verizon(.navOnlyDeeplink)      → SUPPRESS (intentional app handoff)
    ///   .verizon(.greeting)             → SUPPRESS (detector won't fire anyway)
    ///   .toolAction                     → SUPPRESS (already a tool flow)
    ///   .personalSummary                → SUPPRESS (off-topic)
    ///
    /// Question forms ("how do I…") are rejected by ImperativeToolDetector
    /// itself, so they never reach this attach logic.
    private func maybeAttachCompoundTool(
        to message: inout ChatMessage,
        query: String,
        extraction: ExtractionResult,
        lane: UnderstandingLane
    ) {
        // Already a tool flow — don't double-attach.
        if message.toolDecision != nil { return }

        // Policy gate — only certain lanes get the compound affordance.
        guard Self.laneAllowsCompoundTool(lane) else { return }

        // Deterministic μs lookup. Returns nil for question forms,
        // ambiguous phrasings, and any non-imperative surface.
        guard let intent = ImperativeToolDetector.detect(query) else { return }
        guard let tool = toolRegistry.tool(for: intent) else { return }

        let arguments = imperativeArguments(intent: intent, extraction: extraction)
        let decision = ToolDecision(
            intent: intent,
            toolID: tool.id,
            displayName: tool.displayName,
            icon: tool.icon,
            description: tool.description,
            arguments: Self.formatArguments(arguments),
            confidence: 0.95,
            reasoning: "Deterministic imperative pattern — compound affordance.",
            requiresConfirmation: tool.requiresConfirmation,
            isDestructive: tool.isDestructive,
            isCompoundAttachment: true
        )
        message.toolDecision = decision
        AppLog.intelligence.info(
            "compound tool attached lane=\(lane.wireName, privacy: .public) intent=\(intent.rawValue, privacy: .public)"
        )
    }

    /// ADR-024 — apply the side-effects the multi-turn fusion router
    /// emitted. Returns `true` when one of the actions handled dispatch
    /// (fired a pending tool); the caller then skips its own lane
    /// handler. Returns `false` when no short-circuit fired; the caller
    /// proceeds with normal lane dispatch.
    ///
    /// **Order matters**: short-circuiting actions (firePendingTool)
    /// MUST fire first so the user sees the action before the rest of
    /// the lane handler runs. Non-short-circuiting actions
    /// (counter mutations, slot accumulation) run in the order
    /// emitted by the router — each is idempotent.
    private func applyPostDecisionActions(
        _ actions: [PostDecisionAction],
        query: String,
        extraction: ExtractionResult,
        containsPII: Bool,
        understanding: QueryUnderstanding
    ) async -> PostActionResult {
        // Accumulator for the retrieval-augmentation signal — populated
        // when `.augmentRetrievalWithPriorAssistant` fires AND
        // ConversationState has a prior assistant message. Returned
        // to processTextQuery so the lane handler's ColBERT call
        // actually receives the augmented context (ADR-024 follow-up
        // 2026-05-27 — the missing primitive that turned this from a
        // log statement into a real cross-turn signal).
        //
        // Step 5b Pre-flight Fix C iOS-integration follow-up — seed
        // `priorPageID` / `priorLinkID` on EVERY turn (independent of
        // router actions) when the prior turn cited a RAG unit. The
        // dispatcher's short-followup override only USES these when
        // `isShortFollowup(query)` is true, so seeding them
        // unconditionally is safe; gating them behind a router action
        // would silently drop short-followup reuse on most turns and
        // break the Step 5b acceptance gates.
        var retrievalContext: RetrievalContext = RetrievalContext(
            priorAssistantText: nil,
            priorPageID: conversationState.priorPageID,
            priorLinkID: conversationState.priorLinkID
        )

        for action in actions {
            switch action {
            case .firePendingTool(let tool):
                // The user said "yes/ok/go ahead" on a pending tool.
                // Equivalent to tapping the Confirm button on the
                // previously-rendered card. Execute the pending decision
                // directly; re-proposing here would acknowledge the
                // confirmation without performing the requested action.
                //
                // **Short-circuit (CRITICAL fix 2026-05-27)**: this
                // action terminates the dispatch — we return `true`
                // immediately so subsequent actions in the array
                // don't run (no double-clear, no accidental
                // side-effects from a future action emitted after a
                // fire). Pending-state cleanup is done explicitly
                // here BEFORE the runToolProposal call.
                AppLog.intelligence.info(
                    "post-decision: firePendingTool \(tool.toolID, privacy: .public)"
                )
                conversationState.clearPendingToolConfirmation()
                conversationState.clearPendingClarification()
                guard let executable = toolRegistry.tool(id: tool.toolID) else {
                    appendInferenceFailure(
                        error: ToolError.notFound(tool.toolID),
                        mode: .toolAction,
                        containsPII: containsPII
                    )
                    return .shortCircuit
                }
                await executeConfirmedTool(
                    tool: executable,
                    decision: tool,
                    triggerQuery: query
                )
                return .shortCircuit

            case .clearPendingToolConfirmation:
                conversationState.clearPendingToolConfirmation()

            case .clearPendingClarification:
                conversationState.clearPendingClarification()

            case .accumulateSlotsFromAlignment(let intent, let slots):
                // The slot_alignment head says the user's reply fills
                // one or more missing slots. Lift the value(s) — best
                // effort using the existing ExtractionResult, fall
                // back to the trimmed reply itself. Slot-key mapping
                // lives on ToolIntent so a new intent forces compiler
                // exhaustiveness review.
                for slot in slots {
                    let value = Self.valueForSlot(
                        slot,
                        from: extraction,
                        rawReply: query
                    )
                    if let value {
                        conversationState.accumulateSlot(
                            intent: intent,
                            key: intent.argumentKey(for: slot),
                            value: value
                        )
                    }
                }

            case .clearSlotStore(let intent):
                conversationState.clearSlotStore(for: intent)

            case .traceNegativeContinuation:
                // Renamed from `incrementDidntWorkCounter` (2026-05-27)
                // to make the no-op semantics explicit. The actual
                // counter increment happens via
                // ConversationStateRecorder.isDidntWorkContinuation
                // inside recordTurn — same path as before ADR-024.
                // This case exists ONLY for the trace + reasoning
                // string so engineering can see the router's
                // classification. Do not add a mutation here without
                // also removing the regex match in recordTurn.
                break

            case .traceAffirmativeContinuation:
                // Renamed from `decrementFrustrationCounters` (2026-05-27).
                // Counters are append-only; positive signal currently
                // has no decrement path. If we add a per-session
                // "affirmative_count" field in the future, surface it
                // through ConversationState mutators — not here.
                break

            case .augmentRetrievalWithPriorAssistant:
                // ADR-024 follow-up 2026-05-27 — was a log statement.
                // Now actually plumbs the prior assistant text into the
                // lane handler's ColBERT retrieval call via the typed
                // `RetrievalContext` returned to processTextQuery.
                //
                // Source: `ConversationState.priorAssistantText`,
                // populated by `recordTurnSideEffects` at the end of
                // each turn. Nil when the prior turn produced an empty
                // reply OR when this is the first turn — the action
                // becomes a clean no-op in those cases (no spurious
                // augmentation with an empty string).
                //
                // Step 5b Pre-flight Fix C iOS wiring: thread the prior
                // assistant text plus its cited `pageID` / `linkID` as a
                // single coherent turn. Page/link hints are never sent
                // without the visible assistant turn they describe.
                let priorText = conversationState.priorAssistantText
                let priorPageID = conversationState.priorPageID
                let priorLinkID = conversationState.priorLinkID
                let hasText = priorText.map { !$0.isEmpty } ?? false
                if hasText {
                    retrievalContext = RetrievalContext(
                        priorAssistantText: hasText ? priorText : nil,
                        priorPageID: priorPageID,
                        priorLinkID: priorLinkID
                    )
                    AppLog.intelligence.info(
                        "post-decision: retrieval augmented (text=\(priorText?.count ?? 0, privacy: .public)c pageID=\(priorPageID ?? "<none>", privacy: .public) linkID=\(priorLinkID ?? "<none>", privacy: .public))"
                    )
                } else {
                    AppLog.intelligence.info(
                        "post-decision: retrieval augment requested but no prior assistant text — no-op"
                    )
                }

            case .suppressIntentRepeat:
                // The negative-continuation path. The repeat
                // suppression is enforced by re-using priorLane
                // (already done in the router) — this action is the
                // trace marker.
                break
            }
        }

        return PostActionResult(
            shortCircuited: false,
            retrievalContext: retrievalContext
        )
    }

    /// Best-effort value extraction for a missing slot, biased toward
    /// the existing structured extraction and falling back to the raw
    /// reply for bare-noun answers like "kitchen tablet".
    private static func valueForSlot(
        _ slot: Slot,
        from extraction: ExtractionResult,
        rawReply: String
    ) -> String? {
        switch slot {
        case .device:
            return extraction.targetDevice ?? bareNounAsSlotValue(rawReply)
        case .location:
            return extraction.locationHint ?? bareNounAsSlotValue(rawReply)
        case .time:
            return extraction.requestedTime
        case .accountRef:
            return nil  // no tool currently fires on account_ref alone
        }
    }

    /// ADR-023 Phase 2 — output of a successful clarification recovery.
    /// Carries the intent we're re-firing + the merged argument set
    /// (the previously-extracted args from the original imperative
    /// PLUS the slot we just lifted from the answer).
    struct ClarificationRecovery {
        let intent: ToolIntent
        let arguments: ToolArguments
        /// Which slot key in `arguments` was filled from the answer.
        /// `nil` when the recovery was a no-slot-needed re-fire (e.g.,
        /// disambiguating between two intents with no slot at all).
        let recoveredSlotKey: String?
    }

    /// ADR-023 Phase 2 — pre-classifier recovery of a pending
    /// clarification. Given the user's reply + the pending question
    /// context, attempt to fill the missing slot and return a
    /// `ClarificationRecovery` ready for `runToolProposal`. Returns
    /// nil when the reply doesn't match the pending intent's slots
    /// (the user changed topics).
    ///
    /// **Recovery rules** (pure-function over the inputs):
    ///  1. If the user typed a bare "yes" / "ok" and the pending
    ///     clarification didn't have a missing slot (e.g. a
    ///     `.ragClarification` source with no concrete intent), fall
    ///     through — no slot to fill.
    ///  2. If the user's reply, run through `RegexQueryExtractor`,
    ///     lifts a value into one of the missing slot kinds (.device
    ///     → target_device, .location → extender_name), merge it into
    ///     the original extraction and emit a recovery.
    ///  3. Otherwise nil — the caller falls through to normal
    ///     classification.
    func tryFulfillPendingClarification(
        userReply: String,
        extraction: ExtractionResult,
        pending: PendingClarification
    ) -> ClarificationRecovery? {
        // Pending clarification must have an intent for a tool re-fire.
        // `.ragClarification` with no intent isn't recoverable through
        // this path — the user must restate or pick.
        guard let intent = pending.intent else { return nil }
        guard !pending.missingSlots.isEmpty else { return nil }

        // Run the extractor over the user's REPLY in isolation so we
        // don't double-extract from the original query. Each missing
        // slot maps to one extraction field.
        let replyExtraction = queryExtractor.extract(from: userReply)
        var args = imperativeArguments(intent: intent, extraction: extraction).values
        var recoveredSlotKey: String?

        for slot in pending.missingSlots {
            switch slot {
            case .device:
                // Parental controls: target_device. The reply might
                // be a bare noun ("kitchen tablet") that the
                // RegexQueryExtractor's targetDevice patterns won't
                // catch (they expect "pause X for Y" framing). Fall
                // back to using the trimmed reply as the value when
                // the extractor's patterns don't match.
                let target = replyExtraction.targetDevice
                    ?? Self.bareNounAsSlotValue(userReply)
                if let target {
                    args["target_device"] = target
                    recoveredSlotKey = "target_device"
                }
            case .location:
                // Extender reboot: extender_name. Same fallback —
                // "upstairs" answered to "which extender?" is a bare
                // location.
                let location = replyExtraction.locationHint
                    ?? Self.bareNounAsSlotValue(userReply)
                if let location {
                    args["extender_name"] = location
                    recoveredSlotKey = "extender_name"
                }
            case .time:
                // No tool currently requires .time as a missing-slot
                // clarification (set-downtime is a future-scope tool).
                // Leave the slot empty — the tool's sheet UI handles
                // it interactively.
                continue
            case .accountRef:
                // Account references don't drive tool firing — the
                // affected tools all operate on the customer's
                // primary account implicitly.
                continue
            }
        }

        // We must have actually filled SOMETHING for this to be a
        // recovery. If the reply was a non-noun ("uhh", "?", "what?")
        // we get nothing and bail.
        guard recoveredSlotKey != nil else { return nil }

        return ClarificationRecovery(
            intent: intent,
            arguments: ToolArguments(args),
            recoveredSlotKey: recoveredSlotKey
        )
    }

    /// Treat a short bare-noun reply as a slot value. Conservative:
    /// rejects question forms ("which one?"), greetings, single-word
    /// affirmatives, and very long replies (likely a topic change).
    /// Tuned for the canonical clarification answers from the Verizon
    /// corpus: "kitchen tablet", "Sub account", "WiFi extends".
    private static func bareNounAsSlotValue(_ reply: String) -> String? {
        let trimmed = reply.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        // Reject affirmatives ("yes"/"ok") — those are the
        // pendingToolConfirmation path, not slot fill.
        if ConversationStateRecorder.isBareAffirmative(trimmed) { return nil }
        // Reject question forms.
        if trimmed.hasSuffix("?") { return nil }
        if trimmed.lowercased().hasPrefix("how ") { return nil }
        if trimmed.lowercased().hasPrefix("what ") { return nil }
        if trimmed.lowercased().hasPrefix("where ") { return nil }
        if trimmed.lowercased().hasPrefix("why ") { return nil }
        // Reject likely-topic-change long replies.
        let wordCount = trimmed.split(whereSeparator: { $0.isWhitespace }).count
        guard wordCount >= 1 && wordCount <= 6 else { return nil }
        // Strip a leading "the"/"my" so "the kitchen tablet" → "kitchen tablet".
        let stripped = trimmed.replacingOccurrences(
            of: #"^(?i)(?:the|my)\s+"#,
            with: "",
            options: .regularExpression
        )
        return stripped.isEmpty ? nil : stripped
    }

    /// Pure-function policy gate from the compound-response review.
    /// `nonisolated` so tests can call it off the main actor without
    /// hops — the policy is data-free, no actor-bound state to guard.
    nonisolated static func laneAllowsCompoundTool(_ lane: UnderstandingLane) -> Bool {
        switch lane {
        case .verizon(.ragStepByStep),
             .verizon(.unknownFeature),
             .verizon(.clarification):
            return true
        case .verizon(.oosRefusal),
             .verizon(.liveAgentEscalation),
             .verizon(.navOnlyDeeplink),
             .verizon(.greeting),
             .toolAction,
             .personalSummary:
            return false
        }
    }

    private func attachNBAIfAvailable(
        to message: inout ChatMessage,
        query: String,
        understanding: QueryUnderstanding? = nil,
        lane: UnderstandingLane? = nil,
        toolIntent: ToolIntent? = nil
    ) {
        // Promotional NBAs (upsell / retention / plan-fit) stay on the
        // Plan tab — feedback from device testing was that they read
        // as ads when interleaved with support answers. We keep the
        // call site so the engineering trace can still show what would
        // have fired.
        //
        // ADR-022 §4.3 Layer 4 — understanding-aware NBAs are
        // different. They're contextual support affordances (escalate
        // when frustrated, clarify when slots missing) that DO belong
        // beneath the support answer. When `understanding` is present
        // and a v2 NBA matches, attach it.
        //
        // ADR-023 Phase 2 — `conversationState` is passed so the
        // EscalateOnFrustrationNBA can fire on accumulated friction
        // counters even when the trained `emotional_state` head is
        // silent or nil (composite / unavailable strategies).
        guard let understanding, let lane else {
            _ = query
            return
        }
        if let nba = nbaEngine.bestMatchForUnderstanding(
            understanding,
            lane: lane,
            toolIntent: toolIntent,
            conversation: conversationState.snapshot
        ) {
            message.attachedNBAID = nba.id
        }
    }

    /// Produces a synthetic "no KB entry" stub so the LFM grounded-QA
    /// prompt always has an entry to reference. The prompt itself
    /// instructs the model to say it doesn't have information when
    /// the reference doesn't fit — cleaner than a separate code path.
    private func fallbackEntryForEmptyKB() -> KBEntry {
        KBEntry(
            id: "no-kb-match",
            topic: "No knowledge base match",
            aliases: [],
            category: "meta",
            answer: "No matching reference article was found in the on-device knowledge base.",
            deepLinks: [],
            tags: [],
            requiresToolExecution: false
        )
    }

    /// Stub entry shown while the LFM-embedding KB index is still
    /// building at first launch. The grounded-QA prompt sees this
    /// answer as the reference, so the assistant says "warming up,
    /// try again in a moment" instead of "no matching article" —
    /// closing a small but real UX gap on first-install.
    private func warmingUpFallbackEntry() -> KBEntry {
        KBEntry(
            id: "kb-warming-up",
            topic: "Knowledge base is warming up",
            aliases: [],
            category: "meta",
            answer: "I'm still loading the on-device knowledge base — this happens once when the app first launches. Try your question again in a few seconds.",
            deepLinks: [],
            tags: [],
            requiresToolExecution: false
        )
    }

    /// Surface an on-device inference failure as a short system-labeled
    /// chat bubble. No "Phase 2", no "falling back to cloud" — just the
    /// honest error. Structured context goes to the logger per the
    /// CLAUDE.local.md "Observability" principle.
    private func appendInferenceFailure(error: Error, mode: ChatMode?, containsPII: Bool) {
        let modeLabel = mode?.rawValue ?? "<none>"
        AppLog.chat.error("inference failure mode=\(modeLabel, privacy: .public): \(error.localizedDescription, privacy: .public)")
        messages.append(ChatMessage(
            role: .assistant,
            text: "On-device inference error: \(error.localizedDescription)",
            routing: RoutingSummary(
                path: mode?.routingPath ?? .answerWithRAG,
                toolIntent: nil,
                containsPII: containsPII
            )
        ))
    }
}
