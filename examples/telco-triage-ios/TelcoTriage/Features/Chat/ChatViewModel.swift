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

    // Intelligence layer. `decisionEngine` is the preferred multi-head
    // path; the three individual protocols are fallback primitives when
    // classifier artifacts are absent.
    let decisionEngine: (any TelcoDecisionEngine)?
    let chatModeRouter: ChatModeRouter
    let kbExtractor: KBExtractor
    let queryExtractor: QueryExtractor
    let toolSelector: ToolSelector

    // Tool execution + result synthesis
    let toolExecutor: ToolExecutor
    private let useSimulatorFastGroundedQA: Bool

    /// Deterministic off-topic gate. See `TelcoTopicGate` for rationale —
    /// closed-domain vocabulary check, no LFM call, prevents
    /// hallucinated answers to genuinely off-topic queries.
    private let topicGate = TelcoTopicGate()

    /// Captured per-turn from `decisionEngine.decide(...)` so the
    /// engineering-mode pipeline trace card can render all 9 ADR-015
    /// head outputs + lane decision. `nil` until the first turn or when
    /// the multi-head stack isn't loaded. `internal` so the +Vision
    /// extension in a sibling file can clear it before a vision turn.
    var lastTelcoVector: TelcoDecisionVector?
    var lastTelcoLane: TelcoLaneDecision?

    /// Per-message pipeline-card expand state. Lifted out of the card's
    /// `@State` because `LazyVStack` recycles cells aggressively — local
    /// `@State` on a child of a `ForEach` is reset on scroll, which made
    /// collapsed cards re-expand themselves after the user collapsed
    /// them. Keying by `ChatMessage.id` (UUID) survives recycling.
    @Published var expandedTraceMessageIDs: Set<UUID> = []

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

    /// Build the current turn's pipeline trace, if the multi-head stack
    /// produced one. Each CallTrace construction site forwards this so
    /// the engineering-mode pipeline card can render all 9 head outputs
    /// + lane decision under the assistant bubble.
    func currentTelcoPipelineTrace() -> TelcoPipelineTrace? {
        guard let vector = lastTelcoVector else { return nil }
        return TelcoPipelineTrace.from(vector: vector, lane: lastTelcoLane)
    }

    /// Resolves the current brand's welcome greeting at call time. Passed
    /// as a closure so a mid-session brand flip (via Settings) propagates
    /// without ChatViewModel needing to observe BrandRegistry.
    private let welcomeGreetingProvider: @MainActor () -> String

    init(
        decisionEngine: (any TelcoDecisionEngine)? = nil,
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
        useSimulatorFastGroundedQA: Bool = ChatViewModel.shouldUseSimulatorFastGroundedQA,
        welcomeGreetingProvider: @escaping @MainActor () -> String
    ) {
        self.decisionEngine = decisionEngine
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
        lastTelcoVector = nil
        lastTelcoLane = nil
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
        // Stale trace from the previous turn would render with this
        // turn's bubble until the multi-head completes. Reset eagerly
        // so a mid-turn read of `currentTelcoPipelineTrace()` returns
        // nil rather than the previous query's vector.
        lastTelcoVector = nil
        lastTelcoLane = nil
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

        // 2) Deterministic topic gate. The 8-class support_intent
        //    head has no out_of_scope class — every query lands on
        //    one of the telco classes regardless. A vocabulary check
        //    over the KB's own terms is faster, perfectly accurate
        //    for this closed domain, and prevents the LFM from
        //    hallucinating answers to "what is the weather" that
        //    the classifier mistakenly tagged as `outage`.
        let gate = topicGate.decide(query)
        if case .outOfScope(let reason) = gate {
            AppLog.intelligence.info("topic gate: out_of_scope (\(reason, privacy: .public)) — refusing without LFM call")
            await runOutOfScope(
                query: query,
                modePrediction: ChatModePrediction(
                    mode: .outOfScope,
                    confidence: 1.0,
                    reasoning: "topic gate: \(reason)",
                    runtimeMS: 0
                ),
                extraction: extraction
            )
            return
        }

        if let decisionEngine {
            let result = await decisionEngine.decide(
                query: query,
                kb: kb.entries,
                extraction: extraction,
                availableTools: toolRegistry.all
            )
            lastTelcoVector = result.vector
            lastTelcoLane = result.lane

            // Deterministic post-classifier overrides. Each runs only
            // on unambiguous patterns that the LFM heads can't
            // represent in their fixed schemas:
            //   1. PersonalSummaryDetector — "summarize my X" / "show
            //      me my X" / "what devices are on my network" → the
            //      `personalSummary` lane that reads CustomerContext.
            //      Without this, the classifier picks `run_diagnostics`
            //      because that's the closest tool-vocab token.
            //   2. ImperativeToolDetector — "pause internet for kid",
            //      "restart wifi extender" → the iOS-only tools that
            //      the 6-class `required_tool` head can't represent.
            //
            // Order matters: PersonalSummary is checked first because
            // its triggers ("show me my", "summarize") are more
            // specific than imperative verbs alone.
            let summaryAware = applyPersonalSummaryOverride(
                decision: result.decision,
                query: query
            )
            let overridden = applyImperativeToolOverride(
                decision: summaryAware,
                query: query,
                extraction: extraction
            )

            await dispatch(
                overridden,
                query: query,
                extraction: extraction,
                containsPII: containsPII
            )
            return
        }

        // 2) Classify mode via LFM fallback. Pure routing decision — no
        //    lexical retrieval, no TF-IDF. The model picks one of
        //    four mutually-exclusive branches.
        let modePrediction = await chatModeRouter.classify(query: query)

        // 3) Dispatch on the chat mode. KB retrieval runs only on the
        //    question branch and is itself generative (LFM picks an
        //    entry_id from the 32-entry KB and emits a verbatim
        //    passage — no lexical primitive).
        switch modePrediction.mode {
        case .kbQuestion:
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
                retrievalMS: retrievalMS
            )

        case .toolAction:
            routingStage = .preparingAction
            await runToolProposal(
                query: query,
                modePrediction: modePrediction,
                extraction: extraction,
                containsPII: containsPII
            )

        case .personalSummary:
            routingStage = .composing
            await runPersonalizedSummary(
                query: query,
                modePrediction: modePrediction,
                extraction: extraction
            )

        case .outOfScope:
            routingStage = .composing
            await runOutOfScope(
                query: query,
                modePrediction: modePrediction,
                extraction: extraction
            )
        }
    }

    /// Deterministic override that converts a `kbQuestion` or
    /// `toolAction` decision into a `personalSummary` when the query
    /// is unambiguously asking the assistant to *read out* the user's
    /// own data ("summarize my home network", "show my connected
    /// devices"). The LFM heads don't have a "show my data" slot, so
    /// the classifier defaults to either `run_diagnostics` (because
    /// that's the closest tool-vocab token) or `local_answer`
    /// (KB lookup). Neither is what the user wants.
    private func applyPersonalSummaryOverride(
        decision: TelcoRoutingDecision,
        query: String
    ) -> TelcoRoutingDecision {
        guard PersonalSummaryDetector.detect(query) else { return decision }

        // Already a personalSummary? Nothing to do.
        if case .personalSummary = decision { return decision }

        let priorPrediction: ChatModePrediction
        switch decision {
        case .kbQuestion(let p, _),
             .toolAction(let p, _),
             .personalSummary(let p),
             .outOfScope(let p):
            priorPrediction = p
        }

        AppLog.intelligence.info("personal-summary override: classifier=\(priorPrediction.mode.rawValue, privacy: .public) → personalSummary for query")

        let promoted = ChatModePrediction(
            mode: .personalSummary,
            confidence: max(0.85, priorPrediction.confidence),
            reasoning: "personal-summary override: \(priorPrediction.reasoning) → personalSummary",
            runtimeMS: priorPrediction.runtimeMS
        )
        return .personalSummary(modePrediction: promoted)
    }

    /// Deterministic override that converts a `kbQuestion` decision
    /// into a `toolAction` when the query is an unambiguous imperative
    /// for an iOS-only tool the classifier head cannot represent.
    /// Returns the decision unchanged when no imperative pattern hits.
    private func applyImperativeToolOverride(
        decision: TelcoRoutingDecision,
        query: String,
        extraction: ExtractionResult
    ) -> TelcoRoutingDecision {
        guard case .kbQuestion(let modePrediction, _) = decision else {
            return decision
        }
        guard let intent = ImperativeToolDetector.detect(query),
              let _ = toolRegistry.tool(for: intent) else {
            return decision
        }
        AppLog.intelligence.info("imperative override: classifier=kbQuestion → tool=\(intent.rawValue, privacy: .public) for query")

        // Promote the prediction's mode to `.toolAction` so the trace
        // and downstream surfaces reflect the actual decision.
        let promotedPrediction = ChatModePrediction(
            mode: .toolAction,
            confidence: max(0.85, modePrediction.confidence),
            reasoning: "imperative override: \(modePrediction.reasoning) → toolAction(\(intent.rawValue))",
            runtimeMS: modePrediction.runtimeMS
        )
        let selection = ToolSelection(
            intent: intent,
            confidence: 0.95,
            arguments: imperativeOverrideArguments(intent: intent, extraction: extraction),
            reasoning: "deterministic imperative pattern (iOS-only tool, not in classifier schema)",
            runtimeMS: 0
        )
        return .toolAction(modePrediction: promotedPrediction, selection: selection)
    }

    private func imperativeOverrideArguments(
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
            return .empty
        }
    }

    // MARK: - Path handlers

    private func dispatch(
        _ decision: TelcoRoutingDecision,
        query: String,
        extraction: ExtractionResult,
        containsPII: Bool
    ) async {
        switch decision {
        case .kbQuestion(let modePrediction, let routerCitation):
            routingStage = .searching
            // Source-of-truth for KB citation is the LFM-embedding RAG
            // extractor wired in `AppState.buildLFMStack`. The router-
            // provided citation is `.noMatch` by design — see
            // `TelcoDecisionRouter.route` for the why. This call runs
            // mean-pool over LFM2.5-350M hidden states (no LoRA), then
            // cosine over the pre-built KB index.
            let retrievalStart = Date()
            let citation = await kbExtractor.extract(query: query, kb: kb.entries)
            let retrievalMS = Int(Date().timeIntervalSince(retrievalStart) * 1000)
            _ = routerCitation
            routingStage = .composing
            await runGroundedQA(
                query: query,
                citation: citation,
                modePrediction: modePrediction,
                extraction: extraction,
                containsPII: containsPII,
                retrievalMS: retrievalMS
            )

        case .toolAction(let modePrediction, let selection):
            routingStage = .preparingAction
            await runToolProposal(
                query: query,
                modePrediction: modePrediction,
                extraction: extraction,
                containsPII: containsPII,
                preselectedToolSelection: selection
            )

        case .personalSummary(let modePrediction):
            routingStage = .composing
            await runPersonalizedSummary(
                query: query,
                modePrediction: modePrediction,
                extraction: extraction
            )

        case .outOfScope(let modePrediction):
            routingStage = .composing
            await runOutOfScope(
                query: query,
                modePrediction: modePrediction,
                extraction: extraction
            )
        }
    }

    private func runGroundedQA(
        query: String,
        citation: KBCitation,
        modePrediction: ChatModePrediction,
        extraction: ExtractionResult,
        containsPII: Bool,
        retrievalMS: Int
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
                    telcoPipeline: currentTelcoPipelineTrace()
                )
            )
            attachNBAIfAvailable(to: &message, query: query)
            messages.append(message)
            sessionStats.recordLatency(visibleMS)
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
                    telcoPipeline: currentTelcoPipelineTrace()
                )
            )
            attachNBAIfAvailable(to: &message, query: query)
            messages.append(message)
            sessionStats.recordLatency(message.trace?.customerVisibleMS ?? response.latencyMS)
        } catch {
            appendInferenceFailure(error: error, mode: modePrediction.mode, containsPII: containsPII)
        }
    }

    private func runToolProposal(
        query: String,
        modePrediction: ChatModePrediction,
        extraction: ExtractionResult,
        containsPII: Bool,
        preselectedToolSelection: ToolSelection? = nil
    ) async {
        // Tool selection routes on the query alone. The generative-
        // retrieval architecture doesn't pre-fetch a KB entry for the
        // action branch, and the tool-selector prompt never used one.
        let toolSelection: ToolSelection
        if let preselectedToolSelection {
            toolSelection = preselectedToolSelection
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
                retrievalMS: retrievalMS
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
                telcoPipeline: currentTelcoPipelineTrace()
            )
        )
        attachNBAIfAvailable(to: &message, query: query)
        messages.append(message)
        sessionStats.recordLatency(onDeviceMS)
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
        extraction: ExtractionResult
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
                telcoPipeline: currentTelcoPipelineTrace()
            )
        )
        tokenLedger.recordDeflection()
        messages.append(message)
        sessionStats.recordLatency(modePrediction.runtimeMS)
    }

    private func runOutOfScope(
        query: String,
        modePrediction: ChatModePrediction,
        extraction: ExtractionResult
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
                telcoPipeline: currentTelcoPipelineTrace()
            )
        )
        messages.append(message)
        sessionStats.recordLatency(modePrediction.runtimeMS)
    }

    private static let outOfScopeRefusal =
        "That's outside what I handle - I only cover home internet support. " +
        "Your question stayed on this phone; nothing was sent to the cloud."

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

        Task { @MainActor in
            isProcessing = true
            defer { isProcessing = false }
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
                        outputTokens: outcome.outputTokens,
                        // Tool-execution receipt — no classifier ran for
                        // this message, so don't carry the proposal turn's
                        // 9-head trace into it.
                        telcoPipeline: nil
                    )
                )
                messages.append(message)
                sessionStats.recordLatency(total)
                sessionStats.recordToolExecution(
                    toolID: tool.id,
                    status: outcome.toolResult.status
                )
            } catch {
                // Tool execution failed — the original mode was
                // .toolAction; pass it through so the trace row is
                // accurate.
                appendInferenceFailure(error: error, mode: .toolAction, containsPII: false)
            }
        }
    }

    func declineTool(messageID: UUID) {
        guard let idx = messages.firstIndex(where: { $0.id == messageID }) else { return }
        messages[idx].toolDecision = nil
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

    private func attachNBAIfAvailable(to message: inout ChatMessage, query: String) {
        // Intentionally a no-op. NBAs read as promotional inserts when
        // they follow a support answer — feedback from both customer-
        // and engineering-mode device testing. The chat surface is for
        // support; the personalization story stays on the Plan tab's
        // "For You" section and the Savings tab's ARPU signal card,
        // where NBAs have their own framing and don't interrupt a
        // support flow. Keep the call site so it's obvious a future
        // design decision could re-enable, but don't attach anything.
        _ = (message, query)
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
