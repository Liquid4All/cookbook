import SwiftUI
import Combine

@main
struct TelcoTriageApp: App {
    @StateObject private var appState = AppState()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(appState)
                .brand(appState.brands.selected)
                .appMode(appState.appMode)
                .onChange(of: scenePhase) { newPhase in
                    // Microphone / speech lives for the duration of a single
                    // utterance. If the app leaves foreground we tear the
                    // audio session down immediately — otherwise the route
                    // stays captured and other apps (Music, Podcasts, an
                    // active call) stay ducked.
                    if newPhase != .active, appState.voice.isListening {
                        Task { await appState.voice.stop() }
                    }
                }
        }
    }
}

/// One-stop dependency graph for the app. Constructed once at launch and
/// threaded through views via `@EnvironmentObject`.
///
/// All on-device: no cloud escalator and no free-form answer generator
/// deciding customer policy. Boot fails fast when the base LFM or required
/// tool adapter is missing; relation, shared understanding, RAG, and composer
/// status are surfaced through the runtime stack instead of silently swapping
/// in unrelated heuristics.
/// Controls what the UI surfaces to the viewer.
///
/// `.customer` — iMessage-clean chat: no trace rows, no confidence
///   scores, tool confirmations as sheets, 3 starter chips. This is
///   what a home internet customer would see.
///
/// `.engineering` — Full instrumentation: trace row on every response,
///   inline tool cards with extracted arguments + reasoning + confidence,
///   all 6 starters, latency/token counters in Settings.
///
/// The mode is persisted across launches via UserDefaults so a demoer
/// can pre-set it before handing the phone to a telco executive.
public enum AppMode: String, CaseIterable, Sendable {
    case customer
    case engineering
}

@MainActor
final class AppState: ObservableObject {
    // App mode — controls UI density
    @Published var appMode: AppMode {
        didSet { UserDefaults.standard.set(appMode.rawValue, forKey: "appMode") }
    }

    // Brand
    @Published var brands: BrandRegistry

    // Data / retrieval
    let knowledgeBase: KnowledgeBase

    // Model
    let modelProvider: LFMChatProvider
    let piiAnalyzer: PIIAnalyzer

    // Metrics
    let tokenLedger: TokenLedger
    let sessionStats: SessionStats

    // Customer + tools
    let customerContext: CustomerContext
    let toolRegistry: ToolRegistry

    // Specialist packs + capabilities
    let packManager: SpecialistPackManager
    let voice: VoiceCoordinator
    let visionAnalyzer: VisionAnalyzer

    // Personalization / ARPU
    let nbaEngine: NextBestActionEngine

    // Contextual support intelligence
    let supportSignalEngine: SupportSignalEngine

    // Intelligence layer. The normal Liquid Telco path is:
    // relation classifier -> shared understanding heads -> blackboard ->
    // BM25 RAG -> policy engine -> deterministic composer. The older
    // chat/kb routers remain injectable for explicit engineering probes.
    let chatModeRouter: ChatModeRouter
    let kbExtractor: KBExtractor
    let toolSelector: ToolSelector

    // Liquid Telco composer dispatcher. When the canonical RAG unit
    // corpus is bundled, this owns retrieval, deterministic route
    // policy, citation, and answer composition.
    let verizonDispatcher: VerizonChatDispatcher?

    /// Shared support-understanding classifier. One LFM2.5-350M adapter
    /// forward pass projects the 9 telco heads used by the normal
    /// customer/demo path.
    let telcoUnderstandingClassifier: TelcoSharedUnderstandingClassifying?

    /// Compatibility understanding vector used by older engineering probes.
    /// The shipped customer route uses `telcoUnderstandingClassifier` plus
    /// the dispatcher policy/composer stack above.
    let queryUnderstandingClassifier: QueryUnderstandingClassifying

    /// Compatibility relation strategy for the retired pairwise relation
    /// experiment. The active stateful turn classifier is
    /// `TelcoTurnRelationV4Strategy` inside the dispatcher path.
    let relationalStrategy: RelationalHeadsStrategy

    /// Boot-time load status of the composer RAG path. Surfaced by
    /// `RAGStatusChip` in engineering mode so the operator can see
    /// whether the corpus, retriever, and composer are live. Set once
    /// at init and never mutated thereafter — it's a deterministic boot
    /// result, not session state.
    let ragStatus: RAGStackStatus

    // Tool execution + LFM confirmation summary
    let toolExecutor: ToolExecutor

    /// The llama.cpp actor owning the base LFM2.5-350M. Lifetime spans
    /// the app session — the adapter cache is worthless if the backend
    /// reloads every call.
    let llamaBackend: LlamaBackend

    /// Forwards `voice.objectWillChange` to this object so SwiftUI
    /// views bound via `@EnvironmentObject appState` re-render when
    /// `voice.isListening` / `voice.state` change. Without this bridge,
    /// nested `ObservableObject`s don't bubble their changes up — the
    /// mic button appears to "do nothing" because the UI never refreshes.
    private var voiceCancellable: AnyCancellable?
    private var packCancellable: AnyCancellable?

    init() {
        // Restore persisted mode (default: customer for demo)
        let storedMode = UserDefaults.standard.string(forKey: "appMode") ?? "customer"
        self.appMode = AppMode(rawValue: storedMode) ?? .customer

        // Core data + PII scanner
        let kb = KnowledgeBase.loadFromBundle()
        let pii = PIIAnalyzer()

        self.knowledgeBase = kb
        self.piiAnalyzer = pii

        self.tokenLedger = TokenLedger()
        self.sessionStats = SessionStats()

        // Customer + tools
        let context = CustomerContext()
        self.customerContext = context
        self.toolRegistry = ToolRegistry.default(customerContext: context)

        // Specialist packs + capability coordinators
        let packs = SpecialistPackManager()
        self.packManager = packs
        let voiceCoordinator = VoiceCoordinator(packManager: packs)
        self.voice = voiceCoordinator
        self.visionAnalyzer = MockVisionAnalyzer(packManager: packs)

        // Wire the pre-uninstall hook: when the voice pack is removed,
        // tear down any live transcriber (unloads the LEAP ModelRunner
        // and releases mmap'd GGUFs) BEFORE the manager deletes the
        // cached bytes. Without this, deleting mid-session would hit a
        // mmap'd file — use-after-free.
        packs.setPreUninstallHook { [weak voiceCoordinator] pack in
            guard pack.capability == .voice else { return }
            await voiceCoordinator?.stop()
        }

        self.nbaEngine = NextBestActionEngine(
            registry: .default,
            customerContext: context
        )

        self.supportSignalEngine = SupportSignalEngine(context: context)

        self.brands = BrandRegistry()

        // Intelligence + chat stack. The composer dispatcher is the
        // normal support path; the shared LlamaBackend remains loaded
        // for explicit tool execution, profile summaries, and opt-in
        // legacy experiments.
        let stack = Self.buildLFMStack(kb: kb)
        self.llamaBackend = stack.backend
        self.chatModeRouter = stack.chatModeRouter
        self.kbExtractor = stack.kbExtractor
        self.toolSelector = stack.tool
        self.modelProvider = stack.chat
        self.toolExecutor = ToolExecutor(chatProvider: stack.chat)
        self.verizonDispatcher = stack.verizonDispatcher
        self.telcoUnderstandingClassifier = stack.telcoUnderstandingClassifier
        self.queryUnderstandingClassifier = stack.queryUnderstandingClassifier
        self.relationalStrategy = stack.relationalStrategy
        self.ragStatus = stack.ragStatus
        AppLog.lfm.info(
            "boot rag-status: \(stack.ragStatus.summary, privacy: .public)"
        )

        // Bridge nested ObservableObjects up to AppState. `ChatView` and
        // friends observe `appState` via `@EnvironmentObject`; without
        // this, `voice.isListening` / `packManager.states` changes never
        // trigger a redraw. Known SwiftUI pitfall — `@Published` on a
        // nested ObservableObject does not propagate to the parent's
        // observers on its own. Must run AFTER all stored properties
        // are initialized so `[weak self]` can capture `self`.
        self.voiceCancellable = voiceCoordinator.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
        self.packCancellable = packs.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
    }

    /// Constructs the LFM stack. Fails fast with a clear message if the
    /// base model GGUF or active tool-selector adapter GGUF is missing.
    /// Other optional adapters are loaded when present and reported through
    /// explicit runtime status rather than hidden fallback behavior.
    private static func buildLFMStack(kb: KnowledgeBase) -> LFMStack {
        guard let basePath = TelcoModelBundle.basePath(),
              let toolAdapter = TelcoModelBundle.toolAdapterPath()
        else {
            fatalError("Missing required LFM GGUFs in bundle. Expected \(TelcoModelBundle.baseModelName) and \(TelcoModelBundle.toolAdapterName) under Resources/Models/. See bootstrap-models.sh.")
        }
        let chatModeRouterAdapter = TelcoModelBundle.chatModeRouterAdapterPath()

        let backend = LlamaBackend()

        // iOS Simulator reports 0 MiB free on MTL0 — requesting GPU
        // offload there silently produces garbage (every sampled token
        // is <|pad|>, token id 0). Force CPU in the simulator; real
        // devices use full-stack GPU offload. BUG-022 regression guard.
        let gpuLayers: Int32
        #if targetEnvironment(simulator)
        gpuLayers = 0
        #else
        gpuLayers = 99
        #endif

        // Kick the model load off the main thread. The normal Liquid
        // Telco composer path is zero-generation, so boot must not
        // pre-warm the old composite understanding adapters. Loading
        // those LoRAs at startup makes the customer/demo path look like
        // it depends on chat-mode-router / Stage A even when it does not.
        // Keep only the tool adapter warm for explicit tool execution.
        Task.detached(priority: .userInitiated) {
            do {
                try await backend.loadModel(
                    path: basePath,
                    contextLength: 8192,
                    gpuLayers: gpuLayers,
                    temperature: 0
                )
                try await backend.setAdapter(path: toolAdapter)
            } catch {
                AppLog.lfm.error("base model load failed: \(error.localizedDescription, privacy: .public)")
            }
        }

        let bridge = LlamaAdapterBackend(backend: backend)
        let chat = LFMChatProvider(backend: bridge)
        let dialogueRepairVerbalizer = DialogueRepairVerbalizer.bundled(backend: bridge)
        if dialogueRepairVerbalizer != nil {
            AppLog.lfm.info("dialogue repair verbalizer ready (\(TelcoModelBundle.dialogueRepairV4AdapterName, privacy: .public))")
        } else {
            AppLog.lfm.warning("dialogue repair verbalizer unavailable: \(TelcoModelBundle.dialogueRepairV4AdapterName, privacy: .public) not bundled")
        }

        // Liquid Telco composer dispatcher. Built when the canonical RAG
        // corpus and lexical retriever load. The semantic control plane
        // is wired separately below because it classifies the turn before
        // retrieval/route policy.
        var verizonDispatcher: VerizonChatDispatcher?
        var telcoUnderstandingClassifier: TelcoSharedUnderstandingClassifying?
        var ragStatus: RAGStackStatus = .notInitialized
        // Step 6: composer-path dependencies. The composer is the only
        // normal answer path per the Step 5 decision record. Loaded
        // independently of Stage B so the dispatcher can take the
        // composer path even when (post-6.6) the Stage B GGUF is no
        // longer bundled. These are tiny (~150 KB JSON + pure Swift)
        // and never fail to load on a healthy build.
        let composerCorpus: RAGUnitCorpus?
        do {
            composerCorpus = try RAGUnitCorpus.loadFromBundle()
            AppLog.lfm.info("RAGUnitCorpus loaded with \(composerCorpus?.count ?? 0, privacy: .public) units")
        } catch {
            composerCorpus = nil
            AppLog.lfm.error("RAGUnitCorpus load failed: \(error.localizedDescription, privacy: .public)")
        }
        let composerRetriever: BM25HierarchyRetriever? = composerCorpus.map { BM25HierarchyRetriever(corpus: $0) }
        let answerComposer: AnswerComposing = DeterministicAnswerComposer()
        // The dispatcher only uses this registry as a LOOKUP TABLE —
        // it asks `tool(for: ToolIntent.xxx) != nil` to decide whether
        // a query should route to `.toolAction` (guardrail #3). It
        // never executes a tool — execution stays on the ChatViewModel
        // path through AppState.toolRegistry + ToolExecutor. So the
        // CustomerContext baked into this registry is irrelevant; a
        // fresh one is fine.
        let composerToolRegistry = ToolRegistry.default(customerContext: CustomerContext())
        // Step 5b Pre-flight Fix A: link_id → ToolIntent alias map.
        // Resolves the corpus/registry vocabulary gap (e.g.
        // link_id="speed-test" → tool "run-speed-test") AND the
        // imperative-only parental-controls carve-out. Mirrors the
        // Python harness so Swift dispatcher behaviour matches the
        // acceptance gates byte-for-byte.
        let composerToolAliasMap = ToolAliasMap.default()

        let composerWired = composerCorpus != nil && composerRetriever != nil
        if let cc = composerCorpus, composerWired {
            ragStatus = .live(chunkCount: cc.count, embedDim: 0)
            AppLog.lfm.info("Liquid Telco dispatcher ready (composer-only path, \(cc.count, privacy: .public) RAG units)")
            verizonDispatcher = VerizonChatDispatcher(
                stageA: nil,
                stageB: nil,
                kbFallback: KeywordKBExtractor(),
                kb: kb.entries,
                retriever: nil,
                modelHost: nil,
                composer: answerComposer,
                corpus: cc,
                lexicalRetriever: composerRetriever,
                toolRegistry: composerToolRegistry,
                toolAliasMap: composerToolAliasMap,
                dialogueRepairVerbalizer: dialogueRepairVerbalizer
            )
        } else {
            ragStatus = .degraded(reason: "RAGUnitCorpus failed to load")
            AppLog.lfm.error("Liquid Telco composer dispatcher unavailable — RAGUnitCorpus failed to load")
        }

        if TelcoModelBundle.adr015TelcoStackBundled() {
            do {
                telcoUnderstandingClassifier = try TelcoSharedUnderstandingClassifier.bundled(
                    backend: backend
                )
                AppLog.lfm.info("telco shared understanding ready (telco-shared-clf-v1 + 9 heads)")
            } catch {
                AppLog.lfm.error("telco shared understanding failed to load: \(error.localizedDescription, privacy: .public)")
            }
        } else {
            AppLog.lfm.warning(
                "telco shared understanding unavailable: expected \(TelcoModelBundle.sharedClfAdapterName, privacy: .public) and 9 ADR-015 heads"
            )
        }

        // KB selection: deterministic keyword/alias matching. The KB
        // has hand-curated aliases ("pause internet", "ssid", "block
        // websites"). Direct alias scoring beats every encoder
        // approach we tried on this curated, closed-domain data:
        //   - Classifier-adapter embeddings collapse fine-grained KB
        //     entries inside the same intent class (parental-controls,
        //     firmware-version, find-wifi-name all classify as
        //     `device_setup`, so their cosines clustered together).
        //   - Base-model mean-pool gave noisy 0.3–0.5 cosines even
        //     for clearly-related telco pairs.
        //   - Production RAG pattern is BM25/keyword first, embedding
        //     fallback for paraphrase. We mirror that here — keyword
        //     is the primary, with zero ML-component coupling.
        let kbExtractor = KeywordKBExtractor()

        let chatModeRouterForStack: ChatModeRouter
        let queryUnderstandingClassifier: QueryUnderstandingClassifying
        if verizonDispatcher != nil {
            chatModeRouterForStack = StubChatModeRouter(
                mode: .kbQuestion,
                confidence: 1.0,
                reasoning: "disabled on composer runtime path"
            )
            queryUnderstandingClassifier = QueryUnderstandingClassifier(strategy: UnavailableStrategy())
            AppLog.lfm.info("understanding layer disabled for composer runtime path")
        } else {
            // Legacy/degraded path only: pick the best available
            // understanding classifier when the composer dispatcher is
            // unavailable. This keeps the old stack as a fallback, but
            // prevents it from participating in the normal demo path.
            if let chatModeRouterAdapter {
                let liveChatModeRouter = LFMChatModeRouter(
                    backend: bridge,
                    adapterPath: chatModeRouterAdapter
                )
                chatModeRouterForStack = liveChatModeRouter
                let stageAForUnderstanding: VerizonStageAClassifying? =
                    (try? VerizonStageAClassifier.bundled(backend: backend))
                queryUnderstandingClassifier = QueryUnderstandingClassifier.bundled(
                    backend: backend,
                    chatModeRouter: liveChatModeRouter,
                    stageA: stageAForUnderstanding
                )
                AppLog.lfm.info(
                    "understanding layer ready for legacy fallback (target strategy = \(TelcoModelBundle.understandingV2Bundled() ? "shared(v2)" : "composite(v1)", privacy: .public))"
                )
            } else {
                chatModeRouterForStack = StubChatModeRouter(
                    mode: .outOfScope,
                    confidence: 0.0,
                    reasoning: "legacy chat-mode router adapter is not bundled"
                )
                queryUnderstandingClassifier = QueryUnderstandingClassifier(strategy: UnavailableStrategy())
                AppLog.lfm.warning(
                    "understanding layer unavailable for legacy fallback: \(TelcoModelBundle.chatModeRouterAdapterName, privacy: .public) not bundled"
                )
            }
        }

        let relationalStrategy: RelationalHeadsStrategy
        if let turnRelationV4 = try? TelcoTurnRelationV4Strategy.bundled(backend: backend) {
            relationalStrategy = turnRelationV4
            AppLog.lfm.info("relational strategy ready: telco-turn-relation-v4 classifier")
        } else {
            relationalStrategy = UnavailableRelationalStrategy()
            AppLog.lfm.warning(
                "relational strategy unavailable: expected \(TelcoModelBundle.turnRelationV4AdapterName, privacy: .public) and \(TelcoModelBundle.turnRelationV4HeadTask, privacy: .public) classifier triplet"
            )
        }

        return LFMStack(
            backend: backend,
            chatModeRouter: chatModeRouterForStack,
            kbExtractor: kbExtractor,
            tool: LFMToolSelector(backend: bridge, adapterPath: toolAdapter),
            chat: chat,
            verizonDispatcher: verizonDispatcher,
            telcoUnderstandingClassifier: telcoUnderstandingClassifier,
            queryUnderstandingClassifier: queryUnderstandingClassifier,
            ragStatus: ragStatus,
            relationalStrategy: relationalStrategy
        )
    }

    /// ADR-021 §11.4.3 + §11.5: instantiate the ColBERT retriever and
    /// the SwappingModelHost that owns the chat ↔ ColBERT backbone
    /// transitions. Also runs the boot-time corpus drift gate (§11.5 L4)
    /// — mismatch logs loudly via os_log so engineering mode surfaces
    /// the failure mode without dropping the user into a silent path.
    ///
    /// Returns (nil, nil) when any artifact is missing — the dispatcher
    /// then runs in degraded mode (Stage B without grounding). Logs
    /// every missing artifact loudly.
    ///
    /// **2026-05-27 — Short-circuited.** Per ADR-025 + the memory-diet
    /// retreat, ColBERT artifacts no longer ship in the bundle. This
    /// function returns the degraded sentinel immediately so boot
    /// doesn't waste time hunting for missing files. The dispatcher's
    /// normal answer path now uses BM25HierarchyRetriever plus the
    /// deterministic composer; ColBERT is retained only as a dormant
    /// revival path.
    ///
    /// To re-enable: ship the ColBERT GGUF + index files in the bundle
    /// again and delete this short-circuit. Or move to the ADR-025
    /// LoRA-adapter ColBERT path (preferred).
    private static func tryLoadColBERTStack(
        backend: LlamaBackend,
        basePath: String,
        gpuLayers: Int32
    ) -> (ColBERTRetriever?, SwappingModelHost?, RAGStackStatus) {
        // ADR-025 retreat — return immediately. The composer retriever is the
        // production path now; this legacy ColBERT retriever remains nil.
        let reason = "ColBERT removed from bundle (memory diet, 2026-05-27). " +
                     "BM25HierarchyRetriever + deterministic composer is the " +
                     "normal answer path. See ADR-025 for any future " +
                     "LoRA-adapter ColBERT revival."
        AppLog.lfm.info("Legacy ColBERT stack: \(reason, privacy: .public)")
        return (nil, nil, .degraded(reason: reason))

        // swiftlint:disable:next line_length
        // -- Original ColBERT-loading body retained below for revival, never executes --
        #if false
        // 1) Boot-time corpus drift gate — runs even if retrieval
        // can't be wired, so engineering mode surfaces the drift.
        do {
            let manifest = try CorpusManifest.bundled()
            if !manifest.matches {
                AppLog.lfm.warning(
                    "RAG corpus drift detected at boot. Stage B was trained against a different rag-chunks-v1.json. Retrieval still wires but Stage B's grounded-generation quality may degrade — faithfulness gates carry the safety burden."
                )
            }
        } catch {
            AppLog.lfm.info(
                "RAG corpus manifest not present (\(error.localizedDescription, privacy: .public)) — skipping drift gate"
            )
        }

        // 2) Load ColBERT primitives. All three artifacts must be
        // present or the retrieval path stays nil (dispatcher will
        // run ungrounded Stage B + log the missing-signal warning).
        // The status reason on failure carries the exact missing
        // artifact name so the operator's tap-to-diagnose chip can
        // point them at the right fix without needing Console.app.
        let index: ColBERTIndex
        let projection: ColBERTProjection
        do {
            index = try ColBERTIndex.bundled()
        } catch {
            let reason = "rag-index-v1.bin/rag-chunks-v1.json load failed: \(error.localizedDescription)"
            AppLog.lfm.warning(
                "\(reason, privacy: .public). Dispatcher will run Stage B without grounding (F1/F3/F5 active per ADR §11.1)."
            )
            return (nil, nil, .degraded(reason: reason))
        }
        do {
            projection = try ColBERTProjection.bundled()
        } catch {
            let reason = "colbert-projection-v1.bin load failed: \(error.localizedDescription)"
            AppLog.lfm.warning(
                "\(reason, privacy: .public). Dispatcher will run Stage B without grounding (F1/F3/F5 active per ADR §11.1)."
            )
            return (nil, nil, .degraded(reason: reason))
        }

        guard let colbertPath = bundle.path(
            forResource: "lfm2-colbert-350m-Q4_K_M",
            ofType: "gguf"
        ) else {
            let reason = "lfm2-colbert-350m-Q4_K_M.gguf missing from bundle — run bootstrap-models.sh"
            AppLog.lfm.warning("\(reason, privacy: .public)")
            return (nil, nil, .degraded(reason: reason))
        }

        let retriever = ColBERTRetriever(
            index: index,
            projection: projection,
            topK: 5
        )

        // 3) Build the model host. Chat backbone config matches the
        // boot loadModel call below (same path, same params); ColBERT
        // config picks 2048 ctx (queries are short, no need for 8K).
        let chatConfig = LlamaBackendModeConfig(
            path: basePath,
            contextLength: 8192,
            gpuLayers: gpuLayers
        )
        let colbertConfig = LlamaBackendModeConfig(
            path: colbertPath,
            contextLength: 2048,
            gpuLayers: gpuLayers
        )
        let host = SwappingModelHost(
            backend: backend,
            chatConfig: chatConfig,
            colbertConfig: colbertConfig
        )

        // The detached task below loadModels the chat backbone first.
        // Tell the host so its first .ensureMode(.chat) is a no-op
        // (otherwise it'd try to unload-then-reload).
        Task { await host.setInitialMode(.chat) }

        AppLog.lfm.info(
            "ColBERT stack loaded: index=\(index.count, privacy: .public) chunks, dim=\(index.embedDim, privacy: .public), modelHost ready"
        )
        return (
            retriever,
            host,
            .live(chunkCount: index.count, embedDim: index.embedDim)
        )
        #endif
    }

    /// Bundle accessor used by `tryLoadColBERTStack`. Always returns
    /// .main since this is the app's main bundle path lookup.
    private static var bundle: Bundle { .main }

    /// Bundle of every LFM primitive built by `buildLFMStack`. A named
    /// struct is clearer than a tuple — field reordering on the call
    /// site would be silent breakage with tuples.
    private struct LFMStack {
        let backend: LlamaBackend
        let chatModeRouter: ChatModeRouter
        let kbExtractor: KBExtractor
        let tool: ToolSelector
        let chat: LFMChatProvider
        let verizonDispatcher: VerizonChatDispatcher?
        let telcoUnderstandingClassifier: TelcoSharedUnderstandingClassifying?
        let queryUnderstandingClassifier: QueryUnderstandingClassifying
        let ragStatus: RAGStackStatus
        let relationalStrategy: RelationalHeadsStrategy
    }
}
