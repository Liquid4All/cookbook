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
/// All on-device: no cloud escalator, no packet builder. When model files are
/// bundled, the app uses the LFM2.5-350M base plus telco LoRA adapters. When
/// model files are absent, the app enters a degraded cookbook mode so unit
/// tests and source exploration still work before users download GGUFs.
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

    // Intelligence layer. The telco decision engine is the preferred
    // ADR-014-shaped path: one multi-head understanding pass, then a
    // pure router. The individual protocols remain available for the
    // generative fallback path when classifier artifacts are absent.
    let decisionEngine: (any TelcoDecisionEngine)?
    let chatModeRouter: ChatModeRouter
    let kbExtractor: KBExtractor
    let toolSelector: ToolSelector

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

        // Intelligence + chat stack. The adapter-backed tool selector
        // and the base-GGUF chat mode router + KB extractor + chat
        // provider all share one LlamaBackend instance; the adapter
        // cache keeps swap cost sub-millisecond.
        let stack = Self.buildLFMStack(kb: kb)
        self.llamaBackend = stack.backend
        self.decisionEngine = stack.decisionEngine
        self.chatModeRouter = stack.chatModeRouter
        self.kbExtractor = stack.kbExtractor
        self.toolSelector = stack.tool
        self.modelProvider = stack.chat
        self.toolExecutor = ToolExecutor(chatProvider: stack.chat)

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

    /// Constructs the LFM stack. Full model mode requires the base GGUF and
    /// adapters copied by `bootstrap-models.sh`. If those artifacts are absent,
    /// return a degraded stack instead of crashing; this keeps cookbook tests
    /// runnable on a fresh clone where large GGUFs are intentionally omitted.
    ///
    /// The intent-router adapter is removed from the bundle (2026-04-20).
    /// ChatModeRouter replaced its gating role; ToolSelector handles
    /// tool_action queries directly without an intermediate intent step.
    private static func buildLFMStack(kb: KnowledgeBase) -> LFMStack {
        let backend = LlamaBackend()
        guard let basePath = TelcoModelBundle.basePath(),
              let toolAdapter = TelcoModelBundle.toolAdapterPath(),
              let chatModeRouterAdapter = TelcoModelBundle.chatModeRouterAdapterPath(),
              let kbExtractorAdapter = TelcoModelBundle.kbExtractorAdapterPath()
        else {
            AppLog.lfm.warning("Missing LFM GGUFs in bundle - running degraded cookbook stack. Run bootstrap-models.sh for real on-device inference.")
            let missingBackend = MissingModelBackend()
            return LFMStack(
                backend: backend,
                decisionEngine: nil,
                chatModeRouter: LFMChatModeRouter(backend: missingBackend, adapterPath: ""),
                kbExtractor: KeywordKBExtractor(),
                tool: LFMToolSelector(backend: missingBackend, adapterPath: ""),
                chat: LFMChatProvider(backend: missingBackend)
            )
        }

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

        // Try to load classifier heads. Prefer the shared telco adapter
        // (`telco-shared-clf-v1`) when present; otherwise use the
        // existing paired classification adapters in one decision-engine
        // wrapper. The paired path is slower, but still correct because
        // each head sees the hidden-state distribution it was trained on.
        var decisionEngine: (any TelcoDecisionEngine)?
        var classifier: TelcoMultiHeadClassifier?

        if TelcoModelBundle.classifierHeadsBundled(),
           TelcoModelBundle.classifierStackBundled() {
            do {
                let chatModePaths = TelcoModelBundle.classifierHeadPaths(task: "chat-mode")!
                let kbExtractPaths = TelcoModelBundle.classifierHeadPaths(task: "kb-extract")!
                let toolPaths = TelcoModelBundle.classifierHeadPaths(task: "tool-selector")!

                let chatModeHead = try ClassifierHead(
                    weightsURL: chatModePaths.weightsURL,
                    biasURL: chatModePaths.biasURL,
                    metaURL: chatModePaths.metaURL
                )
                let kbEntryHead = try ClassifierHead(
                    weightsURL: kbExtractPaths.weightsURL,
                    biasURL: kbExtractPaths.biasURL,
                    metaURL: kbExtractPaths.metaURL
                )
                let toolHead = try ClassifierHead(
                    weightsURL: toolPaths.weightsURL,
                    biasURL: toolPaths.biasURL,
                    metaURL: toolPaths.metaURL
                )

                // ADR-015 Phase-2 9-head inventory — only loaded when the
                // shared adapter + all 9 head bundles are present.
                let adr015Heads = Self.tryLoadADR015Heads()

                let loadedClassifier = TelcoMultiHeadClassifier(
                    backend: backend,
                    heads: TelcoMultiHeadClassifier.HeadInventory(
                        chatModeHead: chatModeHead,
                        kbEntryHead: kbEntryHead,
                        toolHead: toolHead,
                        adr015: adr015Heads
                    ),
                    adapters: TelcoMultiHeadClassifier.AdapterInventory(
                        sharedAdapterPath: TelcoModelBundle.sharedClfAdapterPath(),
                        chatModeAdapterPath: TelcoModelBundle.chatModeClfAdapterPath(),
                        kbExtractAdapterPath: TelcoModelBundle.kbExtractClfAdapterPath(),
                        toolSelectorAdapterPath: TelcoModelBundle.toolSelectorClfAdapterPath()
                    )
                )
                classifier = loadedClassifier
                decisionEngine = MultiHeadTelcoDecisionEngine(classifier: loadedClassifier)
                let adr015Status = adr015Heads != nil ? "9 ADR-015 heads loaded" : "ADR-015 heads absent"
                AppLog.lfm.info("Telco decision engine loaded: chat-mode (\(chatModeHead.numClasses)-way), kb-extract (\(kbEntryHead.numClasses)-way), tool-selector (\(toolHead.numClasses)-way), mode=\(loadedClassifier.inferenceMode.rawValue, privacy: .public), \(adr015Status, privacy: .public)")
            } catch {
                AppLog.lfm.error("Failed to load classifier stack, falling back to generative LoRA: \(error.localizedDescription, privacy: .public)")
            }
        }

        // Kick the model load off the main thread. LoRA adapters are
        // still loaded for the chat provider (base-model generative
        // responses) even when classifier heads handle classification.
        let classifierForWarmup = classifier
        Task.detached(priority: .userInitiated) {
            do {
                try await backend.loadModel(
                    path: basePath,
                    contextLength: 8192,
                    gpuLayers: gpuLayers,
                    temperature: 0
                )
                // Warm generative LoRA adapter caches. These remain
                // loaded for the chat provider and as generative fallback.
                try await backend.setAdapter(path: toolAdapter)
                try await backend.setAdapter(path: chatModeRouterAdapter)
                try await backend.setAdapter(path: kbExtractorAdapter)

                // Pre-warm classification LoRA adapters if the multi-head
                // decision engine is active.
                if classifierForWarmup != nil {
                    if let shared = TelcoModelBundle.sharedClfAdapterPath() {
                        try await backend.setAdapter(path: shared)
                    } else {
                        if let p = TelcoModelBundle.chatModeClfAdapterPath() {
                            try await backend.setAdapter(path: p)
                        }
                        if let p = TelcoModelBundle.kbExtractClfAdapterPath() {
                            try await backend.setAdapter(path: p)
                        }
                        if let p = TelcoModelBundle.toolSelectorClfAdapterPath() {
                            try await backend.setAdapter(path: p)
                        }
                    }
                }
            } catch {
                AppLog.lfm.error("base model load failed: \(error.localizedDescription, privacy: .public)")
            }
        }

        let bridge = LlamaAdapterBackend(backend: backend)
        let chat = LFMChatProvider(backend: bridge)

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

        return LFMStack(
            backend: backend,
            decisionEngine: decisionEngine,
            chatModeRouter: LFMChatModeRouter(backend: bridge, adapterPath: chatModeRouterAdapter),
            kbExtractor: kbExtractor,
            tool: LFMToolSelector(backend: bridge, adapterPath: toolAdapter),
            chat: chat
        )
    }

    /// Loads the 9 ADR-015 telco heads if the shared adapter + all head
    /// triplets are present. Returns nil if any artifact is missing or
    /// fails to load — the classifier still runs the Phase-1 triad.
    private static func tryLoadADR015Heads() -> TelcoMultiHeadClassifier.ADR015HeadInventory? {
        guard TelcoModelBundle.adr015TelcoStackBundled() else {
            AppLog.lfm.info("ADR-015 telco stack not bundled — using Phase-1 triad only")
            return nil
        }

        do {
            func head(_ task: String) throws -> ClassifierHead {
                guard let p = TelcoModelBundle.classifierHeadPaths(task: task) else {
                    throw NSError(
                        domain: "TelcoModelBundle",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "head missing: \(task)"]
                    )
                }
                return try ClassifierHead(
                    weightsURL: p.weightsURL,
                    biasURL: p.biasURL,
                    metaURL: p.metaURL
                )
            }

            return try TelcoMultiHeadClassifier.ADR015HeadInventory(
                supportIntent: head("telco-support-intent"),
                issueComplexity: head("telco-issue-complexity"),
                routingLane: head("telco-routing-lane"),
                cloudRequirements: head("telco-cloud-requirements"),
                requiredTool: head("telco-required-tool"),
                customerEscalationRisk: head("telco-customer-escalation-risk"),
                piiRisk: head("telco-pii-risk"),
                transcriptQuality: head("telco-transcript-quality"),
                slotCompleteness: head("telco-slot-completeness")
            )
        } catch {
            AppLog.lfm.error("ADR-015 head load failed: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    /// Bundle of every LFM primitive built by `buildLFMStack`. A named
    /// struct is clearer than a 5-tuple — field reordering on the call
    /// site would be silent breakage with tuples.
    private struct LFMStack {
        let backend: LlamaBackend
        let decisionEngine: (any TelcoDecisionEngine)?
        let chatModeRouter: ChatModeRouter
        let kbExtractor: KBExtractor
        let tool: ToolSelector
        let chat: LFMChatProvider
    }

    private struct MissingModelBackend: AdapterInferenceBackend {
        func generate(
            messages: [AdapterChatMessage],
            adapterPath: String,
            maxTokens: Int,
            stopSequences: [String]
        ) async throws -> String {
            throw MissingModelError()
        }

        func generate(
            prompt: String,
            adapterPath: String,
            maxTokens: Int,
            stopSequences: [String]
        ) async throws -> String {
            "Model artifacts are not bundled. Run bootstrap-models.sh, then regenerate the Xcode project to enable real on-device LFM inference."
        }
    }

    private struct MissingModelError: LocalizedError {
        var errorDescription: String? {
            "Model artifacts are not bundled. Run bootstrap-models.sh."
        }
    }
}
