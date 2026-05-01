import Foundation

/// Concrete `AdapterInferenceBackend` implementation wrapping a live
/// shared `LlamaBackend`.
///
/// One shared `LlamaBackend` loads the LFM2.5-350M base GGUF at
/// launch; this wrapper routes each call to either an adapter-applied
/// generation (for the intent classifier and tool selector) or a
/// base-only generation (for the chat provider, which uses the base
/// model directly for grounded QA / tool summaries / personalized
/// responses).
///
/// Convention: passing `adapterPath == ""` means "run the base model
/// without a LoRA adapter". The bridge calls `removeAdapter()` before
/// generation to ensure any previously-applied adapter is detached.
public struct LlamaAdapterBackend: AdapterInferenceBackend {
    public let backend: LlamaBackend

    public init(backend: LlamaBackend) {
        self.backend = backend
    }

    public func generate(
        messages: [AdapterChatMessage],
        adapterPath: String,
        maxTokens: Int,
        stopSequences: [String]
    ) async throws -> String {
        if adapterPath.isEmpty {
            await backend.removeAdapter()
        } else {
            try await backend.setAdapter(path: adapterPath, scale: 1.0)
        }
        // Translate POC-facing chat messages to the LFMEngine type and
        // route through the chat-template path. This is the ONLY correct
        // entrypoint for LoRA adapters trained via leap-finetune, which
        // applies the model's chat template at training time.
        let engineMessages = messages.map { m in
            LlamaChatMessage(role: m.role.rawValue, content: m.content)
        }
        let (text, _, _) = try await backend.generate(
            messages: engineMessages,
            maxTokens: maxTokens,
            temperature: 0,
            stopSequences: stopSequences,
            clearCache: true,
            outputMode: .text
        )
        return text
    }

    public func generate(
        prompt: String,
        adapterPath: String,
        maxTokens: Int,
        stopSequences: [String]
    ) async throws -> String {
        if adapterPath.isEmpty {
            await backend.removeAdapter()
        } else {
            try await backend.setAdapter(path: adapterPath, scale: 1.0)
        }
        let (text, _, _) = try await backend.generate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: 0,
            stopSequences: stopSequences,
            clearCache: true,
            outputMode: .text
        )
        return text
    }
}

/// Resource paths for the GGUFs bundled under `Resources/Models/`.
/// Wrapped in a namespace so callers never typo the file name + extension.
///
/// Architecture (2026-04-20): ChatModeRouter is the first hop. It routes to
/// either KBExtractor (kb_question) or ToolSelector (tool_action). The old
/// IntentRouter is removed — ChatModeRouter replaced its gating role.
public enum TelcoModelBundle {
    // CRITICAL: Must be the BASE model, NOT DPO/instruct. LoRA adapters
    // are trained on LFM2.5-350M-Base — applying them to DPO weights causes
    // hallucinated outputs (67% → 84% accuracy fix, 2026-04-20 session).
    public static let baseModelName = "lfm25-350m-base-Q4_K_M"
    // v3: retrained on LFM2.5-350M-Base. All adapters in this example
    // should be paired with the Base model.
    public static let toolAdapterName = "telco-tool-selector-v3"
    // v2: retrained with augmented data targeting possessive-field lookups
    // ("what is my ipv4", "what firmware am I on") that v1 misrouted to
    // personal_summary.
    public static let chatModeRouterAdapterName = "chat-mode-router-v2"
    public static let kbExtractorAdapterName = "kb-extractor-v1"

    // Classification LoRA adapters — paired with classifier head binaries.
    // These adapters specialize the backbone's hidden states for each
    // classification task. The classifier heads were trained WITH these
    // adapters applied; without them, accuracy drops ~30-70pp.
    public static let chatModeClfAdapterName = "chat-mode-clf-v1"
    public static let kbExtractClfAdapterName = "kb-extract-clf-v1"
    public static let toolSelectorClfAdapterName = "tool-selector-clf-v1"
    public static let sharedClfAdapterName = "telco-shared-clf-v1"

    public static let ext = "gguf"

    public static func basePath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: baseModelName, ofType: ext)
    }

    public static func toolAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: toolAdapterName, ofType: ext)
    }

    public static func chatModeRouterAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: chatModeRouterAdapterName, ofType: ext)
    }

    public static func kbExtractorAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: kbExtractorAdapterName, ofType: ext)
    }

    // Classification adapter paths — paired with classifier head binaries

    public static func chatModeClfAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: chatModeClfAdapterName, ofType: ext)
    }

    public static func kbExtractClfAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: kbExtractClfAdapterName, ofType: ext)
    }

    public static func toolSelectorClfAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: toolSelectorClfAdapterName, ofType: ext)
    }

    public static func sharedClfAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: sharedClfAdapterName, ofType: ext)
    }

    /// True when every GGUF is bundled. AppState fails fast when this
    /// returns false — see `AppState.buildLFMStack`.
    public static func isFullyBundled(in bundle: Bundle = .main) -> Bool {
        return basePath(in: bundle) != nil
            && toolAdapterPath(in: bundle) != nil
            && chatModeRouterAdapterPath(in: bundle) != nil
            && kbExtractorAdapterPath(in: bundle) != nil
    }

    /// True when classifier heads AND their paired classification adapters
    /// are all bundled. Both are required — heads without adapters produce
    /// ~30-70pp accuracy drops (Phase 7 eval, 2026-04-24).
    public static func classifierStackBundled(in bundle: Bundle = .main) -> Bool {
        return sharedClassifierStackBundled(in: bundle)
            || pairedClassifierStackBundled(in: bundle)
    }

    /// True when the shared-adapter multi-head classifier can run one
    /// forward pass for every telco sequence head.
    public static func sharedClassifierStackBundled(in bundle: Bundle = .main) -> Bool {
        return classifierHeadsBundled(in: bundle)
            && sharedClfAdapterPath(in: bundle) != nil
    }

    /// True when existing per-head adapters can run a mathematically
    /// correct transitional multi-head path.
    public static func pairedClassifierStackBundled(in bundle: Bundle = .main) -> Bool {
        return classifierHeadsBundled(in: bundle)
            && chatModeClfAdapterPath(in: bundle) != nil
            && kbExtractClfAdapterPath(in: bundle) != nil
            && toolSelectorClfAdapterPath(in: bundle) != nil
    }

    // MARK: - Classifier Head Binaries

    // Three classifier heads replace generative LoRA calls for
    // classification tasks. One backbone forward pass + a linear
    // head (<1ms) vs autoregressive decoding (~200ms).
    //
    // Files are named {task}_classifier_{weights,bias,meta}.{bin,json}
    // to avoid filename collisions in the flat app bundle.

    /// Resolved URLs for a single classifier head's three artifacts.
    public struct ClassifierHeadPaths {
        public let weightsURL: URL
        public let biasURL: URL
        public let metaURL: URL
    }

    /// Returns paths for a classifier head's three artifacts, or nil
    /// if any artifact is missing from the bundle.
    public static func classifierHeadPaths(
        task: String,
        in bundle: Bundle = .main
    ) -> ClassifierHeadPaths? {
        guard let w = bundle.url(forResource: "\(task)_classifier_weights", withExtension: "bin"),
              let b = bundle.url(forResource: "\(task)_classifier_bias", withExtension: "bin"),
              let m = bundle.url(forResource: "\(task)_classifier_meta", withExtension: "json")
        else { return nil }
        return ClassifierHeadPaths(weightsURL: w, biasURL: b, metaURL: m)
    }

    /// True when all three classifier head artifact sets are bundled.
    public static func classifierHeadsBundled(in bundle: Bundle = .main) -> Bool {
        return classifierHeadPaths(task: "chat-mode", in: bundle) != nil
            && classifierHeadPaths(task: "kb-extract", in: bundle) != nil
            && classifierHeadPaths(task: "tool-selector", in: bundle) != nil
    }

    // MARK: - ADR-015 telco multi-head classifier (Phase 2)

    /// Names of the 9 telco sequence heads from ADR-015. Each one is a
    /// `{name}_classifier_{weights,bias,meta}.{bin,json}` triplet.
    public static let adr015TelcoHeadNames: [String] = [
        "telco-support-intent",
        "telco-issue-complexity",
        "telco-routing-lane",
        "telco-cloud-requirements",
        "telco-required-tool",
        "telco-customer-escalation-risk",
        "telco-pii-risk",
        "telco-transcript-quality",
        "telco-slot-completeness",
    ]

    /// True when all 9 telco head artifact sets AND the shared classification
    /// LoRA are bundled. Drives the ADR-015 lane router on the iOS app.
    public static func adr015TelcoStackBundled(in bundle: Bundle = .main) -> Bool {
        guard sharedClfAdapterPath(in: bundle) != nil else { return false }
        for head in adr015TelcoHeadNames {
            if classifierHeadPaths(task: head, in: bundle) == nil {
                return false
            }
        }
        return true
    }
}
