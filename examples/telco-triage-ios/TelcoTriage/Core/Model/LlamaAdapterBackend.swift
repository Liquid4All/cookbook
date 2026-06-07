import Foundation

/// Concrete `AdapterInferenceBackend` implementation wrapping a live
/// shared `LlamaBackend`.
///
/// One shared `LlamaBackend` loads the LFM2.5-350M base GGUF at launch.
/// This wrapper routes adapter-applied generation for the small number of
/// runtime paths that still need text output, such as tool support and bounded
/// dialogue repair. The normal support-answer path retrieves structured
/// evidence and uses the deterministic composer instead of free-form answer
/// generation.
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
/// Runtime architecture: relation classifier -> shared support-understanding
/// heads -> blackboard -> BM25 RAG -> policy engine -> deterministic composer.
/// Some older adapter names remain here for compatibility probes, but they are
/// not the cookbook bundle contract.
public enum TelcoModelBundle {
    // CRITICAL: Must be the BASE model, NOT DPO/instruct. LoRA adapters
    // are trained on LFM2.5-350M-Base — applying them to DPO weights causes
    // hallucinated outputs (67% → 84% accuracy fix, 2026-04-20 session).
    //
    // PRECISION: Q4_K_M, the trained-against quant tier used by the cookbook
    // bundle. Stateful multi-turn handling is supplied by the separate
    // `telco-turn-relation-v4` classifier rather than the retired pairwise
    // relation experiment.
    public static let baseModelName = "lfm25-350m-base-Q4_K_M"
    // Active helper adapter for ambiguous local action/tool paths.
    public static let toolAdapterName = "telco-tool-selector-v3"
    // Compatibility adapters retained for older engineering probes. The
    // customer answer path does not depend on them.
    public static let chatModeRouterAdapterName = "chat-mode-router-v2"
    public static let kbExtractorAdapterName = "kb-extractor-v1"

    // Legacy per-task classifier adapters. The active customer path uses the
    // shared telco classifier adapter plus the 9 support-understanding heads.
    public static let chatModeClfAdapterName = "chat-mode-clf-v1"
    public static let kbExtractClfAdapterName = "kb-extract-clf-v1"
    public static let toolSelectorClfAdapterName = "tool-selector-clf-v1"
    public static let sharedClfAdapterName = "telco-shared-clf-v1"

    // Retired answer-generator artifacts. They are not part of normal support
    // Q&A, where BM25 selects a structured RAG unit and Swift composes the
    // cited answer. Kept only so engineering builds can compare old probes.
    public static let verizonStageBMergedName = "vz-stage-b-v1.Q4_K_M"
    public static let verizonStageBLoraName = "vz-stage-b-v1.lora.f16"

    // Dialogue-repair verbalizer v4.
    //
    // This LoRA is not a router, retriever, tool selector, or citation
    // authority. It is a response-only verbalizer trained to make
    // multi-turn repair turns sound natural while echoing the route,
    // evidence ids, handoff, and confirmation policy supplied by Swift.
    public static let dialogueRepairV4AdapterName = "telco-dialogue-repair-v4"

    // Legacy two-head router artifacts. The active bundle uses
    // `telco-shared-clf-v1` and the 9 telco heads instead.
    public static let verizonTopicGateAdapterName = "vz-topic-gate-clf-v1.lora"
    public static let verizonRefusalFlagsAdapterName = "vz-refusal-flags-clf-v1.lora"
    public static let verizonTopicGateHeadTask = "vz-topic-gate"
    public static let verizonRefusalFlagsHeadTask = "vz-refusal-flags"

    // Legacy v2 understanding layer retained for compatibility probes.
    // The current customer path uses `sharedClfAdapterName` plus
    // `adr015TelcoHeadNames`.
    public static let understandingV2AdapterName = "telco-shared-clf-v2"
    public static let understandingV2ChatModeHeadTask = "vz-chat-mode-v2"
    public static let understandingV2TopicGateHeadTask = "vz-topic-gate-v2"
    public static let understandingV2RefusalFlagsHeadTask = "vz-refusal-flags-v2"
    public static let understandingV2EmotionalStateHeadTask = "vz-emotional-state-v2"
    public static let understandingV2SlotCompletenessHeadTask = "vz-slot-completeness-v2"

    // Retired pairwise relational heads. Superseded by
    // `telco-turn-relation-v4`; retained only for historical/degraded
    // comparison and not expected in the cookbook bundle.
    public static let relationalV1AdapterName = "telco-relational-v1"
    public static let relationalV1TurnRelHeadTask = "telco-relational-turn-rel"
    public static let relationalV1SlotAlignHeadTask = "telco-relational-slot-align"
    public static let relationalV1StanceChangeHeadTask = "telco-relational-stance-change"

    // Active 12-way telco turn-relation classifier. This is a classifier
    // stack, not the retired generative relation LoRA. The head is valid only
    // with `telco-turn-relation-v4.gguf`.
    public static let turnRelationV4AdapterName = "telco-turn-relation-v4"
    public static let turnRelationV4HeadTask = "telco-turn-relation"

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

    // Backwards-compatible alias for the merged Stage B GGUF.
    // VerizonRAGTestView still loads this via loadModel() to isolate the
    // generator from the chat backend; the merged GGUF stays bundled for
    // that probe view. Production chat now uses verizonStageBLoraPath via
    // adapter swapping on the shared base.
    public static let verizonStageBGeneratorName = verizonStageBMergedName

    /// Path to the Verizon Stage B MERGED GGUF (350M + step-format LoRA
    /// fused, Q4_K_M, ~219 MB). Self-contained — loaded via `loadModel()`
    /// directly. Kept for the engineering probe view (VerizonRAGTestView)
    /// where isolating Stage B from the chat backend lets us measure its
    /// latency without other state in the way. Returns nil when not
    /// bundled (engineering builds / CI clones without the artifact).
    public static func verizonStageBGeneratorPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: verizonStageBMergedName, ofType: ext)
    }

    /// Path to the Verizon Stage B LoRA-adapter GGUF (~23 MB, r=32).
    /// Applied to the shared base via `setAdapter(path:scale:)` on the
    /// `.ragStepByStep` lane. One foundation model, adapters change per
    /// lane — the single-backbone architecture the merged chat dispatcher
    /// runs on. Returns nil when not bundled.
    public static func verizonStageBLoraPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: verizonStageBLoraName, ofType: ext)
    }

    public static func dialogueRepairV4AdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: dialogueRepairV4AdapterName, ofType: ext)
    }

    // MARK: - Legacy Stage A (topic_gate + refusal_flags)

    /// Path to the Verizon topic-gate Stage A classification LoRA
    /// (~11 MB, r=16). Must be applied via `setAdapter` before the
    /// mean-pool forward pass that feeds the topic-gate classifier head
    /// — the head was trained with this adapter active.
    public static func verizonTopicGateAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: verizonTopicGateAdapterName, ofType: ext)
    }

    /// Path to the Verizon refusal-flags Stage A classification LoRA
    /// (~11 MB, r=16). Same contract as `verizonTopicGateAdapterPath`.
    public static func verizonRefusalFlagsAdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: verizonRefusalFlagsAdapterName, ofType: ext)
    }

    /// Resolved paths for the Verizon topic-gate head triplet
    /// (`vz-topic-gate_classifier_{weights,bias,meta}`).
    public static func verizonTopicGateHeadPaths(in bundle: Bundle = .main) -> ClassifierHeadPaths? {
        classifierHeadPaths(task: verizonTopicGateHeadTask, in: bundle)
    }

    /// Resolved paths for the Verizon refusal-flags head triplet
    /// (`vz-refusal-flags_classifier_{weights,bias,meta}`).
    public static func verizonRefusalFlagsHeadPaths(in bundle: Bundle = .main) -> ClassifierHeadPaths? {
        classifierHeadPaths(task: verizonRefusalFlagsHeadTask, in: bundle)
    }

    /// True when the legacy Stage A stack is fully bundled. This is retained
    /// for engineering probes and is not the current customer ship gate.
    public static func verizonStageABundled(in bundle: Bundle = .main) -> Bool {
        return verizonTopicGateAdapterPath(in: bundle) != nil
            && verizonRefusalFlagsAdapterPath(in: bundle) != nil
            && verizonTopicGateHeadPaths(in: bundle) != nil
            && verizonRefusalFlagsHeadPaths(in: bundle) != nil
    }

    /// Legacy router bundle check. The normal answer layer is deterministic
    /// composer over `rag-units-v1.json`.
    public static func verizonRagStackBundled(in bundle: Bundle = .main) -> Bool {
        return verizonStageABundled(in: bundle)
    }

    // MARK: - Legacy v2 understanding layer

    /// Path to the legacy shared LoRA adapter for the v2 understanding layer.
    public static func understandingV2AdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: understandingV2AdapterName, ofType: ext)
    }

    /// Bundled classifier head paths for v2, keyed by the head's task name.
    /// Defensive lookup — any missing head returns nil for that key and
    /// the classifier degrades that head to absent in `QueryUnderstanding`.
    public static func understandingV2HeadPaths(in bundle: Bundle = .main) -> [String: ClassifierHeadPaths] {
        var paths: [String: ClassifierHeadPaths] = [:]
        for task in [
            understandingV2ChatModeHeadTask,
            understandingV2TopicGateHeadTask,
            understandingV2RefusalFlagsHeadTask,
            understandingV2EmotionalStateHeadTask,
            understandingV2SlotCompletenessHeadTask,
        ] {
            if let p = classifierHeadPaths(task: task, in: bundle) {
                paths[task] = p
            }
        }
        return paths
    }

    /// True when at least one legacy v2 head and its shared LoRA are bundled.
    public static func understandingV2Bundled(in bundle: Bundle = .main) -> Bool {
        guard understandingV2AdapterPath(in: bundle) != nil else { return false }
        return !understandingV2HeadPaths(in: bundle).isEmpty
    }

    // MARK: - Legacy relational v1 (pairwise heads)

    /// Path to the retired relational v1 LoRA adapter.
    public static func relationalV1AdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: relationalV1AdapterName, ofType: ext)
    }

    /// Bundled relational head paths, keyed by head task name. Returns
    /// an empty dict when none of the 3 heads are bundled. Partial-
    /// bundle handling: `ChatTemplateRelationalStrategy.bundled` REQUIRES
    /// all 3 heads (stricter than understanding-v2 because the router
    /// fusion treats relational outcomes as a coherent triple — partial
    /// triples create asymmetric coverage in `decideMultiTurn`).
    public static func relationalV1HeadPaths(in bundle: Bundle = .main) -> [String: ClassifierHeadPaths] {
        var paths: [String: ClassifierHeadPaths] = [:]
        for task in [
            relationalV1TurnRelHeadTask,
            relationalV1SlotAlignHeadTask,
            relationalV1StanceChangeHeadTask,
        ] {
            if let p = classifierHeadPaths(task: task, in: bundle) {
                paths[task] = p
            }
        }
        return paths
    }

    /// True when the full retired relational stack is bundled.
    public static func relationalV1Bundled(in bundle: Bundle = .main) -> Bool {
        guard relationalV1AdapterPath(in: bundle) != nil else { return false }
        let heads = relationalV1HeadPaths(in: bundle)
        return heads.count == 3
    }

    // MARK: - Telco turn-relation v4

    public static func turnRelationV4AdapterPath(in bundle: Bundle = .main) -> String? {
        bundle.path(forResource: turnRelationV4AdapterName, ofType: ext)
    }

    public static func turnRelationV4HeadPaths(in bundle: Bundle = .main) -> ClassifierHeadPaths? {
        classifierHeadPaths(task: turnRelationV4HeadTask, in: bundle)
    }

    public static func turnRelationV4Bundled(in bundle: Bundle = .main) -> Bool {
        turnRelationV4AdapterPath(in: bundle) != nil
            && turnRelationV4HeadPaths(in: bundle) != nil
    }

    /// Legacy full-bundle check for older experiments. The active cookbook
    /// ship gate is base + tool adapter at boot, plus explicit shared
    /// classifier, turn-relation, RAG, and composer status checks.
    public static func isFullyBundled(in bundle: Bundle = .main) -> Bool {
        return basePath(in: bundle) != nil
            && toolAdapterPath(in: bundle) != nil
            && chatModeRouterAdapterPath(in: bundle) != nil
            && kbExtractorAdapterPath(in: bundle) != nil
    }

    /// True when either the active shared-head stack or legacy paired-head
    /// stack is bundled.
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

    /// True when legacy per-head adapters can run the transitional path.
    public static func pairedClassifierStackBundled(in bundle: Bundle = .main) -> Bool {
        return classifierHeadsBundled(in: bundle)
            && chatModeClfAdapterPath(in: bundle) != nil
            && kbExtractClfAdapterPath(in: bundle) != nil
            && toolSelectorClfAdapterPath(in: bundle) != nil
    }

    // MARK: - Classifier Head Binaries

    // Classifier heads replace generative LoRA calls for classification tasks:
    // one backbone forward pass plus a linear head instead of autoregressive
    // decoding.
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

    /// True when all three legacy classifier head artifact sets are bundled.
    public static func classifierHeadsBundled(in bundle: Bundle = .main) -> Bool {
        return classifierHeadPaths(task: "chat-mode", in: bundle) != nil
            && classifierHeadPaths(task: "kb-extract", in: bundle) != nil
            && classifierHeadPaths(task: "tool-selector", in: bundle) != nil
    }

    // MARK: - Telco shared support-understanding classifier

    /// Names of the 9 active telco support-understanding heads. Each one is a
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

    /// True when all 9 telco head artifact sets and the shared classification
    /// LoRA are bundled. Drives the shared understanding path on the iOS app.
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
