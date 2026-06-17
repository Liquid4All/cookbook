import Foundation
import os.log

/// Combined Stage A signals for a single chat turn.
///
/// The probe-validated heads (topic_gate + refusal_flags) decide which
/// `TelcoLane` the turn takes. See ADR-021 §2.1 (head inventory),
/// §3 (lane routing), §6.5 (probe-set methodology) and
/// `scripts/telco/schemas.py` (Pydantic wire contract).
public struct TelcoStageADecision: Sendable, Equatable {
    public let topicGate: TelcoTopicGate
    public let topicGateConfidence: Double
    public let topicGateProbabilities: [Float]

    public let refusalFlags: TelcoRefusalFlags
    public let refusalFlagsProbabilities: [Float]

    /// Wall-clock for the two adapter-swap + forward-pass + head-projection
    /// cycles. Surfaced in engineering-mode trace so we can tune the
    /// adapter-swap-vs-shared-backbone trade-off (ADR-021 §6.5 follow-ups).
    public let totalMs: Double

    public init(
        topicGate: TelcoTopicGate,
        topicGateConfidence: Double,
        topicGateProbabilities: [Float],
        refusalFlags: TelcoRefusalFlags,
        refusalFlagsProbabilities: [Float],
        totalMs: Double
    ) {
        self.topicGate = topicGate
        self.topicGateConfidence = topicGateConfidence
        self.topicGateProbabilities = topicGateProbabilities
        self.refusalFlags = refusalFlags
        self.refusalFlagsProbabilities = refusalFlagsProbabilities
        self.totalMs = totalMs
    }
}

/// Errors from `TelcoStageAClassifier.classify(query:)`.
public enum TelcoStageAError: Error, LocalizedError {
    case missingArtifact(name: String)
    case backendFailure(underlying: Error)

    public var errorDescription: String? {
        switch self {
        case .missingArtifact(let name):
            return "Telco Stage A artifact missing: \(name)"
        case .backendFailure(let underlying):
            return "Telco Stage A backend failure: \(underlying.localizedDescription)"
        }
    }
}

/// Async-Sendable wrapper exposing `classify(query:)` so the dispatcher
/// can hold a protocol-typed reference and stubs can be substituted in
/// tests without the real backend.
public protocol TelcoStageAClassifying: Sendable {
    func classify(query: String) async throws -> TelcoStageADecision
}

/// Stage A classifier for the Telco RAG router.
///
/// Runs two probe-validated heads (topic_gate, refusal_flags) over the
/// shared LFM2.5-350M base. Each head has its own r=16 LoRA backbone
/// that must be applied before the embedding pass — the head was trained
/// with that exact adapter active, so running the projection on raw base
/// hidden states would silently produce wrong logits.
///
/// This is intentionally separate from `TelcoMultiHeadClassifier`
/// (which owns the chat-mode / kb-extract / tool-selector triad +
/// 9 ADR-015 heads, all sharing one `telco-shared-clf-v1` adapter).
/// The Stage A heads use *private* adapters and a *different* training
/// distribution (RAG-doc corpus + probe-labeled holdout), so co-locating
/// them in the same classifier would conflate two unrelated head sets.
///
/// **Follow-up (ADR-021 §6.5)**: re-train Stage A heads into the shared
/// backbone (telco-shared-clf-v2). That collapses two swaps per Stage A
/// call into zero (the shared adapter is already loaded for the existing
/// triad). For now, two adapter swaps add tens of ms per chat turn.
public final class TelcoStageAClassifier: TelcoStageAClassifying, @unchecked Sendable {
    private let backend: LlamaBackend
    private let topicGateAdapterPath: String
    private let topicGateHead: ClassifierHead
    private let refusalFlagsAdapterPath: String
    private let refusalFlagsHead: ClassifierHead
    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "TelcoStageA"
    )

    public init(
        backend: LlamaBackend,
        topicGateAdapterPath: String,
        topicGateHead: ClassifierHead,
        refusalFlagsAdapterPath: String,
        refusalFlagsHead: ClassifierHead
    ) {
        self.backend = backend
        self.topicGateAdapterPath = topicGateAdapterPath
        self.topicGateHead = topicGateHead
        self.refusalFlagsAdapterPath = refusalFlagsAdapterPath
        self.refusalFlagsHead = refusalFlagsHead
    }

    /// Convenience factory that wires the bundled artifacts. Returns nil
    /// when any artifact is missing (e.g., a fresh clone without the
    /// GGUFs) — caller can then fall back to the legacy chat path.
    public static func bundled(
        backend: LlamaBackend,
        bundle: Bundle = .main
    ) throws -> TelcoStageAClassifier {
        guard let topicAdapter = TelcoModelBundle.telcoTopicGateAdapterPath(in: bundle) else {
            throw TelcoStageAError.missingArtifact(name: "telco-topic-gate-clf-v1.lora.gguf")
        }
        guard let topicHeadPaths = TelcoModelBundle.telcoTopicGateHeadPaths(in: bundle) else {
            throw TelcoStageAError.missingArtifact(name: "telco-topic-gate_classifier_{weights,bias,meta}")
        }
        guard let refusalAdapter = TelcoModelBundle.telcoRefusalFlagsAdapterPath(in: bundle) else {
            throw TelcoStageAError.missingArtifact(name: "telco-refusal-flags-clf-v1.lora.gguf")
        }
        guard let refusalHeadPaths = TelcoModelBundle.telcoRefusalFlagsHeadPaths(in: bundle) else {
            throw TelcoStageAError.missingArtifact(name: "telco-refusal-flags_classifier_{weights,bias,meta}")
        }

        let topicHead = try ClassifierHead(
            weightsURL: topicHeadPaths.weightsURL,
            biasURL: topicHeadPaths.biasURL,
            metaURL: topicHeadPaths.metaURL
        )
        let refusalHead = try ClassifierHead(
            weightsURL: refusalHeadPaths.weightsURL,
            biasURL: refusalHeadPaths.biasURL,
            metaURL: refusalHeadPaths.metaURL
        )

        return TelcoStageAClassifier(
            backend: backend,
            topicGateAdapterPath: topicAdapter,
            topicGateHead: topicHead,
            refusalFlagsAdapterPath: refusalAdapter,
            refusalFlagsHead: refusalHead
        )
    }

    public func classify(query: String) async throws -> TelcoStageADecision {
        let t0 = CFAbsoluteTimeGetCurrent()

        let topicResult: ClassifierHead.Prediction
        let refusalResult: ClassifierHead.MultiLabelPrediction

        do {
            // ---- topic_gate ----
            let topicEmbedding = try await backend.embeddingsWithAdapter(
                prompt: query,
                adapterPath: topicGateAdapterPath,
                clearCache: true
            )
            topicResult = topicGateHead.classify(topicEmbedding)

            // ---- refusal_flags ----
            let refusalEmbedding = try await backend.embeddingsWithAdapter(
                prompt: query,
                adapterPath: refusalFlagsAdapterPath,
                clearCache: true
            )
            refusalResult = refusalFlagsHead.classifyMultiLabel(refusalEmbedding)
        } catch {
            throw TelcoStageAError.backendFailure(underlying: error)
        }

        let totalMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        let topicGate = Self.topicGateFromLabel(topicResult.label)
        // The trained export script left refusal_flags' id2label empty on
        // the first iteration, which made `activeLabels` silently empty —
        // every turn would route as "no flags set". We backfilled the
        // meta JSON, but defending against the failure here too is cheap:
        // map by index directly (the bias vector size confirms 3 classes,
        // and the label ordering is the wire contract per
        // data/finetune/clf/telco_refusal_flags_label_schema.json).
        let flags = Self.refusalFlagsFromBinaryVector(refusalResult.binaryVector)

        let decision = TelcoStageADecision(
            topicGate: topicGate,
            topicGateConfidence: Double(topicResult.confidence),
            topicGateProbabilities: topicResult.probabilities,
            refusalFlags: flags,
            refusalFlagsProbabilities: refusalResult.probabilities,
            totalMs: totalMs
        )

        logger.info(
            "stage_a topic=\(topicResult.label, privacy: .public)/\(String(format: "%.2f", topicResult.confidence), privacy: .public) flags=\(refusalResult.activeLabels.joined(separator: ","), privacy: .public) total=\(String(format: "%.0f", totalMs), privacy: .public)ms"
        )
        return decision
    }

    /// Maps the training-time label string back to the typed enum the
    /// router consumes. Schema lives in
    /// `data/finetune/clf/telco_topic_gate_label_schema.json`.
    static func topicGateFromLabel(_ label: String) -> TelcoTopicGate {
        switch label {
        case "in_scope": return .inScope
        case "out_of_scope": return .outOfScope
        case "greeting": return .greeting
        default:
            // Unknown label = treat as OOS so we refuse rather than route
            // a turn we don't understand. Safer than the alternative.
            return .outOfScope
        }
    }

    /// Maps the multi-label `activeLabels` (subset of the 3 flag names)
    /// to the typed `TelcoRefusalFlags`. Schema lives in
    /// `data/finetune/clf/telco_refusal_flags_label_schema.json`.
    /// Kept for callers that have already parsed labels; the canonical
    /// path is `refusalFlagsFromBinaryVector(_:)` which is meta-agnostic.
    static func refusalFlagsFromLabels(_ activeLabels: [String]) -> TelcoRefusalFlags {
        let set = Set(activeLabels)
        return TelcoRefusalFlags(
            hasRagAnswer: set.contains("has_rag_answer"),
            navigationOnly: set.contains("navigation_only"),
            liveAgentTrigger: set.contains("live_agent_trigger")
        )
    }

    /// Maps the positional binary vector to typed flags. Index 0 =
    /// has_rag_answer, 1 = navigation_only, 2 = live_agent_trigger
    /// — this is the wire contract from training and must not be
    /// reordered without a head re-export. Defensive against a stale or
    /// incomplete meta JSON: even if id2label is empty, the binary vector
    /// still reflects the sigmoid >= 0.5 decisions correctly.
    static func refusalFlagsFromBinaryVector(_ binary: [Int]) -> TelcoRefusalFlags {
        func bit(_ idx: Int) -> Bool {
            return idx < binary.count && binary[idx] == 1
        }
        return TelcoRefusalFlags(
            hasRagAnswer: bit(0),
            navigationOnly: bit(1),
            liveAgentTrigger: bit(2)
        )
    }
}
