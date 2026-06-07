import Foundation
import os.log

public enum TelcoSharedUnderstandingError: Error, LocalizedError {
    case missingArtifact(name: String)
    case unknownLabel(task: String, label: String)
    case headLoadFailure(task: String, underlying: Error)
    case backendFailure(underlying: Error)

    public var errorDescription: String? {
        switch self {
        case .missingArtifact(let name):
            return "Telco shared understanding artifact missing: \(name)"
        case .unknownLabel(let task, let label):
            return "Telco shared understanding head \(task) emitted unknown label: \(label)"
        case .headLoadFailure(let task, let underlying):
            return "Telco shared understanding head \(task) failed to load: \(underlying.localizedDescription)"
        case .backendFailure(let underlying):
            return "Telco shared understanding backend failure: \(underlying.localizedDescription)"
        }
    }
}

public protocol TelcoSharedUnderstandingClassifying: Sendable {
    func classify(query: String) async throws -> TelcoSharedUnderstanding
}

/// ADR-026 runtime primitive: one shared `telco-shared-clf-v1` LoRA
/// pass, nine linear classifier heads.
///
/// This is the Liquid value layer for Telco Triage. It is deliberately
/// not the old composite stack: no chat-mode-router generation, no
/// private Stage A LoRA pair, no relational generation, no Stage B.
public final class TelcoSharedUnderstandingClassifier: TelcoSharedUnderstandingClassifying, @unchecked Sendable {
    private let backend: LlamaBackend
    private let sharedAdapterPath: String
    private let supportIntentHead: ClassifierHead
    private let issueComplexityHead: ClassifierHead
    private let routingLaneHead: ClassifierHead
    private let cloudRequirementsHead: ClassifierHead
    private let requiredToolHead: ClassifierHead
    private let escalationRiskHead: ClassifierHead
    private let piiRiskHead: ClassifierHead
    private let transcriptQualityHead: ClassifierHead
    private let slotCompletenessHead: ClassifierHead
    /// Off-domain head (ADR-032 pilot). Optional + additive: absent → policy ignores it.
    private let topicScopeHead: ClassifierHead?
    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "TelcoSharedUnderstanding"
    )

    public init(
        backend: LlamaBackend,
        sharedAdapterPath: String,
        supportIntentHead: ClassifierHead,
        issueComplexityHead: ClassifierHead,
        routingLaneHead: ClassifierHead,
        cloudRequirementsHead: ClassifierHead,
        requiredToolHead: ClassifierHead,
        escalationRiskHead: ClassifierHead,
        piiRiskHead: ClassifierHead,
        transcriptQualityHead: ClassifierHead,
        slotCompletenessHead: ClassifierHead,
        topicScopeHead: ClassifierHead? = nil
    ) {
        self.backend = backend
        self.sharedAdapterPath = sharedAdapterPath
        self.supportIntentHead = supportIntentHead
        self.issueComplexityHead = issueComplexityHead
        self.routingLaneHead = routingLaneHead
        self.cloudRequirementsHead = cloudRequirementsHead
        self.requiredToolHead = requiredToolHead
        self.escalationRiskHead = escalationRiskHead
        self.piiRiskHead = piiRiskHead
        self.transcriptQualityHead = transcriptQualityHead
        self.slotCompletenessHead = slotCompletenessHead
        self.topicScopeHead = topicScopeHead
    }

    public static func bundled(
        backend: LlamaBackend,
        bundle: Bundle = .main
    ) throws -> TelcoSharedUnderstandingClassifier {
        guard let adapterPath = TelcoModelBundle.sharedClfAdapterPath(in: bundle) else {
            throw TelcoSharedUnderstandingError.missingArtifact(name: TelcoModelBundle.sharedClfAdapterName)
        }

        func load(_ task: String) throws -> ClassifierHead {
            guard let paths = TelcoModelBundle.classifierHeadPaths(task: task, in: bundle) else {
                throw TelcoSharedUnderstandingError.missingArtifact(name: "\(task)_classifier_{weights,bias,meta}")
            }
            do {
                return try ClassifierHead(
                    weightsURL: paths.weightsURL,
                    biasURL: paths.biasURL,
                    metaURL: paths.metaURL
                )
            } catch {
                throw TelcoSharedUnderstandingError.headLoadFailure(task: task, underlying: error)
            }
        }

        // Off-domain head is additive (ADR-032 pilot): load if bundled, else nil so
        // the app still runs on the pre-topic-scope pack (rollback path, data-plan §9).
        let topicScopeHead: ClassifierHead? = TelcoModelBundle.classifierHeadPaths(
            task: "telco-topic-scope", in: bundle
        ).flatMap {
            try? ClassifierHead(weightsURL: $0.weightsURL, biasURL: $0.biasURL, metaURL: $0.metaURL)
        }

        return try TelcoSharedUnderstandingClassifier(
            backend: backend,
            sharedAdapterPath: adapterPath,
            supportIntentHead: load("telco-support-intent"),
            issueComplexityHead: load("telco-issue-complexity"),
            routingLaneHead: load("telco-routing-lane"),
            cloudRequirementsHead: load("telco-cloud-requirements"),
            requiredToolHead: load("telco-required-tool"),
            escalationRiskHead: load("telco-customer-escalation-risk"),
            piiRiskHead: load("telco-pii-risk"),
            transcriptQualityHead: load("telco-transcript-quality"),
            slotCompletenessHead: load("telco-slot-completeness"),
            topicScopeHead: topicScopeHead
        )
    }

    public func classify(query: String) async throws -> TelcoSharedUnderstanding {
        let t0 = CFAbsoluteTimeGetCurrent()
        let embedding: [Float]
        do {
            try await backend.setAdapter(path: sharedAdapterPath, scale: 1.0)
            // Mean-pool all token hidden states to match the head training
            // contract (train_shared_multi_head.py mean-pools; LFM2.5's final
            // layer is conv, so last-token reads only ~3 tokens). Validated
            // pooling lives in EmbeddingPooling.mean.
            embedding = try await backend.meanPooledEmbedding(prompt: query, clearCache: true)
        } catch {
            throw TelcoSharedUnderstandingError.backendFailure(underlying: error)
        }
        let forwardMs = elapsed(t0)

        let headStart = CFAbsoluteTimeGetCurrent()
        let topicScope: TelcoHeadOutcome<TelcoTopicScope>? = try topicScopeHead.map { head in
            try projectSingle(
                head.classify(embedding), task: "telco-topic-scope", as: TelcoTopicScope.self
            )
        }
        let vector = try TelcoSharedUnderstanding(
            supportIntent: projectSingle(
                supportIntentHead.classify(embedding),
                task: "telco-support-intent",
                as: TelcoSupportIntent.self
            ),
            issueComplexity: projectSingle(
                issueComplexityHead.classify(embedding),
                task: "telco-issue-complexity",
                as: TelcoIssueComplexity.self
            ),
            routingLane: projectSingle(
                routingLaneHead.classify(embedding),
                task: "telco-routing-lane",
                as: TelcoRoutingLane.self
            ),
            cloudRequirements: projectMulti(
                cloudRequirementsHead.classifyMultiLabel(embedding),
                labels: TelcoCloudRequirement.allCases
            ),
            requiredTool: projectSingle(
                requiredToolHead.classify(embedding),
                task: "telco-required-tool",
                as: TelcoRequiredTool.self
            ),
            escalationRisk: projectSingle(
                escalationRiskHead.classify(embedding),
                task: "telco-customer-escalation-risk",
                as: TelcoEscalationRisk.self
            ),
            piiRisk: projectSingle(
                piiRiskHead.classify(embedding),
                task: "telco-pii-risk",
                as: TelcoPIIRisk.self
            ),
            transcriptQuality: projectSingle(
                transcriptQualityHead.classify(embedding),
                task: "telco-transcript-quality",
                as: TelcoTranscriptQuality.self
            ),
            missingSlots: projectMulti(
                slotCompletenessHead.classifyMultiLabel(embedding),
                labels: TelcoMissingSlot.allCases
            ),
            forwardPassMs: forwardMs,
            headProjectionMs: elapsed(headStart),
            topicScope: topicScope
        )

        logger.info(
            "telco_shared_understanding lane=\(vector.routingLane.label.rawValue, privacy: .public) intent=\(vector.supportIntent.label.rawValue, privacy: .public) tool=\(vector.requiredTool.label.rawValue, privacy: .public) cloud=\(vector.cloudRequirements.activeLabels.map(\.rawValue).joined(separator: ","), privacy: .public) total=\(String(format: "%.0f", vector.totalMs), privacy: .public)ms"
        )
        return vector
    }

    private func projectSingle<Label>(
        _ prediction: ClassifierHead.Prediction,
        task: String,
        as type: Label.Type
    ) throws -> TelcoHeadOutcome<Label>
    where Label: RawRepresentable & Sendable & Equatable, Label.RawValue == String {
        guard let label = Label(rawValue: prediction.label) else {
            throw TelcoSharedUnderstandingError.unknownLabel(task: task, label: prediction.label)
        }
        return TelcoHeadOutcome(
            label: label,
            confidence: Double(prediction.confidence),
            probabilities: prediction.probabilities.map(Double.init),
            labelIndex: prediction.labelIndex
        )
    }

    private func projectMulti<Label>(
        _ prediction: ClassifierHead.MultiLabelPrediction,
        labels: [Label]
    ) -> TelcoMultiLabelOutcome<Label>
    where Label: RawRepresentable & Sendable & Equatable, Label.RawValue == String {
        var active: [Label] = []
        for (index, bit) in prediction.binaryVector.enumerated() where bit == 1 {
            if index < labels.count {
                active.append(labels[index])
            }
        }
        return TelcoMultiLabelOutcome(
            activeLabels: active,
            probabilities: prediction.probabilities.map(Double.init)
        )
    }

    private nonisolated func elapsed(_ t0: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
    }
}
