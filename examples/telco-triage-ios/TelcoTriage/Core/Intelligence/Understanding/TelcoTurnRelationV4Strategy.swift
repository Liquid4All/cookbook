import Foundation
import os.log

public enum TelcoTurnRelationV4Error: Error, LocalizedError {
    case missingArtifact(name: String)
    case headLoadFailure(underlying: Error)
    case backendFailure(underlying: Error)
    case unknownLabel(label: String)

    public var errorDescription: String? {
        switch self {
        case .missingArtifact(let name):
            return "Telco turn-relation v4 artifact missing: \(name)"
        case .headLoadFailure(let underlying):
            return "Telco turn-relation v4 head failed to load: \(underlying.localizedDescription)"
        case .backendFailure(let underlying):
            return "Telco turn-relation v4 backend failure: \(underlying.localizedDescription)"
        case .unknownLabel(let label):
            return "Telco turn-relation v4 emitted unknown label: \(label)"
        }
    }
}

/// ADR-028 12-way turn-relation classifier.
///
/// Bundle contract:
/// - `Models/telco-turn-relation-v4.gguf`
/// - `telco-turn-relation_classifier_{weights,bias,meta}.{bin,json}`
///
/// The head was trained with mean pooling, so this strategy explicitly
/// uses `allTokenEmbeddings` and pools on device before projecting the
/// linear head. Deterministic policy handles O(1) safety cases first;
/// the model handles the remaining semantic relation boundary.
public final class TelcoTurnRelationV4Strategy: RelationalHeadsStrategy, @unchecked Sendable {
    private let backend: LlamaBackend
    private let adapterPath: String
    private let head: ClassifierHead
    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "TelcoTurnRelationV4"
    )

    public init(backend: LlamaBackend, adapterPath: String, head: ClassifierHead) {
        self.backend = backend
        self.adapterPath = adapterPath
        self.head = head
    }

    public static func bundled(
        backend: LlamaBackend,
        bundle: Bundle = .main
    ) throws -> TelcoTurnRelationV4Strategy? {
        guard let adapterPath = TelcoModelBundle.turnRelationV4AdapterPath(in: bundle) else {
            return nil
        }
        guard let paths = TelcoModelBundle.turnRelationV4HeadPaths(in: bundle) else {
            return nil
        }
        do {
            let head = try ClassifierHead(
                weightsURL: paths.weightsURL,
                biasURL: paths.biasURL,
                metaURL: paths.metaURL
            )
            return TelcoTurnRelationV4Strategy(
                backend: backend,
                adapterPath: adapterPath,
                head: head
            )
        } catch {
            throw TelcoTurnRelationV4Error.headLoadFailure(underlying: error)
        }
    }

    public func classify(
        currentUserQuery: String,
        priorUserHidden: [Float]?,
        priorAssistantHidden: [Float]?
    ) async throws -> RelationalOutcomes {
        _ = priorUserHidden
        _ = priorAssistantHidden
        return .none
    }

    public func classifyFromText(
        currentUserQuery: String,
        priorAssistantText: String?,
        priorUserText: String?,
        runtimeState: RelationalRuntimeState
    ) async throws -> RelationalOutcomes {
        let priorAssistant = Self.summarize(priorAssistantText)
        guard priorAssistant != nil || runtimeState.pendingConfirmation else {
            return .none
        }

        let start = Date()
        let text = Self.classifierText(
            currentUserQuery: currentUserQuery,
            priorUserText: priorUserText,
            priorAssistantText: priorAssistant,
            runtimeState: runtimeState
        )

        let prediction: ClassifierHead.Prediction
        do {
            try await backend.setAdapter(path: adapterPath, scale: 1.0)
            let pooled = try await backend.meanPooledEmbedding(prompt: text, clearCache: true)
            prediction = head.classify(pooled)
        } catch {
            throw TelcoTurnRelationV4Error.backendFailure(underlying: error)
        }

        guard let modelRelation = TelcoTurnRelation(rawValue: prediction.label) else {
            throw TelcoTurnRelationV4Error.unknownLabel(label: prediction.label)
        }

        let relation = Self.policyOverride(
            for: currentUserQuery,
            runtimeState: runtimeState
        ) ?? modelRelation
        let mapped = Self.mapToLegacyRelationship(relation)
        let runtimeMs = Date().timeIntervalSince(start) * 1_000

        logger.info(
            "turn_relation_v4 label=\(relation.rawValue, privacy: .public) model=\(modelRelation.rawValue, privacy: .public) mapped=\(mapped.wireName, privacy: .public) conf=\(String(format: "%.2f", prediction.confidence), privacy: .public) total=\(String(format: "%.0f", runtimeMs), privacy: .public)ms"
        )

        return RelationalOutcomes(
            telcoTurnRelation: TelcoTurnRelationOutcome(
                value: relation,
                confidence: relation == modelRelation ? Double(prediction.confidence) : 1.0,
                probabilities: prediction.probabilities.map(Double.init)
            ),
            turnRelationship: TurnRelationshipOutcome(
                value: mapped,
                confidence: relation == modelRelation ? Double(prediction.confidence) : 1.0,
                probabilities: Self.legacyProbabilities(from: prediction.probabilities)
            ),
            slotAlignment: nil,
            stanceChange: nil,
            runtimeMs: runtimeMs
        )
    }

    static func classifierText(
        currentUserQuery: String,
        priorUserText: String?,
        priorAssistantText: String?,
        runtimeState: RelationalRuntimeState
    ) -> String {
        [
            "[CURRENT_USER]: \(currentUserQuery)",
            "[PRIOR_USER]: \(Self.valueOrNone(Self.summarize(priorUserText)))",
            "[PRIOR_ASSISTANT_SUMMARY]: \(Self.valueOrNone(priorAssistantText))",
            "[PRIOR_ROUTE]: \(Self.valueOrNone(runtimeState.priorRoute))",
            "[PRIOR_PAGE_ID]: \(Self.valueOrNone(runtimeState.priorPageID))",
            "[PRIOR_LINK_ID]: \(Self.valueOrNone(runtimeState.priorLinkID))",
            "[PENDING_TOOL]: \(Self.valueOrNone(runtimeState.pendingTool))",
            "[PENDING_CONFIRMATION]: \(runtimeState.pendingConfirmation ? "true" : "false")",
            "[PENDING_CLARIFICATION]: \(Self.valueOrNone(runtimeState.pendingClarification))",
            "[FRUSTRATION_COUNT]: \(runtimeState.frustrationCount)",
        ].joined(separator: "\n")
    }

    static func mapToLegacyRelationship(_ relation: TelcoTurnRelation) -> TurnRelationship {
        switch relation {
        case .independentNewTask, .topicSwitch, .escalationRequest, .ambiguousShortTurn:
            return .independent
        case .continuationSameTask, .continuationSameSection, .stepFocus, .repairCannotFind:
            return .anaphoric
        case .clarificationAnswer:
            return .clarificationAnswer
        case .confirmationNo, .repairFailed:
            return .negativeContinuation
        case .confirmationYes:
            return .affirmativeContinuation
        }
    }

    private static func policyOverride(
        for query: String,
        runtimeState: RelationalRuntimeState
    ) -> TelcoTurnRelation? {
        if ConversationStateRecorder.isLiveAgentRequest(query) {
            return .escalationRequest
        }
        if ConversationStateRecorder.isBareAffirmative(query) {
            return runtimeState.pendingConfirmation ? .confirmationYes : .ambiguousShortTurn
        }
        if isBareNegative(query) {
            return runtimeState.pendingConfirmation ? .confirmationNo : .ambiguousShortTurn
        }
        if runtimeState.pendingClarification != nil {
            return .clarificationAnswer
        }
        if ConversationStateRecorder.isDidntWorkContinuation(query) {
            return .repairFailed
        }
        if isCannotFindRepair(query) {
            return .repairCannotFind
        }
        if runtimeState.priorPageID != nil, isStepFocus(query) {
            return .stepFocus
        }
        if runtimeState.priorPageID != nil, isTopicSwitch(query) {
            return .topicSwitch
        }
        return nil
    }

    private static func legacyProbabilities(from probabilities: [Float]) -> [Double] {
        var out = [Double](repeating: 0, count: TurnRelationship.allCases.count)
        for relation in TelcoTurnRelation.allCases {
            guard let index = labelIndex(for: relation), index < probabilities.count else {
                continue
            }
            let mapped = mapToLegacyRelationship(relation)
            out[mapped.rawValue] += Double(probabilities[index])
        }
        return out
    }

    private static func labelIndex(for relation: TelcoTurnRelation) -> Int? {
        switch relation {
        case .independentNewTask:       return 0
        case .continuationSameTask:     return 1
        case .continuationSameSection:  return 2
        case .stepFocus:                return 3
        case .clarificationAnswer:      return 4
        case .confirmationYes:          return 5
        case .confirmationNo:           return 6
        case .repairCannotFind:         return 7
        case .repairFailed:             return 8
        case .topicSwitch:              return 9
        case .escalationRequest:        return 10
        case .ambiguousShortTurn:       return 11
        }
    }

    private static func summarize(_ text: String?, limit: Int = 240) -> String? {
        guard let text else { return nil }
        let collapsed = text
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !collapsed.isEmpty else { return nil }
        return String(collapsed.prefix(limit))
    }

    private static func valueOrNone(_ value: String?) -> String {
        guard let value, !value.isEmpty else { return "none" }
        return value
    }

    private static func normalized(_ text: String) -> String {
        text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    private static func isBareNegative(_ text: String) -> Bool {
        ConversationStateRecorder.isBareNegative(text)
    }

    private static func isCannotFindRepair(_ text: String) -> Bool {
        let q = normalized(text)
        return q.contains("cannot find")
            || q.contains("can't find")
            || q.contains("cant find")
            || q.contains("couldn't find")
            || q.contains("couldnt find")
            || q.contains("not able to find")
            || q.contains("there is no such")
    }

    private static func isStepFocus(_ text: String) -> Bool {
        let q = normalized(text)
        if q.hasPrefix("where") { return true }
        let hasFocusWord = ["button", "option", "menu", "screen", "page", "link", "tab", "tile", "icon"]
            .contains { q.contains($0) }
        return hasFocusWord && (q.contains("where") || q.contains("which") || q.contains("what"))
    }

    private static func isTopicSwitch(_ text: String) -> Bool {
        let q = normalized(text)
        return q.hasPrefix("actually")
            || q.hasPrefix("instead")
            || q.hasPrefix("wait")
            || q.hasPrefix("no show")
            || q.hasPrefix("no, show")
            || q.hasPrefix("forget that")
    }
}
