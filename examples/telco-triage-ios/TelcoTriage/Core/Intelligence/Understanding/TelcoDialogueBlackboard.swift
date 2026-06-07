import Foundation

public enum TelcoTurnRelation: String, CaseIterable, Sendable, Equatable {
    case independentNewTask = "independent_new_task"
    case continuationSameTask = "continuation_same_task"
    case continuationSameSection = "continuation_same_section"
    case stepFocus = "step_focus"
    case clarificationAnswer = "clarification_answer"
    case confirmationYes = "confirmation_yes"
    case confirmationNo = "confirmation_no"
    case repairCannotFind = "repair_cannot_find"
    case repairFailed = "repair_failed"
    case topicSwitch = "topic_switch"
    case escalationRequest = "escalation_request"
    case ambiguousShortTurn = "ambiguous_short_turn"
}

public struct TelcoPendingTool: Sendable, Equatable {
    public let toolID: String
    public let intent: ToolIntent?
    public let pageID: String?
    public let linkID: String?

    public init(toolID: String, intent: ToolIntent?, pageID: String?, linkID: String?) {
        self.toolID = toolID
        self.intent = intent
        self.pageID = pageID
        self.linkID = linkID
    }
}

public struct TelcoRetrievalCandidate: Sendable, Equatable {
    public let pageID: String
    public let linkID: String
    public let score: Double

    public init(pageID: String, linkID: String, score: Double) {
        self.pageID = pageID
        self.linkID = linkID
        self.score = score
    }
}

public struct TelcoPolicyDecision: Sendable, Equatable {
    public let route: ComposerRoute
    public let requiresConfirmation: Bool
    public let handoff: String?

    public init(route: ComposerRoute, requiresConfirmation: Bool, handoff: String? = nil) {
        self.route = route
        self.requiresConfirmation = requiresConfirmation
        self.handoff = handoff
    }
}

public enum TelcoBlackboardEventKind: String, CaseIterable, Sendable, Equatable {
    case userTurn = "user_turn"
    case understanding
    case turnRelation = "turn_relation"
    case retrieval
    case policyDecision = "policy_decision"
    case responseRendered = "response_rendered"
    case toolConfirmationSet = "tool_confirmation_set"
    case toolExecuted = "tool_executed"
    case toolCancelled = "tool_cancelled"
    case clarificationSet = "clarification_set"
    case repairAttempt = "repair_attempt"
    case handoff
    case stateCleared = "state_cleared"
    case fallbackUsed = "fallback_used"
}

public struct TelcoBlackboardEvent: Sendable, Equatable {
    public let turnIndex: Int
    public let kind: TelcoBlackboardEventKind
    public let sourceModule: String
    public let reasonCode: String
    public let pageID: String?
    public let linkID: String?
    public let confidence: Double?
    public let timestamp: Date

    public init(
        turnIndex: Int,
        kind: TelcoBlackboardEventKind,
        sourceModule: String,
        reasonCode: String,
        pageID: String? = nil,
        linkID: String? = nil,
        confidence: Double? = nil,
        timestamp: Date = Date()
    ) {
        self.turnIndex = turnIndex
        self.kind = kind
        self.sourceModule = sourceModule
        self.reasonCode = reasonCode
        self.pageID = pageID
        self.linkID = linkID
        self.confidence = confidence
        self.timestamp = timestamp
    }
}

public struct TelcoDialogueBlackboard: Sendable, Equatable {
    public var conversationID: UUID
    public var turnIndex: Int
    public var lastUserTurn: String?
    public var lastAssistantSummary: String?
    public var activeTaskID: String?
    public var priorPageID: String?
    public var priorLinkID: String?
    public var priorEvidenceTitle: String?
    public var pendingToolConfirmation: TelcoPendingTool?
    public var pendingClarification: PendingClarification?
    public var frustrationCount: Int
    public var failedAttemptCountByTask: [String: Int]
    public var lastUnderstanding: TelcoSharedUnderstanding?
    public var lastTurnRelation: TelcoTurnRelation?
    public var lastRetrievalCandidates: [TelcoRetrievalCandidate]
    public var lastPolicyDecision: TelcoPolicyDecision?
    public var auditTrail: [TelcoBlackboardEvent]

    public init(
        conversationID: UUID = UUID(),
        turnIndex: Int = 0,
        lastUserTurn: String? = nil,
        lastAssistantSummary: String? = nil,
        activeTaskID: String? = nil,
        priorPageID: String? = nil,
        priorLinkID: String? = nil,
        priorEvidenceTitle: String? = nil,
        pendingToolConfirmation: TelcoPendingTool? = nil,
        pendingClarification: PendingClarification? = nil,
        frustrationCount: Int = 0,
        failedAttemptCountByTask: [String: Int] = [:],
        lastUnderstanding: TelcoSharedUnderstanding? = nil,
        lastTurnRelation: TelcoTurnRelation? = nil,
        lastRetrievalCandidates: [TelcoRetrievalCandidate] = [],
        lastPolicyDecision: TelcoPolicyDecision? = nil,
        auditTrail: [TelcoBlackboardEvent] = []
    ) {
        self.conversationID = conversationID
        self.turnIndex = turnIndex
        self.lastUserTurn = lastUserTurn
        self.lastAssistantSummary = lastAssistantSummary
        self.activeTaskID = activeTaskID
        self.priorPageID = priorPageID
        self.priorLinkID = priorLinkID
        self.priorEvidenceTitle = priorEvidenceTitle
        self.pendingToolConfirmation = pendingToolConfirmation
        self.pendingClarification = pendingClarification
        self.frustrationCount = frustrationCount
        self.failedAttemptCountByTask = failedAttemptCountByTask
        self.lastUnderstanding = lastUnderstanding
        self.lastTurnRelation = lastTurnRelation
        self.lastRetrievalCandidates = lastRetrievalCandidates
        self.lastPolicyDecision = lastPolicyDecision
        self.auditTrail = auditTrail
    }

    public var retrievalContext: RetrievalContext {
        RetrievalContext(
            priorAssistantText: lastAssistantSummary,
            priorPageID: priorPageID,
            priorLinkID: priorLinkID
        )
    }

    public func shouldEscalateRepair(threshold: Int = 2) -> Bool {
        frustrationCount >= threshold
    }
}

public enum TelcoDialogueBlackboardReducer {
    public static func reduce(
        _ blackboard: TelcoDialogueBlackboard,
        userTurn: String,
        observedRelation: TelcoTurnRelation?,
        understanding: TelcoSharedUnderstanding? = nil,
        retrievalCandidates: [TelcoRetrievalCandidate] = [],
        selectedPageID: String? = nil,
        selectedLinkID: String? = nil,
        selectedTitle: String? = nil,
        policyDecision: TelcoPolicyDecision? = nil,
        now: Date = Date()
    ) -> TelcoDialogueBlackboard {
        var next = blackboard
        next.turnIndex += 1
        next.lastUserTurn = userTurn
        next.lastUnderstanding = understanding ?? next.lastUnderstanding
        next.lastRetrievalCandidates = retrievalCandidates
        next.lastPolicyDecision = policyDecision ?? next.lastPolicyDecision

        let relation = observedRelation ?? fallbackRelation(for: userTurn, blackboard: blackboard)
        next.lastTurnRelation = relation
        next.auditTrail.append(event(next, .userTurn, "chat", "record_user_turn", now))
        if observedRelation == nil {
            next.auditTrail.append(event(next, .fallbackUsed, "blackboard", "turn_relation_missing", now))
        }
        next.auditTrail.append(event(next, .turnRelation, "blackboard", relation.rawValue, now))

        switch relation {
        case .independentNewTask, .topicSwitch:
            clearActiveState(&next)
            next.auditTrail.append(event(next, .stateCleared, "policy", relation.rawValue, now))
        case .confirmationNo:
            next.pendingToolConfirmation = nil
            next.auditTrail.append(event(next, .toolCancelled, "policy", "confirmation_no", now))
        case .repairFailed, .repairCannotFind:
            // Both are remediation-attempt failures: the prior instruction
            // did not work, or the customer cannot locate the UI element we
            // named. Each consumes one unit of the escalation budget that
            // `TelcoPolicyEngine` reads via `repairAttemptsOnActiveTask`.
            // Active page/link state is intentionally preserved so the next
            // turn can re-guide the same task.
            recordRepairFailure(&next)
            next.auditTrail.append(event(next, .repairAttempt, "policy", relation.rawValue, now))
        case .escalationRequest:
            next.auditTrail.append(event(next, .handoff, "policy", "escalation_request", now))
        case .clarificationAnswer:
            next.pendingClarification = nil
        case .confirmationYes, .ambiguousShortTurn,
             .continuationSameTask, .continuationSameSection, .stepFocus:
            break
        }

        if let selectedPageID {
            next.priorPageID = selectedPageID
            next.priorLinkID = selectedLinkID
            next.priorEvidenceTitle = selectedTitle
            next.activeTaskID = selectedLinkID ?? selectedPageID
        }
        return next
    }

    public static func setPendingTool(
        _ tool: TelcoPendingTool,
        on blackboard: TelcoDialogueBlackboard,
        now: Date = Date()
    ) -> TelcoDialogueBlackboard {
        var next = blackboard
        next.pendingToolConfirmation = tool
        next.auditTrail.append(
            TelcoBlackboardEvent(
                turnIndex: next.turnIndex,
                kind: .toolConfirmationSet,
                sourceModule: "policy",
                reasonCode: "registered_tool_pending",
                pageID: tool.pageID,
                linkID: tool.linkID,
                timestamp: now
            )
        )
        return next
    }

    public static func recordRetrievalAndPolicy(
        on blackboard: TelcoDialogueBlackboard,
        retrievalCandidates: [TelcoRetrievalCandidate],
        selectedPageID: String?,
        selectedLinkID: String?,
        selectedTitle: String?,
        policyDecision: TelcoPolicyDecision?,
        now: Date = Date()
    ) -> TelcoDialogueBlackboard {
        var next = blackboard
        next.lastRetrievalCandidates = retrievalCandidates
        next.lastPolicyDecision = policyDecision ?? next.lastPolicyDecision

        if let selectedPageID {
            next.priorPageID = selectedPageID
            next.priorLinkID = selectedLinkID
            next.priorEvidenceTitle = selectedTitle
            next.activeTaskID = selectedLinkID ?? selectedPageID
        }

        if !retrievalCandidates.isEmpty || selectedPageID != nil {
            next.auditTrail.append(event(next, .retrieval, "retrieval", "candidates_ranked", now))
        }
        if let policyDecision {
            next.auditTrail.append(
                event(next, .policyDecision, "policy", policyDecision.route.wireName, now)
            )
        }

        return next
    }

    public static func recordToolExecuted(
        on blackboard: TelcoDialogueBlackboard,
        now: Date = Date()
    ) -> TelcoDialogueBlackboard {
        var next = blackboard
        let pending = next.pendingToolConfirmation
        next.pendingToolConfirmation = nil
        next.auditTrail.append(
            TelcoBlackboardEvent(
                turnIndex: next.turnIndex,
                kind: .toolExecuted,
                sourceModule: "tool_registry",
                reasonCode: pending?.toolID ?? "unknown_tool",
                pageID: pending?.pageID,
                linkID: pending?.linkID,
                timestamp: now
            )
        )
        return next
    }

    public static func recordToolCancelled(
        on blackboard: TelcoDialogueBlackboard,
        reasonCode: String = "user_declined",
        now: Date = Date()
    ) -> TelcoDialogueBlackboard {
        var next = blackboard
        let pending = next.pendingToolConfirmation
        next.pendingToolConfirmation = nil
        next.auditTrail.append(
            TelcoBlackboardEvent(
                turnIndex: next.turnIndex,
                kind: .toolCancelled,
                sourceModule: "policy",
                reasonCode: reasonCode,
                pageID: pending?.pageID,
                linkID: pending?.linkID,
                timestamp: now
            )
        )
        return next
    }

    public static func recordResponse(
        _ text: String,
        on blackboard: TelcoDialogueBlackboard,
        now: Date = Date()
    ) -> TelcoDialogueBlackboard {
        var next = blackboard
        next.lastAssistantSummary = String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(240))
        next.auditTrail.append(event(next, .responseRendered, "response", "assistant_summary", now))
        return next
    }

    public static func fallbackRelation(
        for userTurn: String,
        blackboard: TelcoDialogueBlackboard
    ) -> TelcoTurnRelation {
        let normalized = userTurn.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if isBareAffirmative(normalized) {
            return blackboard.pendingToolConfirmation == nil ? .ambiguousShortTurn : .confirmationYes
        }
        if isBareNegative(normalized) {
            return blackboard.pendingToolConfirmation == nil ? .ambiguousShortTurn : .confirmationNo
        }
        if ConversationStateRecorder.isLiveAgentRequest(userTurn) {
            return .escalationRequest
        }
        if ConversationStateRecorder.isDidntWorkContinuation(userTurn) {
            return .repairFailed
        }
        if normalized.contains("can't find") || normalized.contains("cannot find") {
            return .repairCannotFind
        }
        if normalized.hasPrefix("where") && blackboard.priorPageID != nil {
            return .stepFocus
        }
        if normalized.hasPrefix("actually") || normalized.hasPrefix("instead") {
            return .topicSwitch
        }
        return .independentNewTask
    }

    private static func clearActiveState(_ blackboard: inout TelcoDialogueBlackboard) {
        blackboard.activeTaskID = nil
        blackboard.priorPageID = nil
        blackboard.priorLinkID = nil
        blackboard.priorEvidenceTitle = nil
        blackboard.pendingToolConfirmation = nil
        blackboard.pendingClarification = nil
    }

    private static func recordRepairFailure(_ blackboard: inout TelcoDialogueBlackboard) {
        blackboard.frustrationCount += 1
        let taskID = blackboard.activeTaskID ?? blackboard.priorLinkID ?? "unknown"
        blackboard.failedAttemptCountByTask[taskID, default: 0] += 1
    }

    private static func event(
        _ blackboard: TelcoDialogueBlackboard,
        _ kind: TelcoBlackboardEventKind,
        _ sourceModule: String,
        _ reasonCode: String,
        _ timestamp: Date
    ) -> TelcoBlackboardEvent {
        TelcoBlackboardEvent(
            turnIndex: blackboard.turnIndex,
            kind: kind,
            sourceModule: sourceModule,
            reasonCode: reasonCode,
            pageID: blackboard.priorPageID,
            linkID: blackboard.priorLinkID,
            timestamp: timestamp
        )
    }

    private static func isBareAffirmative(_ text: String) -> Bool {
        ConversationStateRecorder.isBareAffirmative(text)
    }

    private static func isBareNegative(_ text: String) -> Bool {
        ConversationStateRecorder.isBareNegative(text)
    }
}
