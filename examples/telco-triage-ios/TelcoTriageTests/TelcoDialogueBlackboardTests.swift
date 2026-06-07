import XCTest
@testable import TelcoTriage

final class TelcoDialogueBlackboardTests: XCTestCase {
    func test_fallback_yesWithoutPendingIsAmbiguous() {
        let relation = TelcoDialogueBlackboardReducer.fallbackRelation(
            for: "yes",
            blackboard: TelcoDialogueBlackboard()
        )

        XCTAssertEqual(relation, .ambiguousShortTurn)
    }

    func test_fallback_yesWithPendingIsConfirmation() {
        let blackboard = TelcoDialogueBlackboardReducer.setPendingTool(
            pendingRestart,
            on: TelcoDialogueBlackboard()
        )

        let relation = TelcoDialogueBlackboardReducer.fallbackRelation(
            for: "yes",
            blackboard: blackboard
        )

        XCTAssertEqual(relation, .confirmationYes)
    }

    func test_confirmationNoClearsPendingWithoutExecution() {
        let initial = TelcoDialogueBlackboardReducer.setPendingTool(
            pendingRestart,
            on: TelcoDialogueBlackboard()
        )

        let reduced = TelcoDialogueBlackboardReducer.reduce(
            initial,
            userTurn: "no",
            observedRelation: .confirmationNo
        )

        XCTAssertNil(reduced.pendingToolConfirmation)
        XCTAssertEqual(reduced.lastTurnRelation, .confirmationNo)
        XCTAssertTrue(reduced.auditTrail.contains { $0.kind == .toolCancelled })
        XCTAssertFalse(reduced.auditTrail.contains { $0.kind == .toolExecuted })
    }

    func test_topicSwitchClearsActiveAndPendingState() {
        let initial = TelcoDialogueBlackboardReducer.setPendingTool(
            pendingRestart,
            on: TelcoDialogueBlackboard(
                activeTaskID: "restart-router",
                priorPageID: "02.07",
                priorLinkID: "restart-router"
            )
        )

        let reduced = TelcoDialogueBlackboardReducer.reduce(
            initial,
            userTurn: "actually show connected devices",
            observedRelation: .topicSwitch,
            selectedPageID: "04.00",
            selectedLinkID: "tab-devices",
            selectedTitle: "All devices page"
        )

        XCTAssertNil(reduced.pendingToolConfirmation)
        XCTAssertEqual(reduced.priorPageID, "04.00")
        XCTAssertEqual(reduced.priorLinkID, "tab-devices")
        XCTAssertEqual(reduced.activeTaskID, "tab-devices")
        XCTAssertTrue(reduced.auditTrail.contains { $0.kind == .stateCleared })
    }

    func test_independentNewTaskClearsStaleRetrievalContextBeforeDispatch() {
        let initial = TelcoDialogueBlackboard(
            activeTaskID: "restart-router",
            priorPageID: "02.07",
            priorLinkID: "restart-router"
        )

        let reduced = TelcoDialogueBlackboardReducer.reduce(
            initial,
            userTurn: "show connected devices",
            observedRelation: .independentNewTask
        )

        XCTAssertNil(reduced.activeTaskID)
        XCTAssertNil(reduced.priorPageID)
        XCTAssertNil(reduced.priorLinkID)
        XCTAssertTrue(reduced.auditTrail.contains { $0.kind == .stateCleared })
    }

    func test_repairFailedIncrementsFrustrationAndTaskCounter() {
        let initial = TelcoDialogueBlackboard(
            activeTaskID: "restart-router",
            priorPageID: "02.07",
            priorLinkID: "restart-router"
        )

        let first = TelcoDialogueBlackboardReducer.reduce(
            initial,
            userTurn: "it didn't work",
            observedRelation: .repairFailed
        )
        let second = TelcoDialogueBlackboardReducer.reduce(
            first,
            userTurn: "still not working",
            observedRelation: .repairFailed
        )

        XCTAssertEqual(first.frustrationCount, 1)
        XCTAssertFalse(first.shouldEscalateRepair())
        XCTAssertEqual(second.frustrationCount, 2)
        XCTAssertTrue(second.shouldEscalateRepair())
        XCTAssertEqual(second.failedAttemptCountByTask["restart-router"], 2)
    }

    func test_recordToolExecutedClearsPendingAndAuditsTool() {
        let initial = TelcoDialogueBlackboardReducer.setPendingTool(
            pendingRestart,
            on: TelcoDialogueBlackboard()
        )

        let reduced = TelcoDialogueBlackboardReducer.recordToolExecuted(on: initial)

        XCTAssertNil(reduced.pendingToolConfirmation)
        XCTAssertEqual(reduced.auditTrail.last?.kind, .toolExecuted)
        XCTAssertEqual(reduced.auditTrail.last?.reasonCode, "restart-router")
    }

    func test_recordRetrievalAndPolicyStoresCandidatesAndSelectedEvidence() {
        let initial = TelcoDialogueBlackboard(lastTurnRelation: .stepFocus)
        let reduced = TelcoDialogueBlackboardReducer.recordRetrievalAndPolicy(
            on: initial,
            retrievalCandidates: [
                TelcoRetrievalCandidate(pageID: "02.07", linkID: "restart-router", score: 2.4),
                TelcoRetrievalCandidate(pageID: "02.00", linkID: "equipment", score: 1.1),
            ],
            selectedPageID: "02.07",
            selectedLinkID: "restart-router",
            selectedTitle: "Restart router",
            policyDecision: TelcoPolicyDecision(route: .answerPlusAction, requiresConfirmation: true)
        )

        XCTAssertEqual(reduced.lastRetrievalCandidates.count, 2)
        XCTAssertEqual(reduced.priorPageID, "02.07")
        XCTAssertEqual(reduced.priorLinkID, "restart-router")
        XCTAssertEqual(reduced.priorEvidenceTitle, "Restart router")
        XCTAssertEqual(reduced.lastPolicyDecision?.route, .answerPlusAction)
        XCTAssertTrue(reduced.auditTrail.contains { $0.kind == .retrieval })
        XCTAssertTrue(reduced.auditTrail.contains { $0.kind == .policyDecision })
    }

    func test_retrievalContextProjectsPriorState() {
        let blackboard = TelcoDialogueBlackboard(
            lastAssistantSummary: "Open Home, then Network.",
            priorPageID: "03.00",
            priorLinkID: "network"
        )

        let context = blackboard.retrievalContext

        XCTAssertEqual(context.priorAssistantText, "Open Home, then Network.")
        XCTAssertEqual(context.priorPageID, "03.00")
        XCTAssertEqual(context.priorLinkID, "network")
    }

    private var pendingRestart: TelcoPendingTool {
        TelcoPendingTool(
            toolID: "restart-router",
            intent: .restartRouter,
            pageID: "02.07",
            linkID: "restart-router"
        )
    }
}
