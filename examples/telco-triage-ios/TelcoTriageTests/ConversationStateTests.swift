import XCTest
@testable import TelcoTriage

/// Coverage for ADR-023 Phase 2 — the session-scoped `ConversationState`
/// container, the pure-function `ConversationStateRecorder` detectors,
/// and the `ConversationSnapshot` projection consumed by the NBA layer.
///
/// These are unit tests with NO ChatViewModel — that's intentional.
/// `ConversationState` is the value-shaped substrate for multi-turn
/// behaviour and must be reasonable to test in isolation; the
/// integration paths live in `ConversationFlowTests` separately.
final class ConversationStateTests: XCTestCase {

    // MARK: - Detector: live-agent requests

    func test_detector_acceptsCanonicalLiveAgentPhrases() {
        // Anchored to the Verizon production corpus: every escalation
        // request in the 50-conversation set uses one of these phrasings.
        XCTAssertTrue(ConversationStateRecorder.isLiveAgentRequest("Agent"))
        XCTAssertTrue(ConversationStateRecorder.isLiveAgentRequest("Live agent"))
        XCTAssertTrue(ConversationStateRecorder.isLiveAgentRequest("Speak to live agent"))
        XCTAssertTrue(ConversationStateRecorder.isLiveAgentRequest("Talk to a representative"))
        XCTAssertTrue(ConversationStateRecorder.isLiveAgentRequest("Chat with a human"))
        XCTAssertTrue(ConversationStateRecorder.isLiveAgentRequest("transfer me"))
        XCTAssertTrue(ConversationStateRecorder.isLiveAgentRequest("Can I talk to a real person?"))
        XCTAssertTrue(ConversationStateRecorder.isLiveAgentRequest("customer service"))
        XCTAssertTrue(ConversationStateRecorder.isLiveAgentRequest("Need real human for help"))
    }

    func test_detector_rejectsAgenticAndAgency() {
        // Word-boundary discipline — "agentic AI", "travel agency"
        // must NOT match.
        XCTAssertFalse(ConversationStateRecorder.isLiveAgentRequest("agentic pre-flight"))
        XCTAssertFalse(ConversationStateRecorder.isLiveAgentRequest("travel agency"))
        XCTAssertFalse(ConversationStateRecorder.isLiveAgentRequest("management"))
    }

    func test_detector_rejectsKBQuestions() {
        // KB questions about the network must not trigger escalation.
        XCTAssertFalse(ConversationStateRecorder.isLiveAgentRequest("how do I restart my router"))
        XCTAssertFalse(ConversationStateRecorder.isLiveAgentRequest("what's my plan"))
    }

    // MARK: - Detector: "didn't work" continuations

    func test_detector_acceptsDidntWorkVariations() {
        XCTAssertTrue(ConversationStateRecorder.isDidntWorkContinuation("didn't work"))
        XCTAssertTrue(ConversationStateRecorder.isDidntWorkContinuation("Didn't work"))
        XCTAssertTrue(ConversationStateRecorder.isDidntWorkContinuation("did not work"))
        XCTAssertTrue(ConversationStateRecorder.isDidntWorkContinuation("doesn't work"))
        XCTAssertTrue(ConversationStateRecorder.isDidntWorkContinuation("not working"))
        XCTAssertTrue(ConversationStateRecorder.isDidntWorkContinuation("still broken"))
        XCTAssertTrue(ConversationStateRecorder.isDidntWorkContinuation("still down"))
        XCTAssertTrue(ConversationStateRecorder.isDidntWorkContinuation("tried that"))
        XCTAssertTrue(ConversationStateRecorder.isDidntWorkContinuation("no luck"))
        XCTAssertTrue(ConversationStateRecorder.isDidntWorkContinuation("same problem"))
        XCTAssertTrue(ConversationStateRecorder.isDidntWorkContinuation("that didn't help"))
    }

    func test_detector_rejectsAffirmations() {
        // "worked" alone is positive feedback, not a continuation.
        XCTAssertFalse(ConversationStateRecorder.isDidntWorkContinuation("that worked"))
        XCTAssertFalse(ConversationStateRecorder.isDidntWorkContinuation("works now"))
    }

    // MARK: - Detector: bare affirmatives

    func test_detector_acceptsBareAffirmatives() {
        XCTAssertTrue(ConversationStateRecorder.isBareAffirmative("yes"))
        XCTAssertTrue(ConversationStateRecorder.isBareAffirmative("Yes"))
        XCTAssertTrue(ConversationStateRecorder.isBareAffirmative("yeah"))
        XCTAssertTrue(ConversationStateRecorder.isBareAffirmative("ok"))
        XCTAssertTrue(ConversationStateRecorder.isBareAffirmative("Okay"))
        XCTAssertTrue(ConversationStateRecorder.isBareAffirmative("sure"))
        XCTAssertTrue(ConversationStateRecorder.isBareAffirmative("go ahead"))
        XCTAssertTrue(ConversationStateRecorder.isBareAffirmative("do it"))
        XCTAssertTrue(ConversationStateRecorder.isBareAffirmative("sounds good"))
        XCTAssertTrue(ConversationStateRecorder.isBareAffirmative("that works"))
        XCTAssertTrue(ConversationStateRecorder.isBareAffirmative("ok."))
    }

    func test_detector_rejectsCompoundAffirmatives() {
        // "yes please pause" is NOT bare — it has a verb the LLM can
        // route independently. The bare path is for terse confirmations
        // only.
        XCTAssertFalse(ConversationStateRecorder.isBareAffirmative("yes please pause"))
        XCTAssertFalse(ConversationStateRecorder.isBareAffirmative("ok restart my router"))
        XCTAssertFalse(ConversationStateRecorder.isBareAffirmative("sure how do I"))
        XCTAssertFalse(ConversationStateRecorder.isBareAffirmative("thanks"))
    }

    // MARK: - State machine: frustration counters

    @MainActor
    func test_recordTurn_incrementsLiveAgentCounter() {
        let state = ConversationState()
        XCTAssertEqual(state.liveAgentRequestCount, 0)

        state.recordTurn(
            userMessage: "Speak to live agent",
            assistantLane: .verizon(.liveAgentEscalation),
            toolDecision: nil
        )
        XCTAssertEqual(state.liveAgentRequestCount, 1)

        state.recordTurn(
            userMessage: "Agent",
            assistantLane: .verizon(.liveAgentEscalation),
            toolDecision: nil
        )
        XCTAssertEqual(state.liveAgentRequestCount, 2)
    }

    @MainActor
    func test_recordTurn_incrementsDidntWorkCounter() {
        let state = ConversationState()

        state.recordTurn(
            userMessage: "didn't work",
            assistantLane: .verizon(.ragStepByStep),
            toolDecision: nil
        )
        state.recordTurn(
            userMessage: "still broken",
            assistantLane: .verizon(.ragStepByStep),
            toolDecision: nil
        )
        XCTAssertEqual(state.didntWorkCount, 2)
        XCTAssertEqual(state.liveAgentRequestCount, 0,
                       "didn't-work signal must not increment the live-agent counter")
    }

    @MainActor
    func test_recordTurn_neutralUserMessageDoesNotIncrement() {
        let state = ConversationState()
        state.recordTurn(
            userMessage: "how do I run a speed test",
            assistantLane: .verizon(.ragStepByStep),
            toolDecision: nil
        )
        XCTAssertEqual(state.liveAgentRequestCount, 0)
        XCTAssertEqual(state.didntWorkCount, 0)
        XCTAssertEqual(state.userTurnCount, 1)
    }

    // MARK: - State machine: pendingClarification lifecycle

    @MainActor
    func test_recordTurn_setsPendingClarificationOnRagClarification() {
        let state = ConversationState()
        state.recordTurn(
            userMessage: "extender problems",
            assistantLane: .verizon(.clarification),
            toolDecision: nil,
            missingSlots: [.location],
            pendingIntent: .rebootExtender,
            originalQuery: "extender problems"
        )

        XCTAssertNotNil(state.pendingClarification)
        XCTAssertEqual(state.pendingClarification?.source, .ragClarification)
        XCTAssertEqual(state.pendingClarification?.intent, .rebootExtender)
        XCTAssertEqual(state.pendingClarification?.missingSlots, [.location])
    }

    @MainActor
    func test_recordTurn_setsPendingClarificationOnToolMissingSlot() {
        let state = ConversationState()
        state.recordTurn(
            userMessage: "pause internet",
            assistantLane: .toolAction,
            toolDecision: nil,
            missingSlots: [.device],
            pendingIntent: .toggleParentalControls,
            originalQuery: "pause internet"
        )

        XCTAssertNotNil(state.pendingClarification)
        XCTAssertEqual(state.pendingClarification?.source, .missingSlot)
        XCTAssertEqual(state.pendingClarification?.intent, .toggleParentalControls)
        XCTAssertEqual(state.pendingClarification?.missingSlots, [.device])
    }

    @MainActor
    func test_recordTurn_clearsPendingOnUnrelatedLane() {
        let state = ConversationState()
        // First turn sets the pending.
        state.recordTurn(
            userMessage: "pause internet",
            assistantLane: .toolAction,
            toolDecision: nil,
            missingSlots: [.device],
            pendingIntent: .toggleParentalControls,
            originalQuery: "pause internet"
        )
        XCTAssertNotNil(state.pendingClarification)

        // Second turn: user changed topics → record a RAG lane WITHOUT
        // missing slots / clarification. Must clear pending.
        state.recordTurn(
            userMessage: "what's my plan",
            assistantLane: .verizon(.ragStepByStep),
            toolDecision: nil
        )
        XCTAssertNil(state.pendingClarification)
    }

    @MainActor
    func test_recordTurn_clearPendingClarificationDirect() {
        let state = ConversationState()
        state.recordTurn(
            userMessage: "x",
            assistantLane: .verizon(.clarification),
            toolDecision: nil,
            missingSlots: [.device],
            pendingIntent: .toggleParentalControls
        )
        XCTAssertNotNil(state.pendingClarification)
        state.clearPendingClarification()
        XCTAssertNil(state.pendingClarification)
    }

    // MARK: - State machine: pendingToolConfirmation

    @MainActor
    func test_recordTurn_setsPendingToolConfirmationOnPrimaryTool() {
        let state = ConversationState()
        let primary = ToolDecision(
            intent: .restartRouter,
            toolID: "restart-router",
            displayName: "Restart Router",
            icon: "arrow.clockwise",
            description: "Reboots the gateway.",
            arguments: [],
            confidence: 0.95,
            reasoning: nil,
            requiresConfirmation: true,
            isDestructive: true,
            isCompoundAttachment: false
        )
        state.recordTurn(
            userMessage: "restart my router",
            assistantLane: .toolAction,
            toolDecision: primary
        )
        XCTAssertEqual(state.pendingToolConfirmation?.toolID, "restart-router")
    }

    @MainActor
    func test_recordTurn_doesNotSetPendingForCompoundAttachment() {
        // The compound tool affordance on a RAG turn is SECONDARY. A
        // bare "yes" should follow the RAG instructions, not silently
        // fire the compound tool.
        let state = ConversationState()
        let compound = ToolDecision(
            intent: .toggleParentalControls,
            toolID: "toggle-parental-controls",
            displayName: "Pause internet",
            icon: "shield",
            description: "Pause internet for a device.",
            arguments: [],
            confidence: 0.95,
            reasoning: nil,
            requiresConfirmation: true,
            isDestructive: false,
            isCompoundAttachment: true
        )
        state.recordTurn(
            userMessage: "how do I pause my son's tablet",
            assistantLane: .verizon(.ragStepByStep),
            toolDecision: compound
        )
        XCTAssertNil(state.pendingToolConfirmation)
    }

    @MainActor
    func test_clearPendingToolConfirmation_clears() {
        let state = ConversationState()
        let primary = ToolDecision(
            intent: .restartRouter,
            toolID: "restart-router",
            displayName: "Restart Router",
            icon: "arrow.clockwise",
            description: "Reboots the gateway.",
            arguments: [],
            confidence: 0.95,
            reasoning: nil,
            requiresConfirmation: true,
            isDestructive: true
        )
        state.recordTurn(
            userMessage: "restart",
            assistantLane: .toolAction,
            toolDecision: primary
        )
        XCTAssertNotNil(state.pendingToolConfirmation)
        state.clearPendingToolConfirmation()
        XCTAssertNil(state.pendingToolConfirmation)
    }

    @MainActor
    func test_reset_clearsEverything() {
        let state = ConversationState()
        state.recordTurn(
            userMessage: "Agent",
            assistantLane: .verizon(.liveAgentEscalation),
            toolDecision: nil
        )
        state.recordTurn(
            userMessage: "didn't work",
            assistantLane: .verizon(.ragStepByStep),
            toolDecision: nil,
            missingSlots: [.device],
            pendingIntent: .toggleParentalControls
        )
        XCTAssertGreaterThan(state.liveAgentRequestCount, 0)
        XCTAssertGreaterThan(state.didntWorkCount, 0)

        state.reset()
        XCTAssertEqual(state.liveAgentRequestCount, 0)
        XCTAssertEqual(state.didntWorkCount, 0)
        XCTAssertEqual(state.userTurnCount, 0)
        XCTAssertNil(state.pendingClarification)
        XCTAssertNil(state.pendingToolConfirmation)
    }

    // MARK: - Snapshot

    @MainActor
    func test_snapshot_capturesCurrentCounters() {
        let state = ConversationState()
        state.recordTurn(
            userMessage: "Speak to live agent",
            assistantLane: .verizon(.liveAgentEscalation),
            toolDecision: nil
        )
        let snap = state.snapshot
        XCTAssertEqual(snap.liveAgentRequestCount, 1)
        XCTAssertEqual(snap.didntWorkCount, 0)
        XCTAssertEqual(snap.userTurnCount, 1)
    }

    @MainActor
    func test_snapshot_isPointInTime() {
        // Snapshot must not retroactively update when the state
        // mutates after capture.
        let state = ConversationState()
        let snap = state.snapshot
        state.recordTurn(
            userMessage: "Agent",
            assistantLane: .verizon(.liveAgentEscalation),
            toolDecision: nil
        )
        XCTAssertEqual(snap.liveAgentRequestCount, 0,
                       "snapshot was taken before recordTurn; must stay frozen")
        XCTAssertEqual(state.liveAgentRequestCount, 1)
    }

    func test_snapshot_emptyIsDefault() {
        let empty = ConversationSnapshot.empty
        XCTAssertEqual(empty.liveAgentRequestCount, 0)
        XCTAssertEqual(empty.didntWorkCount, 0)
        XCTAssertEqual(empty.userTurnCount, 0)
        XCTAssertFalse(empty.hasPendingClarification)
        XCTAssertFalse(empty.hasPendingToolConfirmation)
    }
}
