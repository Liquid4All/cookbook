import XCTest
@testable import TelcoTriage

/// ADR-029 §7 — the explicit dialogue **state-operation** resolver.
///
/// The resolver answers "what should happen to the dialogue state?" before
/// retrieval and route policy. These tests pin the seven behaviours the steering
/// note called out, plus the regression guards that make the structural
/// short-turn gate safe (clarification answers, greetings, ambiguous-with-active
/// continuations). The resolver is pure, so these run without a backend.
final class TelcoStateOperationResolverTests: XCTestCase {

    // MARK: - helpers

    private func prior(_ query: String) -> TelcoDeterministicPrior {
        .derive(query: query)
    }

    private func state(
        active: Bool = false,
        priorRouteWasClarify: Bool = false,
        pendingTool: Bool = false,
        repairAttempts: Int = 0
    ) -> TelcoDialogueStateSnapshot {
        TelcoDialogueStateSnapshot(
            hasActiveTask: active,
            priorPageID: active ? "02.07" : nil,
            priorLinkID: active ? "restart-router" : nil,
            pendingToolID: pendingTool ? "restart-router" : nil,
            repairAttemptsOnActiveTask: repairAttempts,
            frustrationCount: repairAttempts,
            hasPriorAssistantTurn: active || priorRouteWasClarify || pendingTool,
            priorRouteWasClarify: priorRouteWasClarify
        )
    }

    private func resolve(
        _ query: String,
        relation: TelcoTurnRelation? = nil,
        state s: TelcoDialogueStateSnapshot? = nil
    ) -> TelcoStateResolution {
        TelcoStateOperationResolver.resolve(
            query: query,
            relation: relation,
            prior: prior(query),
            state: s ?? state()
        )
    }

    // MARK: - 1. fresh local support question after a prior page

    /// A new, independent question must retrieve fresh — never reuse the prior
    /// page's evidence — even when a prior page is still on the blackboard. This
    /// is the artifact the corrected gauge exposed: a fresh same-page answer was
    /// being mis-scored as repair reuse.
    func test_independentNewTask_afterPriorPage_retrievesFresh_notReuse() {
        let r = resolve(
            "how do I change my wifi password",
            relation: .independentNewTask,
            state: state(active: true)
        )
        XCTAssertEqual(r.operation, .updateNewTask)
        XCTAssertEqual(r.retrieval, .fresh)
        XCTAssertNotEqual(r.retrieval, .reusePrior)
    }

    // MARK: - 2. "I can't find it" after prior guidance

    func test_cannotFind_afterGuidance_reusesActiveEvidence() {
        let r = resolve(
            "I can't find that button",
            relation: .repairCannotFind,
            state: state(active: true)
        )
        XCTAssertEqual(r.operation, .repairCannotFind)
        XCTAssertEqual(r.retrieval, .reusePrior)
    }

    // MARK: - 3. "that didn't work"

    func test_didntWork_isRepairFailed() {
        let r = resolve("that didn't work", relation: .repairFailed, state: state(active: true))
        XCTAssertEqual(r.operation, .repairFailed)
        XCTAssertEqual(r.retrieval, .reusePrior)
    }

    // MARK: - 4. "actually show my bill" — clears local, goes account/backend

    func test_topicSwitchToBilling_isUnsupportedBackend_notLocalReuse() {
        // The account lexicon outranks the topic-switch relation: the destination
        // is a live billing system, not a local page to reuse.
        let r = resolve("actually show my bill", relation: .topicSwitch, state: state(active: true))
        XCTAssertEqual(r.operation, .unsupportedBackendOrOutOfScope)
        XCTAssertEqual(r.retrieval, .none)
        XCTAssertEqual(r.reason, "account_lexical")
    }

    // MARK: - 5. direct agent request

    func test_directAgentRequest_isDirectEscalation() {
        let r = resolve("can I please speak to a live agent")
        XCTAssertEqual(r.operation, .directEscalation)
        XCTAssertEqual(r.retrieval, .none)
    }

    // MARK: - 6. ambiguous short turn without valid state → ask

    func test_ambiguousShortTurn_withoutState_asksClarification() {
        for q in ["phone", "where?", "troubleshoot", "change password"] {
            let r = resolve(q, state: state(active: false))
            XCTAssertEqual(r.operation, .askClarification, "expected ask for \(q)")
            XCTAssertEqual(r.retrieval, .none, "expected no retrieval for \(q)")
        }
    }

    /// A bare affirmative with no pending state is also ambiguous → ask, never
    /// ground a fresh page on one word.
    func test_bareYes_withoutState_asksClarification() {
        let r = resolve("Yes", relation: .ambiguousShortTurn, state: state(active: false))
        XCTAssertEqual(r.operation, .askClarification)
    }

    // MARK: - 7. unsupported backend/out-of-scope → no local retrieval

    func test_backendQuery_isUnsupported_noRetrieval() {
        let r = resolve("I want to pay my bill")
        XCTAssertEqual(r.operation, .unsupportedBackendOrOutOfScope)
        XCTAssertEqual(r.retrieval, .none)
    }

    // MARK: - regression guards for the structural short-turn gate

    /// CRITICAL: a short reply to our own clarification question must be treated
    /// as a clarification ANSWER (grounds), not a fresh ambiguous turn (re-asks).
    /// Guarded by `priorRouteWasClarify`. Without this guard the 2-token fixture
    /// answer "Home Verizon" would re-ask and regress the clarification slice.
    func test_shortReply_afterClarify_isClarificationAnswer_notAsk() {
        let r = resolve("Home Verizon", state: state(active: false, priorRouteWasClarify: true))
        XCTAssertEqual(r.operation, .clarificationAnswer)
        XCTAssertNotEqual(r.operation, .askClarification)
    }

    /// A clarification-answer relation grounds **only when the app actually asked**
    /// a clarification (`priorRouteWasClarify`). Then a 1-word reply is the answer.
    func test_clarificationAnswerRelation_withPriorClarify_grounds() {
        let r = resolve(
            "Yes",
            relation: .clarificationAnswer,
            state: state(active: true, priorRouteWasClarify: true)
        )
        XCTAssertEqual(r.operation, .clarificationAnswer)
        XCTAssertEqual(r.retrieval, .priorBias)
    }

    // MARK: - state-gate: clarification_answer requires a prior clarify (§7.2)

    /// ADR-029 §7.2 state-gate: a relation-head `clarification_answer` with NO
    /// prior clarify is an impossible output (we never asked). It must NOT ground
    /// as a clarification answer; with an active task the ambiguous fragment
    /// carries the task over, never re-using a question we didn't pose.
    func test_clarificationAnswerRelation_withoutPriorClarify_doesNotGround() {
        let r = resolve("Home Verizon", relation: .clarificationAnswer, state: state(active: true))
        XCTAssertNotEqual(r.operation, .clarificationAnswer)
        XCTAssertEqual(r.operation, .reuseActiveEvidence)
    }

    /// Same gate, no active task: a vague clarification fragment with nothing to
    /// answer and nothing to continue is asked about, not grounded.
    func test_clarificationAnswerRelation_withoutPriorClarifyOrTask_asks() {
        let r = resolve("Home Verizon", relation: .clarificationAnswer, state: state(active: false))
        XCTAssertNotEqual(r.operation, .clarificationAnswer)
        XCTAssertEqual(r.operation, .askClarification)
    }

    // MARK: - state-gate: confirmation requires a pending tool (§7.2)

    /// ADR-029 §7.2 state-gate: a relation-head `confirmation_yes` with NO pending
    /// tool must never execute or confirm. A bare "Yes" falls through to the
    /// short-turn gate (asks when there is no active task) — never `confirmation_yes`.
    func test_confirmationYes_withoutPendingTool_doesNotConfirm() {
        let r = resolve("Yes", relation: .confirmationYes, state: state(active: false))
        XCTAssertNotEqual(r.operation, .confirmationYes)
        XCTAssertEqual(r.operation, .askClarification)
    }

    /// A multi-word turn the head mislabeled `confirmation_yes`, with no pending
    /// tool, must not confirm — it retrieves fresh as an independent turn instead.
    func test_confirmationYes_withoutPendingTool_onContentTurn_retrievesFresh() {
        let r = resolve("Can I bypass that step", relation: .confirmationYes, state: state(active: false))
        XCTAssertNotEqual(r.operation, .confirmationYes)
        XCTAssertEqual(r.operation, .updateNewTask)
        XCTAssertEqual(r.retrieval, .fresh)
    }

    /// The positive path is preserved: with a real pending tool, `confirmation_yes`
    /// grounds the pending action.
    func test_confirmationYes_withPendingTool_confirms() {
        let r = resolve(
            "Yes",
            relation: .confirmationYes,
            state: state(active: true, pendingTool: true)
        )
        XCTAssertEqual(r.operation, .confirmationYes)
    }

    /// `confirmation_no` is likewise gated: a "no" with nothing pending is not a
    /// declined action; it falls through (asks here, no active task).
    func test_confirmationNo_withoutPendingTool_doesNotDecline() {
        let r = resolve("No", relation: .confirmationNo, state: state(active: false))
        XCTAssertNotEqual(r.operation, .confirmationNo)
        XCTAssertEqual(r.operation, .askClarification)
    }

    func test_confirmationNo_withPendingTool_declines() {
        let r = resolve(
            "No",
            relation: .confirmationNo,
            state: state(active: true, pendingTool: true)
        )
        XCTAssertEqual(r.operation, .confirmationNo)
    }

    // MARK: - state-gate: short turns without valid pending state ask / continue

    /// "How" with no pending clarification (and a noisy head guess) must not become
    /// confirmation or clarification-answer — with no active task it asks.
    func test_how_withoutPendingState_asksNotConfirmOrClarify() {
        let r = resolve("How", relation: .confirmationYes, state: state(active: false))
        XCTAssertEqual(r.operation, .askClarification)
        XCTAssertNotEqual(r.operation, .confirmationYes)
        XCTAssertNotEqual(r.operation, .clarificationAnswer)
    }

    /// "Phone" with no pending clarification clarifies (1 token, no active task).
    func test_phone_withoutPendingState_clarifies() {
        let r = resolve("Phone", state: state(active: false))
        XCTAssertEqual(r.operation, .askClarification)
    }

    // MARK: - state-gate: repair only reuses with active evidence

    /// Repair with no active task to reuse falls back to asking what failed — it
    /// must not reuse evidence that doesn't exist.
    func test_repairFailed_withoutActiveTask_doesNotReuse() {
        let r = resolve("that didn't work", relation: .repairFailed, state: state(active: false))
        XCTAssertEqual(r.operation, .repairFailed)
        XCTAssertEqual(r.retrieval, .none)
        XCTAssertEqual(r.reason, "repair_failed_no_task")
    }

    /// A one-word greeting must reach the greeting rung, not the ambiguity gate.
    func test_greeting_isNotAsked() {
        let r = resolve("hi", state: state(active: false))
        XCTAssertEqual(r.operation, .updateNewTask)
        XCTAssertEqual(r.reason, "greeting")
        XCTAssertNotEqual(r.operation, .askClarification)
    }

    /// CHAT_017 regression: a noisy relation head that mislabels a bare "Hello"
    /// as `escalation_request` must NOT escalate — an exact phatic greeting
    /// outranks relation-head escalation.
    func test_greeting_outranksNoisyRelationEscalation() {
        let r = resolve("Hello", relation: .escalationRequest, state: state(active: false))
        XCTAssertEqual(r.operation, .updateNewTask)
        XCTAssertEqual(r.reason, "greeting")
        XCTAssertNotEqual(r.operation, .directEscalation)
    }

    /// An ambiguous short turn WITH an active task carries the task over (reuses
    /// its evidence) rather than asking — `repair_same_task` is acceptable here.
    func test_ambiguousShortTurn_withActiveTask_reusesEvidence() {
        let r = resolve("which one", state: state(active: true))
        XCTAssertEqual(r.operation, .reuseActiveEvidence)
        XCTAssertEqual(r.retrieval, .reusePrior)
    }

    // MARK: - continuations / topic switch / step focus

    func test_topicSwitch_clearsContext() {
        let r = resolve("actually I want something completely different", relation: .topicSwitch, state: state(active: true))
        XCTAssertEqual(r.operation, .clearContextTopicSwitch)
        XCTAssertEqual(r.retrieval, .fresh)
    }

    func test_continuationWithActiveTask_carriesOverWithPriorBias() {
        let r = resolve("what about the 5ghz band", relation: .continuationSameTask, state: state(active: true))
        XCTAssertEqual(r.operation, .carryoverActiveTask)
        XCTAssertEqual(r.retrieval, .priorBias)
    }

    func test_stepFocusWithActiveTask_retrievesWithPriorBias() {
        let r = resolve("where is the settings menu", relation: .stepFocus, state: state(active: true))
        XCTAssertEqual(r.operation, .retrieveWithPriorBias)
        XCTAssertEqual(r.retrieval, .priorBias)
    }

    func test_continuationWithoutActiveTask_retrievesFresh() {
        let r = resolve("what about the 5ghz band", relation: .continuationSameTask, state: state(active: false))
        XCTAssertEqual(r.operation, .retrieveFresh)
        XCTAssertEqual(r.retrieval, .fresh)
    }

    // MARK: - structural short-turn helper

    func test_isStructurallyShort_matchesTwoTokenCeiling() {
        XCTAssertTrue(TelcoStateOperationResolver.isStructurallyShort("change password"))
        XCTAssertTrue(TelcoStateOperationResolver.isStructurallyShort("How"))
        XCTAssertTrue(TelcoStateOperationResolver.isStructurallyShort("  losing   signal  "))
        XCTAssertFalse(TelcoStateOperationResolver.isStructurallyShort("how do I reset"))
        XCTAssertFalse(TelcoStateOperationResolver.isStructurallyShort(""))
        XCTAssertFalse(TelcoStateOperationResolver.isStructurallyShort("   "))
    }
}
