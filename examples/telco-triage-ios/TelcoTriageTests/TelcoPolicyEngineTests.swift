import XCTest
@testable import TelcoTriage

/// ADR-029 §1/§2 — focused, deterministic coverage of the authoritative
/// `TelcoPolicyEngine` decision lattice. Each test pins one rung of the
/// risk-ordered policy and the slices the phone-flow holdout exercises:
/// escalation, affective escalation, human-required, repair (in-progress vs
/// exhausted vs no-task), account/backend, unsupported external, clarify,
/// greeting, the grounded capability gate, and the shared-head evidence gate.
///
/// The engine is pure, so these run without a backend, a simulator GGUF, or
/// the chat view model — they validate the policy contract directly.
@MainActor
final class TelcoPolicyEngineTests: XCTestCase {
    private var corpus: RAGUnitCorpus!
    private var toolRegistry: ToolRegistry!
    private var aliasMap: ToolAliasMap!

    override func setUpWithError() throws {
        try super.setUpWithError()
        corpus = try RAGUnitCorpus.loadFromBundle()
        toolRegistry = ToolRegistry.default(customerContext: CustomerContext())
        aliasMap = ToolAliasMap.default()
    }

    // MARK: - Rung 1: sensitive-data safety

    func test_paymentIdentityData_blocksToOutOfScope() {
        let understanding = makeUnderstanding(
            piiRisk: .containsPaymentIdentityData, piiConfidence: 0.95
        )
        let decision = decide(query: "my card number is 4111 1111 1111 1111", understanding: understanding)
        XCTAssertEqual(decision.route, .outOfScope)
        XCTAssertEqual(decision.reason, "pii_block")
    }

    // MARK: - Rung 2: explicit human handoff

    func test_relationEscalationRequest_routesLiveAgent() {
        let decision = decide(query: "tell me about my router", relation: .escalationRequest)
        XCTAssertEqual(decision.route, .liveAgent)
        XCTAssertEqual(decision.handoff, "live_agent")
    }

    func test_explicitHumanPhrase_routesLiveAgent_withoutRelationOrHeads() {
        // No relation, no understanding — the deterministic prior alone must
        // escalate an unambiguous human request.
        let decision = decide(query: "Speak to live agent")
        XCTAssertEqual(decision.route, .liveAgent)
        XCTAssertEqual(decision.reason, "explicit_handoff")
    }

    func test_confidentAgentHandoffHead_routesLiveAgent() {
        let understanding = makeUnderstanding(supportIntent: .agentHandoff, supportConfidence: 0.9)
        let decision = decide(query: "this is hopeless", understanding: understanding)
        XCTAssertEqual(decision.route, .liveAgent)
    }

    // MARK: - Rung 3: affective escalation

    func test_confidentComplaintWithoutLocalEvidence_routesLiveAgent() {
        let understanding = makeUnderstanding(escalationRisk: .complaint, escalationConfidence: 0.85)
        let decision = decide(query: "you guys are terrible", understanding: understanding, unit: nil)
        XCTAssertEqual(decision.route, .liveAgent)
        XCTAssertEqual(decision.reason, "head_escalation")
    }

    /// ADR-029 §2 corroboration gate: a single (miscalibrated) escalation head
    /// must NOT preempt a real local answer. When a page grounds the turn the
    /// grounded answer wins and the head outcome stays trace-only.
    func test_confidentComplaintWithGroundableEvidence_answersLocally() {
        let understanding = makeUnderstanding(escalationRisk: .complaint, escalationConfidence: 0.95)
        let unit = corpus.unit(forPageID: "03.00")  // Network, a real local page
        let decision = decide(query: "change my wifi password", understanding: understanding, unit: unit)
        XCTAssertEqual(decision.route, .ragAnswer)
        XCTAssertNotEqual(decision.route, .liveAgent)
    }

    func test_lowConfidenceComplaint_doesNotEscalateOverGroundedEvidence() {
        let understanding = makeUnderstanding(escalationRisk: .complaint, escalationConfidence: 0.49)
        let unit = corpus.unit(forPageID: "03.00")  // Network, no tool
        let decision = decide(query: "change my wifi password", understanding: understanding, unit: unit)
        XCTAssertEqual(decision.route, .ragAnswer)
        XCTAssertNotEqual(decision.route, .liveAgent)
    }

    // MARK: - Rung 4: human-required complexity

    func test_humanRequiredComplexity_withoutLocalEvidence_routesLiveAgent() {
        let understanding = makeUnderstanding(issueComplexity: .humanRequired, complexityConfidence: 0.8)
        let decision = decide(query: "the technician never sealed the wall", understanding: understanding, unit: nil)
        XCTAssertEqual(decision.route, .liveAgent)
        XCTAssertEqual(decision.reason, "head_escalation")
    }

    // MARK: - Rung 5/8: repair budget

    func test_repairBelowThreshold_reusesActiveEvidence() {
        let state = makeState(priorPageID: "02.07", priorLinkID: "restart-router", repairAttempts: 1)
        let decision = decide(
            query: "that didn't work",
            relation: .repairFailed,
            state: state,
            unit: corpus.unit(forPageID: "03.00")  // a different fresh hit
        )
        XCTAssertEqual(decision.route, .ragAnswer)
        XCTAssertTrue(decision.reuseActiveEvidence)
        XCTAssertEqual(decision.reason, "repair_in_progress")
    }

    func test_repairAtThreshold_escalatesToLiveAgent() {
        let state = makeState(priorPageID: "02.07", priorLinkID: "restart-router", repairAttempts: 2)
        let decision = decide(query: "still broken", relation: .repairFailed, state: state)
        XCTAssertEqual(decision.route, .liveAgent)
        XCTAssertEqual(decision.reason, "repair_exhausted")
    }

    func test_repairWithoutActiveTask_clarifies() {
        let decision = decide(query: "didn't work", relation: .repairFailed, state: .empty)
        XCTAssertEqual(decision.route, .clarify)
        XCTAssertEqual(decision.reason, "repair_without_task")
    }

    func test_cannotFindRepair_reusesActivePageNotFreshHit() {
        let state = makeState(priorPageID: "02.07", priorLinkID: "restart-router", repairAttempts: 1)
        let decision = decide(
            query: "I can't find that button",
            relation: .repairCannotFind,
            state: state,
            unit: corpus.unit(forPageID: "04.00")  // unrelated fresh hit must be ignored
        )
        XCTAssertEqual(decision.route, .ragAnswer)
        XCTAssertTrue(decision.reuseActiveEvidence)
    }

    // MARK: - Rung 6: account / backend

    func test_confidentCloudAssist_routesAccountNav() {
        let understanding = makeUnderstanding(routingLane: .cloudAssist, routingConfidence: 0.9)
        let decision = decide(query: "what is my current bill", understanding: understanding)
        XCTAssertEqual(decision.route, .accountNav)
    }

    func test_billingPrior_routesAccountNav_whenNoHead() {
        let decision = decide(query: "I need to pay my bill")
        XCTAssertEqual(decision.route, .accountNav)
        XCTAssertEqual(decision.reason, "account_lexical")
    }

    // MARK: - Rung 7: unsupported external

    func test_unsupportedExternalAction_outOfScope_whenNoHead() {
        let decision = decide(query: "email me the steps")
        XCTAssertEqual(decision.route, .outOfScope)
        XCTAssertEqual(decision.reason, "unsupported_external")
    }

    // MARK: - Rung 9: clarification

    func test_partialTranscript_clarifies() {
        let understanding = makeUnderstanding(transcriptQuality: .partial, transcriptConfidence: 0.8)
        let decision = decide(query: "the thing is", understanding: understanding)
        XCTAssertEqual(decision.route, .clarify)
    }

    func test_ambiguousShortTurn_withoutEvidenceOrTask_clarifies() {
        let decision = decide(query: "hm", relation: .ambiguousShortTurn, state: .empty, unit: nil)
        XCTAssertEqual(decision.route, .clarify)
        // ADR-029 §7: the ambiguous turn is now resolved by the state-operation
        // layer (ask_clarification), not the legacy grounding-gated fallback.
        XCTAssertEqual(decision.reason, "ask_clarification")
    }

    // MARK: - Rung 10: greeting

    func test_greeting_routesGreeting() {
        let decision = decide(query: "hello", unit: nil)
        XCTAssertEqual(decision.route, .greeting)
    }

    // MARK: - Rung 11: grounded capability gate

    func test_restartRouterImperative_toolActionWithConfirmation() {
        let decision = decide(query: "restart my router", unit: corpus.unit(forPageID: "02.07"))
        XCTAssertEqual(decision.route, .toolAction)
        XCTAssertEqual(decision.executableToolIntent, .restartRouter)
        XCTAssertTrue(decision.requiresConfirmation)
    }

    func test_restartRouterQuestion_answerPlusAction() {
        let decision = decide(query: "how do I restart my router?", unit: corpus.unit(forPageID: "02.07"))
        XCTAssertEqual(decision.route, .answerPlusAction)
    }

    func test_viewPage_staysRagAnswer_noToolTheatre() {
        let decision = decide(query: "change my wifi password", unit: corpus.unit(forPageID: "03.00"))
        XCTAssertEqual(decision.route, .ragAnswer)
        XCTAssertFalse(decision.requiresConfirmation)
        XCTAssertNil(decision.executableToolIntent)
    }

    func test_noEvidence_noRagAnswer() {
        let decision = decide(query: "qwerty asdf zxcv", unit: nil)
        XCTAssertEqual(decision.route, .noRagAnswer)
    }

    // MARK: - Rung 11 shared-head gate (ADR-029 §2)

    func test_confidentLocalAnswerHead_suppressesActionOffer() {
        let understanding = makeUnderstanding(routingLane: .localAnswer, routingConfidence: 1.0)
        let decision = decide(
            query: "how do I restart my router?",
            understanding: understanding,
            unit: corpus.unit(forPageID: "02.07")
        )
        XCTAssertEqual(decision.route, .ragAnswer)
        XCTAssertNil(decision.executableToolIntent)
    }

    func test_confidentLocalToolHead_forcesToolAction() {
        let understanding = makeUnderstanding(routingLane: .localTool, routingConfidence: 1.0)
        let decision = decide(
            query: "restart my router",
            understanding: understanding,
            unit: corpus.unit(forPageID: "02.07")
        )
        XCTAssertEqual(decision.route, .toolAction)
    }

    func test_lowConfidenceLocalToolHead_doesNotForceToolAction() {
        let understanding = makeUnderstanding(routingLane: .localTool, routingConfidence: 0.42)
        let decision = decide(
            query: "how do I restart my router?",
            understanding: understanding,
            unit: corpus.unit(forPageID: "02.07")
        )
        XCTAssertEqual(decision.route, .answerPlusAction)
    }

    /// A confident `local_tool` head on a passive statement (no action mood) is
    /// a head false positive and must not manufacture a side-effecting tool
    /// offer — it stays a grounded answer. The statement is intentionally longer
    /// than the structural short-turn ceiling (ADR-029 §7) so it reaches the
    /// grounding gate this test targets rather than the ambiguity gate.
    func test_confidentLocalToolHeadOnPassiveStatement_staysRagAnswer() {
        let understanding = makeUnderstanding(routingLane: .localTool, routingConfidence: 1.0)
        let decision = decide(
            query: "my connection has issues lately",
            understanding: understanding,
            unit: corpus.unit(forPageID: "02.07")
        )
        XCTAssertEqual(decision.route, .ragAnswer)
        XCTAssertNil(decision.executableToolIntent)
    }

    // MARK: - Rung C.5: out-of-local-scope grounding gate (scope-risk × coverage)

    /// A peripheral-hardware turn the home-internet corpus cannot serve
    /// ("bluetooth printer") retrieves a tangential page (coverage ≈ 0). The
    /// two-factor gate (out-of-scope lexicon × weak grounding) declines instead
    /// of inventing a local answer from the unrelated page.
    func test_outOfScopeHardware_weaklyGrounded_declines() {
        let decision = decide(
            query: "my bluetooth printer won't print",
            unit: corpus.unit(forPageID: "02.07")  // restart-router: does not cover the query
        )
        XCTAssertEqual(decision.route, .outOfScope)
        XCTAssertEqual(decision.reason, "oos_no_grounding")
    }

    func test_outOfScopeFieldService_declines() {
        let decision = decide(
            query: "bury the optic fiber line",
            unit: corpus.unit(forPageID: "10.00")
        )
        XCTAssertEqual(decision.route, .outOfScope)
        XCTAssertEqual(decision.reason, "oos_no_grounding")
    }

    /// Critical non-regression: an *in-scope* support turn with low lexical
    /// coverage (no out-of-scope risk signal) must STILL ground. Coverage alone
    /// never declines — only scope-risk × weak-coverage does. Measured holdout
    /// distributions show ~40 genuine terse support turns with coverage < 0.5,
    /// so a blanket coverage floor would over-decline.
    func test_inScopeWeakCoverage_stillGroundsLocally() {
        let decision = decide(
            query: "why is my router blinking white",
            unit: corpus.unit(forPageID: "05.00")
        )
        XCTAssertEqual(decision.route, .ragAnswer)
        XCTAssertNotEqual(decision.route, .outOfScope)
    }

    /// A confident human-required escalation head outranks the out-of-scope
    /// lexical decline: a field-service complaint with no groundable page
    /// escalates to a human (Tier C precedes the scope gate).
    func test_outOfScopeButConfidentEscalationHead_routesLiveAgent() {
        let understanding = makeUnderstanding(issueComplexity: .humanRequired, complexityConfidence: 0.85)
        let decision = decide(
            query: "the technician who installed my internet damaged the wall",
            understanding: understanding,
            unit: nil
        )
        XCTAssertEqual(decision.route, .liveAgent)
        XCTAssertEqual(decision.reason, "head_escalation")
    }

    // MARK: - Rung A3: backend / account / order topic prior (extended lexicon)

    func test_backendAddAnotherLine_routesAccountNav() {
        let decision = decide(query: "how much would it cost to add another mobile phone line")
        XCTAssertEqual(decision.route, .accountNav)
        XCTAssertEqual(decision.reason, "account_lexical")
    }

    func test_backendSetUpPlan_routesAccountNav() {
        let decision = decide(query: "yes I'd like to set up a plan")
        XCTAssertEqual(decision.route, .accountNav)
        XCTAssertEqual(decision.reason, "account_lexical")
    }

    func test_backendOrderStatus_routesAccountNav() {
        let decision = decide(query: "the system indicates the order is awaiting completion")
        XCTAssertEqual(decision.route, .accountNav)
        XCTAssertEqual(decision.reason, "account_lexical")
    }

    /// The backend/order prior is a hard topic deflect (coverage-independent):
    /// "add another line" retrieves the My-Verizon-App page with *decent*
    /// coverage, yet must still route to account navigation, not a local answer.
    func test_backendOrder_isCoverageIndependent() {
        let decision = decide(
            query: "how much would it cost to add another mobile phone line",
            unit: corpus.unit(forPageID: "10.00")  // My Verizon App — high coverage, still deflects
        )
        XCTAssertEqual(decision.route, .accountNav)
    }

    // MARK: - ADR-029 §7.2 resolver state-gates (end-to-end through decide)

    /// A relation-head `confirmation_yes` with NO pending tool must never execute
    /// or confirm. End-to-end: the engine asks (no tool, no confirmation handshake).
    func test_confirmationYes_withoutPendingTool_doesNotExecuteOrConfirm() {
        let decision = decide(query: "yes", relation: .confirmationYes, state: .empty, unit: nil)
        XCTAssertNotEqual(decision.route, .toolAction)
        XCTAssertFalse(decision.requiresConfirmation)
        XCTAssertNil(decision.executableToolIntent)
        XCTAssertNotEqual(decision.stateOperation, .confirmationYes)
        XCTAssertEqual(decision.route, .clarify)
    }

    /// A relation-head `clarification_answer` with NO prior clarify must not ground
    /// as a clarification answer (we never asked) — the vague turn is asked about.
    func test_clarificationAnswer_withoutPriorClarify_doesNotGround() {
        let decision = decide(query: "home verizon", relation: .clarificationAnswer, state: .empty, unit: nil)
        XCTAssertNotEqual(decision.stateOperation, .clarificationAnswer)
        XCTAssertEqual(decision.route, .clarify)
    }

    /// A genuine clarification answer (the prior route asked) still grounds the
    /// reply against the active task — the gate does not break the happy path.
    func test_clarificationAnswer_withPriorClarify_grounds() {
        let state = TelcoDialogueStateSnapshot(
            hasActiveTask: true,
            priorPageID: "02.07",
            priorLinkID: "restart-router",
            pendingToolID: nil,
            repairAttemptsOnActiveTask: 0,
            frustrationCount: 0,
            hasPriorAssistantTurn: true,
            priorRouteWasClarify: true
        )
        let decision = decide(
            query: "the first one",
            relation: .clarificationAnswer,
            state: state,
            unit: corpus.unit(forPageID: "02.07")
        )
        XCTAssertEqual(decision.stateOperation, .clarificationAnswer)
        XCTAssertNotEqual(decision.route, .clarify)
    }

    /// An exact phatic greeting outranks relation-head escalation noise: a "hello"
    /// the head mislabeled `escalation_request` must greet, never hand off.
    func test_greeting_beatsRelationEscalationNoise() {
        let decision = decide(query: "hello", relation: .escalationRequest, unit: nil)
        XCTAssertEqual(decision.route, .greeting)
        XCTAssertNotEqual(decision.route, .liveAgent)
    }

    // MARK: - Off-domain scope (telco_topic_scope two-factor gate @ τ=0.97, ADR-032 §3c)

    /// Confident off-domain (≥τ=0.97) with no grounding declines cleanly — the fix.
    func test_topicScopeOutOfScope_noGrounding_declines() {
        let u = makeUnderstanding(topicScope: .outOfScope, topicScopeConfidence: 0.99)
        let decision = decide(query: "what is the weather in new york", understanding: u, unit: nil)
        XCTAssertEqual(decision.route, .outOfScope)
        XCTAssertEqual(decision.reason, "topic_scope_off_domain")
    }

    /// Bug fix: a tangential page no longer hijacks an off-domain query (the page
    /// exists but doesn't strongly ground it, so the two-factor gate still declines).
    func test_topicScopeOutOfScope_tangentialPage_declines() {
        let u = makeUnderstanding(topicScope: .outOfScope, topicScopeConfidence: 0.99)
        let unit = corpus.unit(forPageID: "03.00")  // a real but tangential page
        let decision = decide(query: "what is the weather in new york", understanding: u, unit: unit)
        XCTAssertEqual(decision.route, .outOfScope)
        XCTAssertEqual(decision.reason, "topic_scope_off_domain")
    }

    /// Over-decline protection: even a near-certain off-domain head is VETOED when a
    /// page strongly grounds the turn — the grounded answer wins (corroboration-gated).
    func test_topicScopeOutOfScope_stronglyGrounded_answersLocally() {
        let u = makeUnderstanding(topicScope: .outOfScope, topicScopeConfidence: 0.99)
        let unit = corpus.unit(forPageID: "03.00")
        let decision = decide(query: "change my wifi password", understanding: u, unit: unit)
        XCTAssertEqual(decision.route, .ragAnswer)
        XCTAssertNotEqual(decision.reason, "topic_scope_off_domain")
    }

    /// A terse vague in-domain turn the head calls in_scope is never declined.
    func test_topicScopeInScope_terseVague_notDeclined() {
        let u = makeUnderstanding(topicScope: .inScope, topicScopeConfidence: 0.9)
        let decision = decide(query: "my box is blinking", understanding: u, unit: nil)
        XCTAssertNotEqual(decision.reason, "topic_scope_off_domain")
    }

    /// Below the high-precision threshold τ=0.97 the gate does not fire — the
    /// register-safety property: borderline in-domain (gigs/Fios-TV ≈0.92–0.96) answers.
    func test_topicScopeOutOfScope_belowTau_notDeclined() {
        let u = makeUnderstanding(topicScope: .outOfScope, topicScopeConfidence: 0.9)
        let decision = decide(query: "how many gigs of wifi did we use", understanding: u, unit: nil)
        XCTAssertNotEqual(decision.reason, "topic_scope_off_domain")
    }

    /// Additive safety: head absent (rollback / headless) → policy ignores topicScope.
    func test_topicScopeAbsent_noDecline() {
        let u = makeUnderstanding()  // topicScope defaults nil
        let decision = decide(query: "what is the weather in new york", understanding: u, unit: nil)
        XCTAssertNotEqual(decision.reason, "topic_scope_off_domain")
    }

    // MARK: - Harness

    private func decide(
        query: String,
        relation: TelcoTurnRelation? = nil,
        understanding: TelcoSharedUnderstanding? = nil,
        state: TelcoDialogueStateSnapshot = .empty,
        unit: RAGUnit? = nil
    ) -> TelcoPolicyResolution {
        let candidates = unit.map {
            [TelcoRetrievalCandidate(pageID: $0.pageID, linkID: $0.linkID, score: 1.0)]
        } ?? []
        let signals = TelcoPolicySignals(
            query: query,
            relation: relation,
            understanding: understanding,
            state: state,
            prior: TelcoDeterministicPrior.derive(query: query)
        )
        return TelcoPolicyEngine.decide(
            signals: signals,
            candidates: candidates,
            selectedUnit: unit,
            toolRegistry: toolRegistry,
            aliasMap: aliasMap
        )
    }

    private func makeState(
        priorPageID: String?,
        priorLinkID: String?,
        repairAttempts: Int
    ) -> TelcoDialogueStateSnapshot {
        TelcoDialogueStateSnapshot(
            hasActiveTask: priorPageID != nil,
            priorPageID: priorPageID,
            priorLinkID: priorLinkID,
            pendingToolID: nil,
            repairAttemptsOnActiveTask: repairAttempts,
            frustrationCount: repairAttempts,
            hasPriorAssistantTurn: true
        )
    }

    private func makeUnderstanding(
        supportIntent: TelcoSupportIntent = .troubleshooting,
        supportConfidence: Double = 1.0,
        issueComplexity: TelcoIssueComplexity = .guided,
        complexityConfidence: Double = 1.0,
        routingLane: TelcoRoutingLane = .localAnswer,
        routingConfidence: Double = 0.0,
        escalationRisk: TelcoEscalationRisk = .low,
        escalationConfidence: Double = 1.0,
        piiRisk: TelcoPIIRisk = .safe,
        piiConfidence: Double = 1.0,
        transcriptQuality: TelcoTranscriptQuality = .clean,
        transcriptConfidence: Double = 1.0,
        topicScope: TelcoTopicScope? = nil,
        topicScopeConfidence: Double = 1.0
    ) -> TelcoSharedUnderstanding {
        TelcoSharedUnderstanding(
            supportIntent: TelcoHeadOutcome(label: supportIntent, confidence: supportConfidence, probabilities: [], labelIndex: 0),
            issueComplexity: TelcoHeadOutcome(label: issueComplexity, confidence: complexityConfidence, probabilities: [], labelIndex: 0),
            routingLane: TelcoHeadOutcome(label: routingLane, confidence: routingConfidence, probabilities: [], labelIndex: 0),
            cloudRequirements: TelcoMultiLabelOutcome(activeLabels: [], probabilities: []),
            requiredTool: TelcoHeadOutcome(label: .noTool, confidence: 1.0, probabilities: [], labelIndex: 0),
            escalationRisk: TelcoHeadOutcome(label: escalationRisk, confidence: escalationConfidence, probabilities: [], labelIndex: 0),
            piiRisk: TelcoHeadOutcome(label: piiRisk, confidence: piiConfidence, probabilities: [], labelIndex: 0),
            transcriptQuality: TelcoHeadOutcome(label: transcriptQuality, confidence: transcriptConfidence, probabilities: [], labelIndex: 0),
            missingSlots: TelcoMultiLabelOutcome(activeLabels: [], probabilities: []),
            forwardPassMs: 0,
            headProjectionMs: 0,
            topicScope: topicScope.map {
                TelcoHeadOutcome(
                    label: $0, confidence: topicScopeConfidence, probabilities: [], labelIndex: 0
                )
            }
        )
    }
}
