import XCTest
@testable import TelcoTriage

/// End-to-end smoke tests for the Step 6.4 composer path through
/// `VerizonChatDispatcher`. Exercises the route-derivation gate
/// (`ToolRegistry`) + lexical retrieval + composer wiring.
///
/// These tests use a stub Stage A classifier so they don't need
/// llama_cpp models or the full app stack — the composer path only
/// requires `composer + corpus + lexicalRetriever + toolRegistry`.
@MainActor
final class VerizonDispatcherComposerPathTests: XCTestCase {
    private var corpus: RAGUnitCorpus!
    private var retriever: BM25HierarchyRetriever!
    private var composer: DeterministicAnswerComposer!
    private var toolRegistry: ToolRegistry!

    override func setUpWithError() throws {
        try super.setUpWithError()
        corpus = try RAGUnitCorpus.loadFromBundle()
        retriever = BM25HierarchyRetriever(corpus: corpus)
        composer = DeterministicAnswerComposer()
        toolRegistry = ToolRegistry.default(customerContext: CustomerContext())
    }

    // MARK: - tool_action: real tool exists

    func test_restart_router_routes_to_toolAction_with_confirmation() async {
        let result = await dispatch(query: "restart my router")
        XCTAssertEqual(result.source, .composer)
        XCTAssertEqual(result.composerRoute, .toolAction)
        XCTAssertEqual(result.requiresConfirmation, true,
                       "restart-router is a registered ToolIntent with requiresConfirmation=true")
        XCTAssertEqual(result.executableToolIntent, .restartRouter)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "02.07")
        XCTAssertEqual(result.retrievalCandidates.first?.pageID, "02.07")
        XCTAssertFalse(result.retrievalCandidates.isEmpty)
        XCTAssertTrue(result.text.lowercased().contains("confirm"),
                      "tool_action text must carry the Confirm clause")
    }

    func test_speed_test_action_fires_without_confirmation() async {
        // run-speed-test is registered but ToolIntent.requiresConfirmation = false
        // (read-only — no destructive side effect)
        let result = await dispatch(query: "run a speed test")
        XCTAssertEqual(result.composerRoute, .toolAction)
        XCTAssertEqual(result.requiresConfirmation, false)
        XCTAssertEqual(result.executableToolIntent, .runSpeedTest)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "01.02")
    }

    // MARK: - answer_plus_action: question form + real tool

    func test_question_about_restart_routes_to_answerPlusAction() async {
        let result = await dispatch(query: "how do I restart my router?")
        XCTAssertEqual(result.composerRoute, .answerPlusAction,
                       "Question form of a real-tool action → explain + offer")
        XCTAssertEqual(result.requiresConfirmation, true)
        XCTAssertEqual(result.executableToolIntent, .restartRouter)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "02.07")
    }

    func test_telcoSharedLocalAnswerSuppressesActionOffer() async {
        let understanding = makeTelcoUnderstanding(
            routingLane: .localAnswer,
            requiredTool: .restartGateway
        )
        let result = await dispatch(
            query: "how do I restart my router?",
            telcoUnderstanding: understanding
        )

        XCTAssertEqual(result.composerRoute, .ragAnswer)
        XCTAssertEqual(result.requiresConfirmation, false)
        XCTAssertNil(result.executableToolIntent)
    }

    func test_telcoSharedLocalToolRoutesRegisteredToolAction() async {
        let understanding = makeTelcoUnderstanding(
            routingLane: .localTool,
            requiredTool: .restartGateway
        )
        let result = await dispatch(
            query: "restart my router",
            telcoUnderstanding: understanding
        )

        XCTAssertEqual(result.composerRoute, .toolAction)
        XCTAssertEqual(result.requiresConfirmation, true)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "02.07")
    }

    func test_telcoSharedCloudAssistDeflectsBeforeRAG() async {
        var understanding = makeTelcoUnderstanding(routingLane: .cloudAssist)
        understanding = TelcoSharedUnderstanding(
            supportIntent: understanding.supportIntent,
            issueComplexity: understanding.issueComplexity,
            routingLane: understanding.routingLane,
            cloudRequirements: TelcoMultiLabelOutcome(activeLabels: [.accountState], probabilities: [1.0]),
            requiredTool: understanding.requiredTool,
            escalationRisk: understanding.escalationRisk,
            piiRisk: understanding.piiRisk,
            transcriptQuality: understanding.transcriptQuality,
            missingSlots: understanding.missingSlots,
            forwardPassMs: understanding.forwardPassMs,
            headProjectionMs: understanding.headProjectionMs
        )

        let result = await dispatch(
            query: "what is my current bill?",
            telcoUnderstanding: understanding
        )

        XCTAssertEqual(result.composerRoute, .accountNav)
        XCTAssertNil(result.citedRAGUnit)
    }

    func test_telcoSharedHumanEscalationDeflectsBeforeRAG() async {
        let base = makeTelcoUnderstanding(routingLane: .humanEscalation)
        let understanding = TelcoSharedUnderstanding(
            supportIntent: TelcoHeadOutcome(
                label: .agentHandoff,
                confidence: 1.0,
                probabilities: [],
                labelIndex: 7
            ),
            issueComplexity: base.issueComplexity,
            routingLane: base.routingLane,
            cloudRequirements: base.cloudRequirements,
            requiredTool: base.requiredTool,
            escalationRisk: base.escalationRisk,
            piiRisk: base.piiRisk,
            transcriptQuality: base.transcriptQuality,
            missingSlots: base.missingSlots,
            forwardPassMs: base.forwardPassMs,
            headProjectionMs: base.headProjectionMs
        )

        let result = await dispatch(
            query: "get me a human",
            telcoUnderstanding: understanding
        )

        XCTAssertEqual(result.composerRoute, .liveAgent)
        XCTAssertNil(result.citedRAGUnit)
    }

    func test_fallbackPolicyRoutesBillingBeforeRetrieval() async {
        let result = await dispatch(query: "Hello I need my bill")

        XCTAssertEqual(result.composerRoute, .accountNav)
        XCTAssertNil(result.citedRAGUnit)
        XCTAssertEqual(result.deepLink, AnswerComposerConstants.myVerizonURL)
    }

    func test_fallbackPolicyRoutesHumanHelpBeforeRetrieval() async {
        let result = await dispatch(query: "Some one can help me?")

        XCTAssertEqual(result.composerRoute, .liveAgent)
        XCTAssertNil(result.citedRAGUnit)
        XCTAssertEqual(result.deepLink, AnswerComposerConstants.liveAgentPhone)
    }

    func test_fallbackPolicyRoutesRealHumanRequestBeforeRetrieval() async {
        let result = await dispatch(query: "Need real human for help")

        XCTAssertEqual(result.composerRoute, .liveAgent)
        XCTAssertNil(result.citedRAGUnit)
        XCTAssertEqual(result.deepLink, AnswerComposerConstants.liveAgentPhone)
    }

    func test_fallbackPolicyRejectsUnsupportedEmailCommandBeforeRetrieval() async {
        let result = await dispatch(query: "Send for email")

        XCTAssertEqual(result.composerRoute, .outOfScope)
        XCTAssertNil(result.citedRAGUnit)
        XCTAssertNil(result.deepLink)
    }

    func test_lowConfidenceHardPolicySignalsDoNotDeflectBeforeRAG() async {
        let base = makeTelcoUnderstanding(routingLane: .localAnswer)
        let understanding = TelcoSharedUnderstanding(
            supportIntent: TelcoHeadOutcome(
                label: .agentHandoff,
                confidence: 0.49,
                probabilities: [],
                labelIndex: 7
            ),
            issueComplexity: TelcoHeadOutcome(
                label: .humanRequired,
                confidence: 0.64,
                probabilities: [],
                labelIndex: 4
            ),
            routingLane: TelcoHeadOutcome(
                label: .humanEscalation,
                confidence: 0.45,
                probabilities: [],
                labelIndex: 3
            ),
            cloudRequirements: base.cloudRequirements,
            requiredTool: base.requiredTool,
            escalationRisk: base.escalationRisk,
            piiRisk: base.piiRisk,
            transcriptQuality: base.transcriptQuality,
            missingSlots: base.missingSlots,
            forwardPassMs: base.forwardPassMs,
            headProjectionMs: base.headProjectionMs
        )

        let result = await dispatch(
            query: "restart my router",
            telcoUnderstanding: understanding
        )

        XCTAssertEqual(result.composerRoute, .toolAction)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "02.07")
        XCTAssertNotEqual(result.composerRoute, .liveAgent)
    }

    func test_lowConfidenceLocalToolSignalDoesNotForceToolAction() async {
        let base = makeTelcoUnderstanding(routingLane: .localAnswer)
        let understanding = TelcoSharedUnderstanding(
            supportIntent: base.supportIntent,
            issueComplexity: base.issueComplexity,
            routingLane: TelcoHeadOutcome(
                label: .localTool,
                confidence: 0.42,
                probabilities: [],
                labelIndex: 1
            ),
            cloudRequirements: base.cloudRequirements,
            requiredTool: base.requiredTool,
            escalationRisk: base.escalationRisk,
            piiRisk: base.piiRisk,
            transcriptQuality: base.transcriptQuality,
            missingSlots: base.missingSlots,
            forwardPassMs: base.forwardPassMs,
            headProjectionMs: base.headProjectionMs
        )

        let result = await dispatch(
            query: "how do I restart my router?",
            telcoUnderstanding: understanding
        )

        XCTAssertEqual(result.composerRoute, .answerPlusAction)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "02.07")
    }

    // MARK: - rag_answer: NO tool exists (no theatre confirmation)

    func test_change_wifi_password_is_rag_answer_no_confirmation() async {
        // 03.01 Edit Wi-Fi is the most specific grounded page, but no
        // ToolIntent is registered for the shared `network` linkID. Per
        // guardrail #3, no confirmation theatre.
        let result = await dispatch(query: "change my wifi password")
        XCTAssertEqual(result.composerRoute, .ragAnswer)
        XCTAssertEqual(result.requiresConfirmation, false,
                       "no real tool → no confirmation theatre (guardrail #3)")
        XCTAssertEqual(result.citedRAGUnit?.pageID, "03.01")
        XCTAssertFalse(result.text.lowercased().contains("reply 'yes'"),
                       "non-executable support page must NOT show the yes-to-confirm clause")
    }

    func test_share_wifi_password_is_rag_answer_no_confirmation() async {
        // 03.02 Share Wi-Fi — no ToolIntent registered for `share-wifi`.
        let result = await dispatch(query: "share my wifi password")
        XCTAssertEqual(result.composerRoute, .ragAnswer)
        XCTAssertEqual(result.requiresConfirmation, false)
    }

    func test_create_profile_is_grounded_navigation_not_fake_tool() async {
        // 13.02 has action-like language, but no registered create-profile
        // tool exists. The composer should explain/open the page, not
        // manufacture a confirmation flow via the shared `home` link_id.
        let result = await dispatch(query: "add a profile for my son")
        XCTAssertEqual(result.composerRoute, .ragAnswer)
        XCTAssertEqual(result.requiresConfirmation, false)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "13.02")
        XCTAssertTrue(result.text.contains("group their children's devices"))
        XCTAssertFalse(result.text.contains("I found the relevant page"))
        XCTAssertFalse(result.text.contains("Reply 'yes'"))
    }

    // MARK: - Multi-turn reuse

    func test_how_to_do_it_reuses_prior_parental_controls_page() async {
        let context = RetrievalContext(
            priorAssistantText: "I can help with Parental Controls. Open Parental Controls to manage profiles and pause internet.",
            priorPageID: "13.00",
            priorLinkID: "home"
        )
        let result = await dispatch(query: "Can you tell me how to do it", context: context)

        XCTAssertEqual(result.composerRoute, .ragAnswer)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "13.00")
        XCTAssertFalse(
            result.text.lowercased().contains("don't have specific information"),
            "anaphoric follow-up should reuse prior Parental Controls evidence, not fall back to no_rag_answer"
        )
    }

    func test_cannot_find_restart_button_uses_active_restart_task() async {
        let context = RetrievalContext(
            priorAssistantText: "To restart router: select Equipment, then select Restart router.",
            priorPageID: "02.07",
            priorLinkID: "restart-router"
        )
        let result = await dispatch(query: "Not able to find restart button", context: context)

        XCTAssertEqual(result.composerRoute, .ragAnswer)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "02.07")
        XCTAssertEqual(result.citedRAGUnit?.linkID, "restart-router")
        XCTAssertFalse(
            result.text.lowercased().contains("set-top box"),
            "active restart-router task should not drift to set-top-box restart content"
        )
    }

    func test_dialogueRepairV4VerbalizerIsUsedForRepairTurn() async {
        let verbalizer = StubDialogueRepairVerbalizer(text: "Stay on Restart Router and look near the top of Router Details.")
        let context = RetrievalContext(
            priorAssistantText: "To restart router: select Equipment, then Restart router.",
            priorPageID: "02.07",
            priorLinkID: "restart-router"
        )

        let result = await dispatch(
            query: "Not able to find restart button",
            context: context,
            dialogueState: DialogueRepairConversationState(
                priorPageID: "02.07",
                priorLinkID: "restart-router"
            ),
            verbalizer: verbalizer
        )

        let seenInput = await verbalizer.lastInput
        XCTAssertEqual(result.source, .dialogueRepair)
        XCTAssertEqual(result.text, "Stay on Restart Router and look near the top of Router Details.")
        XCTAssertEqual(result.citedRAGUnit?.pageID, "02.07")
        XCTAssertEqual(seenInput?.act, .repairCannotFind)
        XCTAssertEqual(seenInput?.evidence?.pageID, "02.07")
        XCTAssertEqual(seenInput?.conversationState.priorPageID, "02.07")
    }

    func test_dialogueRepairV4KeepsToolConfirmationOwnedByDispatcher() async {
        let verbalizer = StubDialogueRepairVerbalizer(text: "I can try restarting the router from here.")
        let understanding = makeTelcoUnderstanding(
            routingLane: .localTool,
            requiredTool: .restartGateway
        )
        let context = RetrievalContext(
            priorAssistantText: "To restart router: select Equipment, then Restart router.",
            priorPageID: "02.07",
            priorLinkID: "restart-router"
        )

        let result = await dispatch(
            query: "It's not starting anything else I can do",
            context: context,
            telcoUnderstanding: understanding,
            dialogueState: DialogueRepairConversationState(
                priorPageID: "02.07",
                priorLinkID: "restart-router",
                pendingTool: "restart-router",
                frustrationCount: 1,
                pendingConfirmation: true
            ),
            verbalizer: verbalizer
        )

        XCTAssertEqual(result.source, .dialogueRepair)
        XCTAssertEqual(result.composerRoute, .toolAction)
        XCTAssertEqual(result.requiresConfirmation, true)
        XCTAssertEqual(result.executableToolIntent, .restartRouter)
        XCTAssertTrue(
            result.text.contains("Reply 'yes' to confirm."),
            "Swift owns the confirmation clause even when V4 verbalizes the repair text"
        )
    }

    func test_dialogueRepairV4NotUsedForFirstTurnRestartQuestionWithoutQuestionMark() async {
        let verbalizer = StubDialogueRepairVerbalizer(text: "This text should never render.")

        let result = await dispatch(
            query: "how do I restart my router",
            verbalizer: verbalizer
        )

        let seenInput = await verbalizer.lastInput
        XCTAssertEqual(result.source, .composer)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "02.07")
        XCTAssertNil(seenInput)
        XCTAssertFalse(result.text.contains("This text should never render."))
    }

    func test_firstTurnRestartWithDeviceSetupUnderstandingStillRetrievesRestartRouter() async {
        let understanding = makeTelcoUnderstanding(
            supportIntent: .deviceSetup,
            routingLane: .localAnswer,
            requiredTool: .noTool
        )
        let verbalizer = StubDialogueRepairVerbalizer(text: "[forbids_contact(act=turn_style=local_answer)]")

        let result = await dispatch(
            query: "how do I restart my router",
            telcoUnderstanding: understanding,
            verbalizer: verbalizer
        )

        let seenInput = await verbalizer.lastInput
        XCTAssertEqual(result.source, .composer)
        XCTAssertEqual(result.composerRoute, .ragAnswer)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "02.07")
        XCTAssertEqual(result.citedRAGUnit?.linkID, "restart-router")
        XCTAssertNil(seenInput)
        XCTAssertFalse(result.text.contains("forbids_contact"))
        XCTAssertFalse(result.text.lowercased().contains("troubleshoot"))
    }

    func test_stalePriorPageWithoutAssistantTextDoesNotBiasFirstTurnRestart() async {
        let verbalizer = StubDialogueRepairVerbalizer(text: "This text should never render.")
        let staleContext = RetrievalContext(
            priorAssistantText: nil,
            priorPageID: "01.01",
            priorLinkID: "tab-troubleshoot"
        )

        let result = await dispatch(
            query: "how do I restart my router",
            context: staleContext,
            verbalizer: verbalizer
        )

        let seenInput = await verbalizer.lastInput
        XCTAssertEqual(result.source, .composer)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "02.07")
        XCTAssertNil(seenInput)
        XCTAssertFalse(
            result.text.contains("forbids_contact"),
            "invisible prior page state must not trigger dialogue-repair generation"
        )
    }

    func test_dialogueRepairV4RejectsInternalControlOutput() async throws {
        let backend = ScriptedBackend()
        await backend.script(.init(
            matches: "current_user_turn",
            response: "[forbids_contact(act=turn_style=local_answer, language=null, unsafe_contact=false)]"
        ))
        let verbalizer = DialogueRepairVerbalizer(backend: backend, adapterPath: "/tmp/fake.gguf")
        let input = makeDialogueRepairInput(
            act: .repairCannotFind,
            query: "I can't find the restart button"
        )

        let result = await verbalizer.verbalize(input)

        XCTAssertTrue(result.usedFallback)
        XCTAssertEqual(result.extractionMode, "missing_response_field")
        XCTAssertFalse(result.text.contains("forbids_contact"))
        XCTAssertFalse(result.text.contains("unsafe_contact"))
    }

    func test_dialogueRepairV4AcceptsStructuredResponseFieldOnly() async throws {
        let backend = ScriptedBackend()
        await backend.script(.init(
            matches: "current_user_turn",
            response: #"<|tool_call_start|>telco.dialogue_reply(act="repair_cannot_find", response="Stay on Restart Router and look near the top of Router Details.", source_page_id="02.07", source_link_id="restart-router", handoff="none", unsafe_action=false)<|tool_call_end|>"#
        ))
        let verbalizer = DialogueRepairVerbalizer(backend: backend, adapterPath: "/tmp/fake.gguf")
        let input = makeDialogueRepairInput(
            act: .repairCannotFind,
            query: "I can't find the restart button"
        )

        let result = await verbalizer.verbalize(input)

        XCTAssertFalse(result.usedFallback)
        XCTAssertEqual(result.extractionMode, "response_field")
        XCTAssertEqual(result.text, "Stay on Restart Router and look near the top of Router Details.")
    }

    func testEquipmentTileFollowupAnswersRestartSubstep() async {
        let context = RetrievalContext(
            priorAssistantText: "To restart router: select the Equipment tile, then select Restart router.",
            priorPageID: "02.07",
            priorLinkID: "restart-router"
        )
        let result = await dispatch(query: "Where is the equipment tile", context: context)

        XCTAssertEqual(result.composerRoute, .ragAnswer)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "02.07")
        XCTAssertTrue(result.text.contains("Equipment"))
        XCTAssertTrue(result.text.contains("Home page"))
        XCTAssertFalse(
            result.text.lowercased().contains("equipment details"),
            "sub-step follow-up should not be re-synthesized as the Equipment details page"
        )
        XCTAssertFalse(
            result.text.lowercased().contains("want me to do this"),
            "a navigation sub-question inside a tool flow should not create a fresh confirmation offer"
        )
    }

    func test_active_task_context_does_not_block_clear_new_topic() async {
        let context = RetrievalContext(
            priorAssistantText: nil,
            priorPageID: "02.07",
            priorLinkID: "restart-router"
        )
        let result = await dispatch(query: "show me my connected devices", context: context)

        XCTAssertEqual(result.composerRoute, .ragAnswer)
        XCTAssertEqual(result.citedRAGUnit?.pageID, "04.00")
        XCTAssertEqual(result.citedRAGUnit?.linkID, "tab-devices")
    }

    // MARK: - Link grounding: composer only renders selected unit's URL

    func test_rendered_link_is_canonical_url_of_selected_unit() async {
        let result = await dispatch(query: "restart my router")
        let expected = corpus.unit(forPageID: "02.07")?.canonicalURL
        XCTAssertEqual(result.deepLink, expected,
                       "composer can only render the selected unit's canonical_url")
    }

    func test_rendered_link_present_in_known_canonical_set() async {
        let known = corpus.allCanonicalURLs
        for query in [
            "restart my router", "change wifi password", "show me my devices",
            "run a speed test", "share my wifi", "parental controls",
        ] {
            let result = await dispatch(query: query)
            guard let link = result.deepLink else { continue }
            XCTAssertTrue(
                known.contains(link) || known.contains(link.split(separator: "?").first.map(String.init) ?? link),
                "composer rendered an unknown vzhome:// URL on query '\(query)': \(link)"
            )
        }
    }

    // MARK: - Runtime split

    func test_dispatchComposer_doesNotInvokeStageAOrLegacyEvents() async {
        let stageA = CountingStageAClassifier()
        let dispatcher = makeDispatcher(stageA: stageA)

        var events: [VerizonDispatchEvent] = []
        for await event in dispatcher.dispatchComposer(query: "restart my router") {
            events.append(event)
        }

        let stageACalls = await stageA.callCount()
        XCTAssertEqual(stageACalls, 0)
        XCTAssertFalse(events.contains(.stageAStarted))
        XCTAssertFalse(events.contains(.stageBStarted))
        XCTAssertFalse(events.contains { event in
            if case .stageAComplete = event { return true }
            return false
        })
    }

    func test_chatViewModelComposerPathUsesADR028RelationAndBypassesCompositeUnderstanding() async {
        let stageA = CountingStageAClassifier()
        let telcoUnderstanding = CountingTelcoUnderstandingClassifier()
        let understanding = CountingUnderstandingClassifier()
        let relational = CountingRelationalStrategy()
        let dispatcher = makeDispatcher(stageA: stageA)
        let harness = TestChatHarness(
            verizonDispatcher: dispatcher,
            telcoUnderstandingClassifier: telcoUnderstanding,
            understandingClassifier: understanding,
            relationalStrategy: relational
        )

        await harness.send("restart my router")

        let stageACalls = await stageA.callCount()
        let telcoCalls = await telcoUnderstanding.callCount()
        let understandingCalls = await understanding.callCount()
        let relationalTextCalls = await relational.textCallCount()
        let chatModeCalls = await harness.chatModeRouter.recordedQueryCount()
        XCTAssertEqual(stageACalls, 0)
        XCTAssertEqual(telcoCalls, 1)
        XCTAssertEqual(understandingCalls, 0)
        XCTAssertEqual(relationalTextCalls, 1)
        XCTAssertEqual(chatModeCalls, 0)
        XCTAssertEqual(harness.lastAssistantMessage?.trace?.chatModeRuntimeMS, 12)
        XCTAssertEqual(harness.lastAssistantMessage?.trace?.telcoUnderstandingMS, 12)
        XCTAssertNotNil(harness.lastAssistantMessage?.trace?.telcoUnderstanding)
        XCTAssertNotNil(harness.lastAssistantMessage?.trace?.retrievalMS)
        XCTAssertNotNil(harness.lastAssistantMessage?.trace?.routePolicyMS)
        XCTAssertNotNil(harness.lastAssistantMessage?.trace?.composerMS)
        XCTAssertNotNil(harness.lastAssistantMessage?.trace?.totalWallMS)
    }

    func test_chatViewModelComposerPathRecordsADR028BlackboardFacts() async {
        let relational = ScriptedTelcoTurnRelationStrategy([
            .independentNewTask,
            .stepFocus
        ])
        let dispatcher = makeDispatcher()
        let harness = TestChatHarness(
            verizonDispatcher: dispatcher,
            telcoUnderstandingClassifier: CountingTelcoUnderstandingClassifier(),
            relationalStrategy: relational
        )

        await harness.send("how do I restart my router?")
        await harness.send("where is that button?")

        let blackboard = harness.vm.dialogueBlackboard
        XCTAssertEqual(blackboard.lastTurnRelation, .stepFocus)
        XCTAssertEqual(blackboard.priorPageID, "02.07")
        XCTAssertEqual(blackboard.priorLinkID, "restart-router")
        XCTAssertFalse(blackboard.lastRetrievalCandidates.isEmpty)
        XCTAssertNotNil(blackboard.lastPolicyDecision)
        XCTAssertTrue(blackboard.auditTrail.contains { $0.kind == .turnRelation })
        XCTAssertTrue(blackboard.auditTrail.contains { $0.kind == .retrieval })
        XCTAssertTrue(blackboard.auditTrail.contains { $0.kind == .policyDecision })
        XCTAssertTrue(blackboard.auditTrail.contains { $0.kind == .responseRendered })
    }

    func test_bareNoOnPendingToolCancelsWithoutExecutionAndRecordsBlackboard() async {
        let relational = ScriptedTelcoTurnRelationStrategy([
            .independentNewTask,
            .confirmationNo
        ])
        let dispatcher = makeDispatcher()
        let harness = TestChatHarness(
            verizonDispatcher: dispatcher,
            telcoUnderstandingClassifier: CountingTelcoUnderstandingClassifier(),
            relationalStrategy: relational
        )

        await harness.send("restart my router")
        XCTAssertNotNil(harness.vm.dialogueBlackboard.pendingToolConfirmation)

        await harness.send("no")

        let blackboard = harness.vm.dialogueBlackboard
        XCTAssertEqual(harness.lastAssistantMessage?.text, "Okay, I won't do that.")
        XCTAssertEqual(blackboard.lastTurnRelation, .confirmationNo)
        XCTAssertNil(blackboard.pendingToolConfirmation)
        XCTAssertTrue(blackboard.auditTrail.contains { $0.kind == .toolCancelled })
        XCTAssertFalse(blackboard.auditTrail.contains { $0.kind == .toolExecuted })
    }

    // MARK: - Helpers

    private func makeDispatcher(
        stageA: VerizonStageAClassifying? = nil,
        verbalizer: DialogueRepairVerbalizing? = nil
    ) -> VerizonChatDispatcher {
        VerizonChatDispatcher(
            stageA: stageA,
            stageB: nil,
            kbFallback: StubKBExtractor(),
            kb: [],
            retriever: nil,
            modelHost: nil,
            composer: composer,
            corpus: corpus,
            lexicalRetriever: retriever,
            toolRegistry: toolRegistry,
            toolAliasMap: ToolAliasMap.default(),
            dialogueRepairVerbalizer: verbalizer
        )
    }

    private func dispatch(
        query: String,
        context: RetrievalContext = .empty,
        telcoUnderstanding: TelcoSharedUnderstanding? = nil,
        dialogueState: DialogueRepairConversationState = .empty,
        verbalizer: DialogueRepairVerbalizing? = nil
    ) async -> VerizonDispatchResult {
        let dispatcher = makeDispatcher(verbalizer: verbalizer)
        // Mirror the V4 dialogue state into the policy-engine state snapshot the
        // production path builds from the blackboard, so these dispatcher-level
        // tests exercise the same state threading as `ChatViewModel`.
        let policyState = TelcoDialogueStateSnapshot(
            hasActiveTask: context.priorPageID != nil,
            priorPageID: context.priorPageID,
            priorLinkID: context.priorLinkID,
            pendingToolID: dialogueState.pendingTool,
            repairAttemptsOnActiveTask: dialogueState.frustrationCount,
            frustrationCount: dialogueState.frustrationCount,
            hasPriorAssistantTurn: context.priorAssistantText != nil
        )
        var finalResult: VerizonDispatchResult?
        for await event in dispatcher.dispatchComposer(
            query: query,
            retrievalContext: context,
            telcoUnderstanding: telcoUnderstanding,
            dialogueState: dialogueState,
            turnRelation: nil,
            policyState: policyState
        ) {
            if case .response(let r) = event { finalResult = r }
        }
        return finalResult ?? VerizonDispatchResult(
            text: "<no response>", lane: .ragStepByStep, source: .composer, totalMs: 0
        )
    }

    private func makeDialogueRepairInput(
        act: DialogueRepairAct,
        query: String
    ) -> DialogueRepairVerbalizerInput {
        DialogueRepairVerbalizerInput(
            currentUserTurn: query,
            priorAssistantText: "To restart router: select Equipment, then Restart router.",
            conversationState: DialogueRepairConversationState(
                priorPageID: "02.07",
                priorLinkID: "restart-router"
            ),
            understanding: nil,
            evidence: corpus.unit(forPageID: "02.07"),
            route: .ragAnswer,
            act: act,
            handoff: .none,
            requiresConfirmation: false
        )
    }
}

private actor StubDialogueRepairVerbalizer: DialogueRepairVerbalizing {
    let text: String
    private(set) var lastInput: DialogueRepairVerbalizerInput?

    init(text: String) {
        self.text = text
    }

    func verbalize(_ input: DialogueRepairVerbalizerInput) async -> DialogueRepairVerbalizerResult {
        lastInput = input
        return DialogueRepairVerbalizerResult(
            text: text,
            rawOutput: "stub",
            usedFallback: false,
            extractionMode: "stub",
            latencyMs: 2
        )
    }
}

// MARK: - Stage A stub

private func makeStageADecision() -> VerizonStageADecision {
    VerizonStageADecision(
        topicGate: .inScope,
        topicGateConfidence: 0.95,
        topicGateProbabilities: [0.05, 0.95],
        refusalFlags: .none,
        refusalFlagsProbabilities: [0, 0, 0],
        totalMs: 0
    )
}

private func makeTelcoUnderstanding(
    supportIntent: TelcoSupportIntent = .troubleshooting,
    routingLane: TelcoRoutingLane = .localAnswer,
    requiredTool: TelcoRequiredTool = .noTool,
    totalMs: Double = 0
) -> TelcoSharedUnderstanding {
    TelcoSharedUnderstanding(
        supportIntent: TelcoHeadOutcome(
            label: supportIntent,
            confidence: 1.0,
            probabilities: [],
            labelIndex: 0
        ),
        issueComplexity: TelcoHeadOutcome(
            label: .guided,
            confidence: 1.0,
            probabilities: [],
            labelIndex: 1
        ),
        routingLane: TelcoHeadOutcome(
            label: routingLane,
            confidence: 1.0,
            probabilities: [],
            labelIndex: 0
        ),
        cloudRequirements: TelcoMultiLabelOutcome(activeLabels: [], probabilities: []),
        requiredTool: TelcoHeadOutcome(
            label: requiredTool,
            confidence: 1.0,
            probabilities: [],
            labelIndex: 0
        ),
        escalationRisk: TelcoHeadOutcome(
            label: .low,
            confidence: 1.0,
            probabilities: [],
            labelIndex: 0
        ),
        piiRisk: TelcoHeadOutcome(
            label: .safe,
            confidence: 1.0,
            probabilities: [],
            labelIndex: 0
        ),
        transcriptQuality: TelcoHeadOutcome(
            label: .clean,
            confidence: 1.0,
            probabilities: [],
            labelIndex: 0
        ),
        missingSlots: TelcoMultiLabelOutcome(activeLabels: [], probabilities: []),
        forwardPassMs: totalMs,
        headProjectionMs: 0
    )
}

private struct StubStageAClassifier: VerizonStageAClassifying {
    func classify(query: String) async throws -> VerizonStageADecision {
        makeStageADecision()
    }
}

private actor CountingStageAClassifier: VerizonStageAClassifying {
    private var calls = 0

    func classify(query: String) async throws -> VerizonStageADecision {
        calls += 1
        return makeStageADecision()
    }

    func callCount() -> Int {
        calls
    }
}

private actor CountingUnderstandingClassifier: QueryUnderstandingClassifying {
    private var calls = 0

    func classify(query: String) async throws -> QueryUnderstanding {
        calls += 1
        return QueryUnderstanding(
            chatMode: ChatModePrediction(
                mode: .kbQuestion,
                confidence: 1.0,
                reasoning: "test should not be called",
                runtimeMS: 1
            )
        )
    }

    func callCount() -> Int {
        calls
    }
}

private actor CountingTelcoUnderstandingClassifier: TelcoSharedUnderstandingClassifying {
    private var calls = 0

    func classify(query: String) async throws -> TelcoSharedUnderstanding {
        calls += 1
        return makeTelcoUnderstanding(
            routingLane: .localTool,
            requiredTool: .restartGateway,
            totalMs: 12
        )
    }

    func callCount() -> Int {
        calls
    }
}

private actor CountingRelationalStrategy: RelationalHeadsStrategy {
    private var classifyCalls = 0
    private var textCalls = 0

    func classify(
        currentUserQuery: String,
        priorUserHidden: [Float]?,
        priorAssistantHidden: [Float]?
    ) async throws -> RelationalOutcomes {
        classifyCalls += 1
        return .none
    }

    func classifyFromText(
        currentUserQuery: String,
        priorAssistantText: String?,
        priorUserText: String?
    ) async throws -> RelationalOutcomes {
        textCalls += 1
        return .none
    }

    func textCallCount() -> Int {
        textCalls
    }
}

private actor ScriptedTelcoTurnRelationStrategy: RelationalHeadsStrategy {
    private var labels: [TelcoTurnRelation]

    init(_ labels: [TelcoTurnRelation]) {
        self.labels = labels
    }

    func classify(
        currentUserQuery: String,
        priorUserHidden: [Float]?,
        priorAssistantHidden: [Float]?
    ) async throws -> RelationalOutcomes {
        _ = currentUserQuery
        _ = priorUserHidden
        _ = priorAssistantHidden
        return .none
    }

    func classifyFromText(
        currentUserQuery: String,
        priorAssistantText: String?,
        priorUserText: String?,
        runtimeState: RelationalRuntimeState
    ) async throws -> RelationalOutcomes {
        _ = currentUserQuery
        _ = priorAssistantText
        _ = priorUserText
        _ = runtimeState
        let label = labels.isEmpty ? .independentNewTask : labels.removeFirst()
        return RelationalOutcomes(
            telcoTurnRelation: TelcoTurnRelationOutcome(value: label, confidence: 1.0),
            turnRelationship: TurnRelationshipOutcome(
                value: TelcoTurnRelationV4Strategy.mapToLegacyRelationship(label),
                confidence: 1.0
            ),
            runtimeMs: 0
        )
    }
}

private extension ScriptedChatModeRouter {
    func recordedQueryCount() -> Int {
        recordedQueries.count
    }
}
