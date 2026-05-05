import XCTest
@testable import TelcoTriage

/// Integration tests for the ChatViewModel pipeline dispatch.
///
/// Exercises the 4-mode router (`.kbQuestion` / `.toolAction` /
/// `.personalSummary` / `.outOfScope`) + confirmTool / declineTool
/// end-to-end, using `TestChatHarness` to script the mode router,
/// KB extractor, tool selector, and chat-provider responses. No
/// live models, no network.
///
/// Each test reads as a scenario: script the primitive outputs for
/// every stage the query will touch, send, assert the messages list
/// + any state mutation on CustomerContext.
@MainActor
final class ChatViewModelIntegrationTests: XCTestCase {

    // MARK: - Path 1: grounded Q&A (KB question)

    func test_kbQuestionPath_returnsGroundedAnswerWithReadMoreChip() async {
        // Mode router classifies as kb_question; KB extractor cites
        // the non-tool `change-wifi-password` entry.
        let harness = TestChatHarness()
        await harness
            .whenModeIs(.kbQuestion, matching: "wifi password")
            .whenKBCitation(
                entryID: "change-wifi-password",
                passage: "Open the Network panel and set a new password.",
                matching: "wifi password"
            )
            .whenChatPrompt(contains: "Reference: Change Wi-Fi Password",
                            returns: "Open Network, select your Wi-Fi, tap Password, save.")

        await harness.send("how do I change my wifi password")

        guard let reply = harness.lastAssistantMessage else {
            return XCTFail("no assistant reply")
        }
        XCTAssertEqual(reply.routing?.path, .answerWithRAG)
        XCTAssertNotNil(reply.sourceEntry, "read-more chip needs sourceEntry")
        XCTAssertEqual(reply.sourceEntry?.id, "change-wifi-password")
        XCTAssertTrue(reply.text.contains("Network"))
        XCTAssertEqual(reply.trace?.surface, .onDeviceRAG)
        XCTAssertEqual(reply.trace?.chatMode, .kbQuestion)
        // Egress: nothing left the device.
        XCTAssertGreaterThan(harness.tokenLedger.messagesOnDevice, 0)
        XCTAssertEqual(harness.tokenLedger.messagesCloudEscalated, 0)
        await harness.assertAllPromptsMatched()
    }

    func test_kbQuestionPath_fastGroundedPath_usesKBAnswerWithoutChatGeneration() async {
        let harness = TestChatHarness(useFastGroundedQA: true)
        await harness
            .whenModeIs(.kbQuestion, matching: "wifi password")
            .whenKBCitation(
                entryID: "change-wifi-password",
                passage: "Open the Network panel and set a new password.",
                matching: "wifi password"
            )

        await harness.send("how do I change my wifi password")

        guard let reply = harness.lastAssistantMessage else {
            return XCTFail("no assistant reply")
        }
        XCTAssertEqual(reply.routing?.path, .answerWithRAG)
        XCTAssertEqual(reply.sourceEntry?.id, "change-wifi-password")
        XCTAssertTrue(reply.text.contains("To change your Wi-Fi password"))
        XCTAssertTrue(reply.text.contains("Select the \"Network\" tile"))
        XCTAssertEqual(reply.trace?.surface, .onDeviceRAG)
        XCTAssertEqual(reply.trace?.inferenceMS, 0)
        let prompts = await harness.backend.recordedPrompts
        XCTAssertEqual(prompts.count, 0)
        await harness.assertAllPromptsMatched()
    }

    func test_decisionEngine_usesChatModeRouterForModeArbitration() async {
        let decision = TelcoRoutingDecision.toolAction(
            modePrediction: ChatModePrediction(
                mode: .toolAction,
                confidence: 0.91,
                reasoning: "ADR-015 lane=local_tool intent=troubleshooting tool=run_diagnostics",
                runtimeMS: 12
            ),
            selection: ToolSelection(
                intent: .runDiagnostics,
                confidence: 0.88,
                arguments: .empty,
                reasoning: "ADR-015 required_tool=run_diagnostics",
                runtimeMS: 0
            )
        )
        let engine = StaticTelcoDecisionEngine(
            result: TelcoDecisionResult(decision: decision, vector: nil, lane: nil)
        )
        let harness = TestChatHarness(useFastGroundedQA: true, decisionEngine: engine)
        await harness
            .whenModeIs(.kbQuestion, matching: "wifi slow")
            .whenKBCitation(
                entryID: "internet-slow-troubleshoot",
                passage: "If your internet feels slow, try these steps in order.",
                matching: "wifi slow"
            )

        await harness.send("why is my wifi slow")

        guard let reply = harness.lastAssistantMessage else {
            return XCTFail("no assistant reply")
        }
        XCTAssertEqual(reply.routing?.path, .answerWithRAG)
        XCTAssertNil(reply.toolDecision)
        XCTAssertEqual(reply.sourceEntry?.id, "internet-slow-troubleshoot")
        XCTAssertEqual(reply.trace?.chatMode, .kbQuestion)
        await harness.assertAllPromptsMatched()
    }

    func test_decisionEngine_cloudAssistLanePreemptsBadPersonalSummaryMode() async {
        let vector = TelcoDecisionVector(
            inputHash: 77,
            mode: .unavailable,
            kbEntry: .unavailable,
            tool: .unavailable,
            supportIntent: TelcoHeadResult(label: "outage", confidence: 1.0),
            issueComplexity: TelcoHeadResult(label: "backend_required", confidence: 0.98),
            routingLane: TelcoHeadResult(label: "local_answer", confidence: 0.99),
            cloudRequirements: .unavailable,
            requiredTool: TelcoHeadResult(label: "no_tool", confidence: 0.86),
            customerEscalationRisk: TelcoHeadResult(label: "frustrated", confidence: 0.82),
            piiRisk: TelcoHeadResult(label: "safe", confidence: 0.95),
            transcriptQuality: TelcoHeadResult(label: "clean", confidence: 0.99),
            slotCompleteness: .unavailable,
            inferenceMode: .sharedAdapter,
            forwardPassMs: 200,
            headProjectionMs: 2
        )
        let lane = TelcoLaneRouter.route(vector)
        let engine = StaticTelcoDecisionEngine(
            result: TelcoDecisionResult(
                decision: .personalSummary(modePrediction: ChatModePrediction(
                    mode: .personalSummary,
                    confidence: 1.0,
                    reasoning: "bad downstream mode head",
                    runtimeMS: 194
                )),
                vector: vector,
                lane: lane
            )
        )
        let harness = TestChatHarness(decisionEngine: engine)

        await harness.send("Is there an outage in my area?")

        guard let reply = harness.lastAssistantMessage else {
            return XCTFail("no assistant reply")
        }
        XCTAssertEqual(reply.routing?.path, .cloudAssist)
        XCTAssertTrue(reply.text.contains("live outage status"))
        XCTAssertTrue(reply.text.contains("Nothing has been sent"))
        XCTAssertEqual(reply.trace?.telcoPipeline?.target, "Cloud assist")
        XCTAssertEqual(reply.trace?.telcoPipeline?.downstreamStep?.id, "cloud_payload")
    }

    func test_decisionEngine_toolActionModeUsesLFMToolSelector() async {
        let decision = TelcoRoutingDecision.kbQuestion(
            modePrediction: ChatModePrediction(
                mode: .kbQuestion,
                confidence: 0.91,
                reasoning: "ADR-015 lane=local_answer intent=troubleshooting tool=no_tool",
                runtimeMS: 12
            ),
            citation: KBCitation.noMatch(runtimeMS: 0)
        )
        let engine = StaticTelcoDecisionEngine(
            result: TelcoDecisionResult(decision: decision, vector: nil, lane: nil)
        )
        let harness = TestChatHarness(decisionEngine: engine)
        await harness
            .whenModeIs(.toolAction, matching: "diagnostics")
            .whenToolPrompt(returns: """
                {"tool_id": "run-diagnostics",
                 "arguments": {},
                 "reasoning": "Customer asked to run diagnostics",
                 "requires_confirmation": false, "confidence": 0.91}
                """)

        await harness.send("run diagnostics on my home network")

        guard let decision = harness.lastToolDecision else {
            return XCTFail("expected toolDecision on the assistant message")
        }
        XCTAssertEqual(decision.toolID, "run-diagnostics")
        XCTAssertEqual(harness.lastAssistantMessage?.routing?.path, .toolCall)
        XCTAssertEqual(harness.lastAssistantMessage?.trace?.chatMode, .toolAction)
        await harness.assertAllPromptsMatched()
    }

    // MARK: - Path 2: tool call + confirm executes tool

    func test_toolCallPath_rendersCardAndConfirmExecutesTool() async {
        let harness = TestChatHarness()
        await harness
            .whenModeIs(.toolAction, matching: "block my son")
            .whenToolPrompt(returns: """
                {"tool_id": "toggle-parental-controls",
                 "arguments": {"action": "pause_internet", "target_device": "son's tablet"},
                 "reasoning": "Customer wants to pause a specific device",
                 "requires_confirmation": true, "confidence": 0.88}
                """)
            .whenChatPrompt(contains: "One-sentence confirmation prompt:",
                            returns: "I'll pause Son's Tablet. Confirm?")
            .whenChatPrompt(contains: "Customer-facing summary:",
                            returns: "Son's Tablet is now paused.")

        await harness.send("block my son's tablet from the internet")

        // First assertion pass — the proposal card
        guard let decision = harness.lastToolDecision else {
            return XCTFail("expected toolDecision on the assistant message")
        }
        XCTAssertEqual(decision.toolID, "toggle-parental-controls")
        XCTAssertEqual(harness.lastAssistantMessage?.routing?.path, .toolCall)
        XCTAssertTrue(decision.arguments.contains(where: {
            $0.label == "Target Device" && $0.value == "son's tablet"
        }))

        // Before confirm: no state mutation on CustomerContext
        XCTAssertEqual(
            harness.customerContext.managedDevices
                .first(where: { $0.name == "Son's Tablet" })?.accessState,
            .unrestricted
        )

        // Confirm
        await harness.confirmLatestTool()

        // After confirm: the device is paused + a second assistant
        // bubble appeared with the LFM-composed summary
        XCTAssertEqual(
            harness.customerContext.managedDevices
                .first(where: { $0.name == "Son's Tablet" })?.accessState,
            .paused
        )
        guard let confirmation = harness.lastAssistantMessage else {
            return XCTFail("no confirmation message after confirmTool")
        }
        XCTAssertTrue(confirmation.text.contains("Son's Tablet"))
        XCTAssertNotEqual(confirmation.id, harness.vm.messages.dropLast().last?.id)
        await harness.assertAllPromptsMatched()
    }

    // MARK: - Path 3: personalized summary

    func test_personalSummaryPath_summarizesProfile() async {
        let harness = TestChatHarness()
        await harness
            .whenModeIs(.personalSummary, matching: "summarize my home")
            .whenChatPrompt(contains: "Plan: Fiber Gigabit Connection",
                            returns: "You're on Fiber Gigabit. The E3200 extender upstairs is unhealthy.")

        await harness.send("summarize my home network")

        guard let reply = harness.lastAssistantMessage else {
            return XCTFail("no assistant reply")
        }
        XCTAssertEqual(reply.routing?.path, .personalized)
        XCTAssertTrue(reply.text.contains("Fiber"))
        XCTAssertNil(reply.sourceEntry, "personalized path doesn't use RAG")
        XCTAssertNil(reply.toolDecision)
        XCTAssertEqual(reply.trace?.chatMode, .personalSummary)
        await harness.assertAllPromptsMatched()
    }

    func test_personalSummaryPath_answersSSIDFromCustomerContext() async {
        let harness = TestChatHarness()
        await harness
            .whenModeIs(.personalSummary, matching: "SSID")

        await harness.send("Can you tell me what's my SSID")

        guard let reply = harness.lastAssistantMessage else {
            return XCTFail("no assistant reply")
        }
        XCTAssertEqual(reply.routing?.path, .personalized)
        XCTAssertEqual(reply.trace?.chatMode, .personalSummary)
        XCTAssertTrue(reply.text.contains("Alex-Fiber-Home"))
        XCTAssertTrue(reply.text.contains("SSID"))
        XCTAssertFalse(reply.text.contains("18 devices"))
        await harness.assertAllPromptsMatched()
    }

    // MARK: - Path 4: privacy boundary (out of scope)

    func test_outOfScopePath_returnsPrivacyBoundaryMessage() async {
        let harness = TestChatHarness()
        await harness
            .whenModeIs(.outOfScope, matching: "weather", confidence: 0.22)
            .whenChatPrompt(contains: "ONE short, friendly sentence",
                            returns: "I only handle home internet support. Nothing from this query left your phone.")

        await harness.send("what is the weather in new york")

        guard let reply = harness.lastAssistantMessage else {
            return XCTFail("no assistant reply")
        }
        XCTAssertEqual(reply.routing?.path, .outOfScope)
        XCTAssertNil(reply.sourceEntry)
        XCTAssertEqual(reply.trace?.chatMode, .outOfScope)
        // The deterministic `TelcoTopicGate` is fully confident when
        // a query has no telco vocabulary — it's a closed-domain
        // lookup, not a learned probability. 1.0 is the right
        // contract here; the prior < 0.5 assertion encoded the
        // legacy "LLM not confident → outOfScope" path that the
        // first-principles redesign deleted.
        XCTAssertEqual(reply.trace?.chatModeConfidence, 1.0)
        // Zero bytes egressed — the whole point of this path.
        XCTAssertEqual(harness.tokenLedger.messagesCloudEscalated, 0)
        await harness.assertAllPromptsMatched()
    }

    // MARK: - Path 5: decline drops proposal without execution

    func test_declineTool_dropsProposalWithoutExecution() async {
        let harness = TestChatHarness()
        await harness
            .whenModeIs(.toolAction, matching: "block my son")
            .whenToolPrompt(returns: """
                {"tool_id": "toggle-parental-controls",
                 "arguments": {"action": "pause_internet", "target_device": "son's tablet"},
                 "reasoning": "pause the tablet",
                 "requires_confirmation": true, "confidence": 0.88}
                """)
            .whenChatPrompt(contains: "One-sentence confirmation prompt:",
                            returns: "I'll pause Son's Tablet. Confirm?")

        await harness.send("block my son's tablet from the internet")
        XCTAssertNotNil(harness.lastToolDecision)
        let messageCountBeforeDecline = harness.vm.messages.count

        harness.declineLatestTool()

        // No new assistant message, no state mutation
        XCTAssertEqual(harness.vm.messages.count, messageCountBeforeDecline)
        XCTAssertEqual(
            harness.customerContext.managedDevices
                .first(where: { $0.name == "Son's Tablet" })?.accessState,
            .unrestricted
        )
        // The decision was cleared so the card stops being actionable
        XCTAssertNil(harness.lastToolDecision)
        await harness.assertAllPromptsMatched()
    }

    // MARK: - Path 6: provider failure surfaces as inference error

    func test_providerFailure_surfacesInferenceError() async {
        let harness = TestChatHarness()
        await harness
            .whenModeIs(.kbQuestion, matching: "signal weak")
            .whenKBCitation(
                entryID: "weak-signal-upstairs",
                passage: "Reposition the extender for best coverage.",
                matching: "signal weak"
            )
        // No chat-prompt rule scripted — the base-model call will
        // return an empty string, which LFMChatProvider surfaces as
        // `LFMChatError.emptyResponse`. ChatViewModel.appendInferenceFailure
        // should render a short, labeled error bubble.

        await harness.send("why is my wifi signal weak upstairs")

        guard let reply = harness.lastAssistantMessage else {
            return XCTFail("no assistant reply")
        }
        XCTAssertTrue(
            reply.text.starts(with: "On-device inference error"),
            "expected labeled error, got: \(reply.text)"
        )
        // Mode is preserved — we know where the failure happened.
        XCTAssertEqual(reply.routing?.path, .answerWithRAG)
    }
}

private final class StaticTelcoDecisionEngine: TelcoDecisionEngine, @unchecked Sendable {
    private let result: TelcoDecisionResult

    init(result: TelcoDecisionResult) {
        self.result = result
    }

    func decide(
        query: String,
        kb: [KBEntry],
        extraction: ExtractionResult,
        availableTools: [Tool]
    ) async -> TelcoDecisionResult {
        result
    }
}
