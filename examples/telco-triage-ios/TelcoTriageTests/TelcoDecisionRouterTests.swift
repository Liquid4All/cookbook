import XCTest
@testable import TelcoTriage

final class TelcoDecisionRouterTests: XCTestCase {
    func test_kbQuestion_routesWithDeferredCitation() {
        // The router intentionally emits `KBCitation.noMatch` for the
        // kbQuestion path now. The chat dispatcher owns retrieval so
        // the misaligned `kbEntry` head no longer drives citations.
        let entry = KBEntry(
            id: "restart-router",
            topic: "Restart Router",
            aliases: [],
            category: "support",
            answer: "Restart from the equipment screen. Wait for the router to come back online.",
            deepLinks: [],
            tags: [],
            requiresToolExecution: false
        )
        let decision = TelcoDecisionRouter.route(
            vector(mode: "kb_question", kbEntry: "restart-router"),
            kb: [entry],
            availableTools: [],
            extraction: .empty
        )

        guard case .kbQuestion(let mode, let citation) = decision else {
            return XCTFail("expected KB decision")
        }
        XCTAssertEqual(mode.mode, .kbQuestion)
        XCTAssertEqual(citation.entryId, KBCitation.noMatchID)
    }

    @MainActor
    func test_toolAction_routesToToolSelectionWithExtractedArguments() {
        let context = CustomerContext()
        let tools = ToolRegistry.default(customerContext: context).all
        let extraction = ExtractionResult(
            requestedAction: "pause_internet",
            targetDevice: "son's tablet"
        )

        let decision = TelcoDecisionRouter.route(
            vector(mode: "tool_action", tool: "toggle-parental-controls"),
            kb: [],
            availableTools: tools,
            extraction: extraction
        )

        guard case .toolAction(let mode, let selection) = decision else {
            return XCTFail("expected tool decision")
        }
        XCTAssertEqual(mode.mode, .toolAction)
        XCTAssertEqual(selection.intent, .toggleParentalControls)
        XCTAssertEqual(selection.arguments.values["target_device"], "son's tablet")
        XCTAssertEqual(selection.arguments.values["action"], "pause_internet")
        XCTAssertEqual(selection.runtimeMS, 0)
    }

    func test_personalSummary_routesWithoutKBOrTool() {
        let decision = TelcoDecisionRouter.route(
            vector(mode: "personal_summary"),
            kb: [],
            availableTools: [],
            extraction: .empty
        )

        guard case .personalSummary(let mode) = decision else {
            return XCTFail("expected personal summary")
        }
        XCTAssertEqual(mode.mode, .personalSummary)
    }

    func test_unknownModeFallsBackToOutOfScope() {
        let decision = TelcoDecisionRouter.route(
            vector(mode: "nonsense"),
            kb: [],
            availableTools: [],
            extraction: .empty
        )

        guard case .outOfScope(let mode) = decision else {
            return XCTFail("expected out of scope")
        }
        XCTAssertEqual(mode.mode, .outOfScope)
    }

    func test_modelModeArbiter_usesChatModeRouterForKbQuestion() {
        let candidate = TelcoRoutingDecision.toolAction(
            modePrediction: prediction(.toolAction),
            selection: ToolSelection(
                intent: .runDiagnostics,
                confidence: 0.88,
                reasoning: "ADR-015 required_tool=run_diagnostics",
                runtimeMS: 0
            )
        )

        let decision = TelcoModelModeArbiter.arbitrate(
            candidate: candidate,
            modePrediction: prediction(.kbQuestion)
        )

        guard case .kbQuestion(let mode, let citation) = decision else {
            return XCTFail("expected KB decision")
        }
        XCTAssertEqual(mode.mode, .kbQuestion)
        XCTAssertEqual(citation.entryId, KBCitation.noMatchID)
    }

    func test_modelModeArbiter_toolActionDefersToToolSelector() {
        let candidate = TelcoRoutingDecision.kbQuestion(
            modePrediction: prediction(.kbQuestion),
            citation: .noMatch(runtimeMS: 0)
        )

        let decision = TelcoModelModeArbiter.arbitrate(
            candidate: candidate,
            modePrediction: prediction(.toolAction)
        )

        guard case .toolAction(let mode, let selection) = decision else {
            return XCTFail("expected tool decision")
        }
        XCTAssertEqual(mode.mode, .toolAction)
        XCTAssertNil(selection.intent)
    }

    func test_modelModeArbiter_preservesCandidateToolSelection() {
        let candidate = TelcoRoutingDecision.toolAction(
            modePrediction: prediction(.toolAction),
            selection: ToolSelection(
                intent: .runDiagnostics,
                confidence: 0.88,
                arguments: .empty,
                reasoning: "ADR-015 required_tool=run_diagnostics",
                runtimeMS: 0
            )
        )

        let decision = TelcoModelModeArbiter.arbitrate(
            candidate: candidate,
            modePrediction: prediction(.toolAction)
        )

        guard case .toolAction(let mode, let selection) = decision else {
            return XCTFail("expected tool decision")
        }
        XCTAssertEqual(mode.mode, .toolAction)
        XCTAssertEqual(selection.intent, .runDiagnostics)
        XCTAssertEqual(selection.runtimeMS, 0)
    }

    func test_modelModeArbiter_fallsBackWhenModeRouterFails() {
        let candidate = TelcoRoutingDecision.toolAction(
            modePrediction: prediction(.toolAction),
            selection: ToolSelection(
                intent: .runDiagnostics,
                confidence: 0.88,
                reasoning: "ADR-015 required_tool=run_diagnostics",
                runtimeMS: 0
            )
        )

        let decision = TelcoModelModeArbiter.arbitrate(
            candidate: candidate,
            modePrediction: ChatModePrediction(
                mode: .outOfScope,
                confidence: 0,
                reasoning: "unparseable model output",
                runtimeMS: 8
            )
        )

        guard case .toolAction(_, let selection) = decision else {
            return XCTFail("expected fallback candidate")
        }
        XCTAssertEqual(selection.intent, .runDiagnostics)
    }

    private func vector(
        mode: String,
        kbEntry: String = "none",
        tool: String = "none"
    ) -> TelcoDecisionVector {
        TelcoDecisionVector(
            inputHash: 42,
            mode: TelcoHeadResult(label: mode, confidence: 0.91),
            kbEntry: TelcoHeadResult(label: kbEntry, confidence: 0.88),
            tool: TelcoHeadResult(label: tool, confidence: 0.87),
            inferenceMode: .sharedAdapter,
            forwardPassMs: 44,
            headProjectionMs: 1
        )
    }

    private func prediction(_ mode: ChatMode) -> ChatModePrediction {
        ChatModePrediction(
            mode: mode,
            confidence: 0.91,
            reasoning: "test",
            runtimeMS: 12
        )
    }
}
