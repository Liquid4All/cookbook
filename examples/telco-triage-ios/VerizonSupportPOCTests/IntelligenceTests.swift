import XCTest
@testable import VerizonSupportPOC

/// Coverage for the Intelligence-layer slots (QueryExtractor,
/// ToolSelector). The keyword/heuristic fallbacks have been removed
/// from the product — all production paths go through the LoRA-backed
/// LFM classifiers. These tests exercise the LFM adapters via a
/// stubbed `AdapterInferenceBackend`.
///
/// The `LFMChatModeRouter` and `LFMKBExtractor` primitives do not yet
/// have direct unit tests here — Phase A.4 will backfill that coverage
/// alongside a validation harness for the 4-way mode classification.
final class IntelligenceTests: XCTestCase {

    // MARK: - QueryExtractor

    func test_regexExtractor_pullsDeviceAndErrorFromLEDQuery() {
        let ext = RegexQueryExtractor()
        let r = ext.extract(from: "my G3100 is blinking orange since yesterday")
        XCTAssertEqual(r.device, "Fiber Router G3100")
        XCTAssertEqual(r.errorCode, "blinking orange")
    }

    func test_regexExtractor_pullsPlanNameFromUpgradeQuery() {
        let ext = RegexQueryExtractor()
        let r = ext.extract(from: "I'm on the More Premium plan and want to upgrade to gigabit")
        XCTAssertEqual(r.planName, "More Premium")
        XCTAssertEqual(r.requestedAction, "upgrade")
    }

    func test_regexExtractor_pullsTargetDeviceAndTimeHints() {
        let ext = RegexQueryExtractor()
        let r = ext.extract(from: "block my son's tablet from the internet and send a tech next week")
        XCTAssertEqual(r.targetDevice, "son's tablet")
        XCTAssertEqual(r.requestedTime, "next week")
    }

    func test_regexExtractor_detectsHighUrgency() {
        let ext = RegexQueryExtractor()
        let r = ext.extract(from: "emergency — nothing works, totally down")
        XCTAssertEqual(r.urgency, .high)
    }

    func test_regexExtractor_plainQuery_returnsEmpty() {
        let ext = RegexQueryExtractor()
        let r = ext.extract(from: "hello")
        XCTAssertFalse(r.hasAnyField)
        XCTAssertEqual(r.urgency, .low)
    }

    func test_lfmExtractorStub_returnsEmpty() {
        let stub = LFMQueryExtractor()
        XCTAssertFalse(stub.extract(from: "my router is broken").hasAnyField)
    }

    // MARK: - LFMToolSelector

    private struct StubBackend: AdapterInferenceBackend {
        let response: String
        let shouldThrow: Bool
        func generate(
            messages: [AdapterChatMessage],
            adapterPath: String,
            maxTokens: Int,
            stopSequences: [String]
        ) async throws -> String {
            if shouldThrow {
                throw NSError(domain: "StubBackend", code: 1, userInfo: nil)
            }
            return response
        }

        func generate(
            prompt: String,
            adapterPath: String,
            maxTokens: Int,
            stopSequences: [String]
        ) async throws -> String {
            if shouldThrow {
                throw NSError(domain: "StubBackend", code: 1, userInfo: nil)
            }
            return response
        }
    }

    @MainActor
    func test_lfmToolSelector_happyPath_returnsMappedIntent() async {
        let backend = StubBackend(
            response: """
            {"tool_id": "restart-router", "arguments": {}, "reasoning": "User asked to reboot router", "requires_confirmation": true, "confidence": 0.94}
            """,
            shouldThrow: false
        )
        let selector = LFMToolSelector(backend: backend, adapterPath: "/does/not/matter.gguf")
        let selection = await selector.select(
            query: "restart my router please",
            extraction: .empty,
            availableTools: []
        )
        XCTAssertEqual(selection.intent, .restartRouter)
        XCTAssertEqual(selection.confidence, 0.94, accuracy: 0.001)
        XCTAssertTrue(selection.reasoning.contains("reboot"))
    }

    @MainActor
    func test_lfmToolSelector_noneToolID_returnsNoneSelection() async {
        let backend = StubBackend(
            response: """
            {"tool_id": "none", "arguments": {}, "reasoning": "query is about billing", "requires_confirmation": false, "confidence": 0.4}
            """,
            shouldThrow: false
        )
        let selector = LFMToolSelector(backend: backend, adapterPath: "/x")
        let selection = await selector.select(
            query: "explain my bill",
            extraction: .empty,
            availableTools: []
        )
        XCTAssertNil(selection.intent)
        XCTAssertEqual(selection.confidence, 0.4, accuracy: 0.001)
    }

    @MainActor
    func test_lfmToolSelector_fencedJSON_parsesOut() async {
        let backend = StubBackend(
            response: """
            ```json
            {"tool_id": "reboot-extender", "arguments": {"extender_name": "living_room"}, "reasoning": "extender reboot", "requires_confirmation": true, "confidence": 0.88}
            ```
            """,
            shouldThrow: false
        )
        let selector = LFMToolSelector(backend: backend, adapterPath: "/x")
        let selection = await selector.select(
            query: "reboot the living room extender",
            extraction: .empty,
            availableTools: []
        )
        XCTAssertEqual(selection.intent, .rebootExtender)
        XCTAssertEqual(selection.arguments.values["extender_name"], "living_room")
    }

    @MainActor
    func test_lfmToolSelector_invalidJSON_returnsEmptyReasoningNotRawBytes() async {
        let backend = StubBackend(response: "sorry I can't help with that", shouldThrow: false)
        let selector = LFMToolSelector(backend: backend, adapterPath: "/x")
        let selection = await selector.select(
            query: "what",
            extraction: .empty,
            availableTools: []
        )
        XCTAssertNil(selection.intent)
        XCTAssertEqual(selection.confidence, 0.0)
        XCTAssertEqual(selection.reasoning, "")
    }

    @MainActor
    func test_lfmToolSelector_backendThrows_returnsEmptyReasoningNotErrorText() async {
        let backend = StubBackend(response: "", shouldThrow: true)
        let selector = LFMToolSelector(backend: backend, adapterPath: "/x")
        let selection = await selector.select(
            query: "restart",
            extraction: .empty,
            availableTools: []
        )
        XCTAssertNil(selection.intent)
        XCTAssertEqual(selection.reasoning, "")
    }

    @MainActor
    func test_lfmToolSelector_unknownToolID_returnsNoSelection() async {
        // Regression: if the model ever hallucinates a tool_id like
        // "set-downtime" (which the v2 adapter was never trained on)
        // the selector must fall through cleanly rather than crash.
        let backend = StubBackend(
            response: """
            {"tool_id": "set-downtime", "arguments": {}, "reasoning": "", "requires_confirmation": true, "confidence": 0.9}
            """,
            shouldThrow: false
        )
        let selector = LFMToolSelector(backend: backend, adapterPath: "/x")
        let selection = await selector.select(
            query: "pause my son's tablet until 7",
            extraction: .empty,
            availableTools: []
        )
        XCTAssertNil(selection.intent)
        XCTAssertEqual(selection.reasoning, "")
    }

    func test_lfmToolSelector_promptMatchesTrainingCatalog() {
        let prompt = LFMToolSelector.buildPrompt(query: "my wifi is dead")

        XCTAssertTrue(prompt.hasPrefix("Select the correct tool and fill parameters for this telco home internet customer query."))
        XCTAssertTrue(prompt.contains("Query: \"my wifi is dead\""))
        XCTAssertTrue(prompt.contains("Available tools:"))
        XCTAssertTrue(prompt.contains("If no tool matches the query, return tool_id \"none\"."))
        XCTAssertTrue(prompt.hasSuffix("JSON:"))

        // All 8 trained tool ids must appear. Critically, `set-downtime`
        // must NOT appear — the adapter was never trained on it.
        let expectedOrder = [
            "\"id\": \"restart-router\"",
            "\"id\": \"run-speed-test\"",
            "\"id\": \"check-connection\"",
            "\"id\": \"enable-wps\"",
            "\"id\": \"run-diagnostics\"",
            "\"id\": \"schedule-technician\"",
            "\"id\": \"toggle-parental-controls\"",
            "\"id\": \"reboot-extender\"",
        ]
        var cursor = prompt.startIndex
        for needle in expectedOrder {
            if let range = prompt.range(of: needle, range: cursor..<prompt.endIndex) {
                cursor = range.upperBound
            } else {
                XCTFail("Prompt missing or out of order: \(needle)")
            }
        }
        XCTAssertFalse(
            prompt.contains("\"id\": \"set-downtime\""),
            "set-downtime must not appear in the prompt catalog — the adapter is not trained on it"
        )
    }

    // MARK: - LFMChatModeRouter

    @MainActor
    func test_lfmChatModeRouter_happyPath_allFourModesRoundTrip() async {
        for mode in ChatMode.allCases {
            let backend = StubBackend(
                response: """
                {"mode": "\(mode.rawValue)", "confidence": 0.88, "reasoning": "test"}
                """,
                shouldThrow: false
            )
            let router = LFMChatModeRouter(backend: backend, adapterPath: "")
            let prediction = await router.classify(query: "anything")
            XCTAssertEqual(prediction.mode, mode, "\(mode.rawValue) did not round-trip")
            XCTAssertEqual(prediction.confidence, 0.88, accuracy: 0.001)
            XCTAssertEqual(prediction.reasoning, "test")
        }
    }

    @MainActor
    func test_lfmChatModeRouter_unknownModeString_fallsBackToOutOfScope() async {
        let backend = StubBackend(
            response: """
            {"mode": "not_a_real_mode", "confidence": 0.9, "reasoning": "x"}
            """,
            shouldThrow: false
        )
        let router = LFMChatModeRouter(backend: backend, adapterPath: "")
        let prediction = await router.classify(query: "x")
        // Parse fails → fallback. Confidence is the hardcoded 0.0
        // fallback, not the model's 0.9 — the fallback reasoning is
        // what the caller should surface.
        XCTAssertEqual(prediction.mode, .outOfScope)
        XCTAssertEqual(prediction.confidence, 0.0)
    }

    @MainActor
    func test_lfmChatModeRouter_fencedJSON_parsesOut() async {
        let backend = StubBackend(
            response: """
            ```json
            {"mode": "tool_action", "confidence": 0.77, "reasoning": "wants to restart"}
            ```
            """,
            shouldThrow: false
        )
        let router = LFMChatModeRouter(backend: backend, adapterPath: "")
        let prediction = await router.classify(query: "restart my router")
        XCTAssertEqual(prediction.mode, .toolAction)
        XCTAssertEqual(prediction.confidence, 0.77, accuracy: 0.001)
    }

    @MainActor
    func test_lfmChatModeRouter_invalidJSON_returnsOutOfScope() async {
        let backend = StubBackend(response: "not json at all", shouldThrow: false)
        let router = LFMChatModeRouter(backend: backend, adapterPath: "")
        let prediction = await router.classify(query: "x")
        XCTAssertEqual(prediction.mode, .outOfScope)
        XCTAssertEqual(prediction.confidence, 0.0)
    }

    @MainActor
    func test_lfmChatModeRouter_backendThrows_returnsOutOfScope() async {
        let backend = StubBackend(response: "", shouldThrow: true)
        let router = LFMChatModeRouter(backend: backend, adapterPath: "")
        let prediction = await router.classify(query: "x")
        XCTAssertEqual(prediction.mode, .outOfScope)
        XCTAssertEqual(prediction.confidence, 0.0)
    }

    func test_lfmChatModeRouter_promptContainsAllFourModes() {
        let prompt = LFMChatModeRouter.buildPrompt(query: "sample query")
        XCTAssertTrue(prompt.contains("kb_question"))
        XCTAssertTrue(prompt.contains("tool_action"))
        XCTAssertTrue(prompt.contains("personal_summary"))
        XCTAssertTrue(prompt.contains("out_of_scope"))
        XCTAssertTrue(prompt.contains("Query: \"sample query\""))
        XCTAssertTrue(prompt.hasSuffix("JSON:"))
    }

    // MARK: - LFMKBExtractor

    @MainActor
    func test_lfmKBExtractor_happyPath_returnsCitation() async {
        let backend = StubBackend(
            response: """
            {"entry_id": "restart-router", "passage": "Steps to restart your router.", "confidence": 0.91}
            """,
            shouldThrow: false
        )
        let extractor = LFMKBExtractor(backend: backend)
        let citation = await extractor.extract(query: "how do I restart", kb: KBFixtures.all)
        XCTAssertEqual(citation.entryId, "restart-router")
        XCTAssertEqual(citation.passage, "Steps to restart your router.")
        XCTAssertEqual(citation.confidence, 0.91, accuracy: 0.001)
        XCTAssertTrue(citation.isMatch)
    }

    @MainActor
    func test_lfmKBExtractor_noMatchSentinel_returnsNoMatch() async {
        let backend = StubBackend(
            response: """
            {"entry_id": "none", "passage": "", "confidence": 0.3}
            """,
            shouldThrow: false
        )
        let extractor = LFMKBExtractor(backend: backend)
        let citation = await extractor.extract(query: "what's the weather", kb: KBFixtures.all)
        XCTAssertEqual(citation.entryId, KBCitation.noMatchID)
        XCTAssertFalse(citation.isMatch)
        XCTAssertEqual(citation.passage, "")
        // Model-reported confidence is preserved on explicit noMatch so
        // the caller can distinguish "confident none" from "guessed none".
        XCTAssertEqual(citation.confidence, 0.3, accuracy: 0.001)
    }

    @MainActor
    func test_lfmKBExtractor_hallucinatedEntryID_returnsNoMatch() async {
        let backend = StubBackend(
            response: """
            {"entry_id": "not-a-real-kb-entry", "passage": "made up", "confidence": 0.95}
            """,
            shouldThrow: false
        )
        let extractor = LFMKBExtractor(backend: backend)
        let citation = await extractor.extract(query: "x", kb: KBFixtures.all)
        XCTAssertEqual(citation.entryId, KBCitation.noMatchID)
        XCTAssertFalse(citation.isMatch)
        // Defensive: hallucinated citations must not leak their fake
        // confidence into the trace row.
        XCTAssertEqual(citation.confidence, 0.0)
    }

    @MainActor
    func test_lfmKBExtractor_fencedJSON_parsesOut() async {
        let backend = StubBackend(
            response: """
            ```json
            {"entry_id": "change-wifi-password", "passage": "Steps to change your Wi-Fi password.", "confidence": 0.82}
            ```
            """,
            shouldThrow: false
        )
        let extractor = LFMKBExtractor(backend: backend)
        let citation = await extractor.extract(query: "change password", kb: KBFixtures.all)
        XCTAssertEqual(citation.entryId, "change-wifi-password")
        XCTAssertTrue(citation.isMatch)
    }

    @MainActor
    func test_lfmKBExtractor_invalidJSON_returnsNoMatch() async {
        let backend = StubBackend(response: "i dont know", shouldThrow: false)
        let extractor = LFMKBExtractor(backend: backend)
        let citation = await extractor.extract(query: "x", kb: KBFixtures.all)
        XCTAssertEqual(citation.entryId, KBCitation.noMatchID)
        XCTAssertFalse(citation.isMatch)
    }

    @MainActor
    func test_lfmKBExtractor_backendThrows_returnsNoMatch() async {
        let backend = StubBackend(response: "", shouldThrow: true)
        let extractor = LFMKBExtractor(backend: backend)
        let citation = await extractor.extract(query: "x", kb: KBFixtures.all)
        XCTAssertEqual(citation.entryId, KBCitation.noMatchID)
        XCTAssertFalse(citation.isMatch)
    }

    func test_lfmKBExtractor_promptIsCompactNoInlineKB() {
        // Phase C: prompt is ~40 tokens (no inline KB). The LoRA memorizes
        // the KB; the prompt just passes the query through.
        let prompt = LFMKBExtractor.buildPrompt(query: "sample")
        XCTAssertTrue(prompt.contains("Query: \"sample\""))
        XCTAssertTrue(prompt.contains("\"none\""),
                      "Prompt must mention the \"none\" sentinel so the model can decline")
        XCTAssertTrue(prompt.hasSuffix("JSON:"))
        // Must NOT contain inline KB entries — that's the whole point of Phase C
        XCTAssertFalse(prompt.contains("restart-router"),
                       "Phase C prompt must not inline KB entry IDs")
        // Short prompt: under 200 chars (was 3,200+ tokens with inline KB)
        XCTAssertLessThan(prompt.count, 300,
                          "Phase C prompt should be ~40 tokens, not thousands")
    }

    // MARK: - Taxonomy sanity

    func test_toolIntent_coversFineTuneToolCatalog() {
        // 8 tool ids — `set-downtime` intentionally excluded.
        let all = Set(ToolIntent.allCases.map(\.toolID))
        let expected: Set<String> = [
            "restart-router", "run-speed-test", "check-connection", "enable-wps",
            "run-diagnostics", "schedule-technician",
            "toggle-parental-controls", "reboot-extender",
        ]
        XCTAssertEqual(all, expected)
    }

    func test_toolIntent_parseHyphenatedID() {
        XCTAssertEqual(ToolIntent(toolID: "restart-router"), .restartRouter)
        XCTAssertEqual(ToolIntent(toolID: "reboot-extender"), .rebootExtender)
        XCTAssertEqual(ToolIntent(toolID: "enable-wps"), .wpsPair)
        XCTAssertNil(ToolIntent(toolID: "none"))
        XCTAssertNil(ToolIntent(toolID: "bogus-tool"))
        XCTAssertNil(ToolIntent(toolID: "set-downtime"), "set-downtime must not parse — adapter isn't trained on it")
    }
}
