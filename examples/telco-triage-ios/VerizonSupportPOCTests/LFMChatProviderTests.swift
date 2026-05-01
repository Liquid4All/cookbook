import XCTest
@testable import VerizonSupportPOC

/// Tests for the on-device chat response generator. All tests stub the
/// `AdapterInferenceBackend` so they run in milliseconds without the
/// bundled GGUF.
@MainActor
final class LFMChatProviderTests: XCTestCase {

    /// Records the prompt it's handed and returns a canned response —
    /// lets every test assert both the prompt shape and the cleaning
    /// of the raw model output.
    ///
    /// Declared as an `actor` (not `@unchecked Sendable`) so strict
    /// concurrency is happy — reads of `lastPrompt` / `lastAdapterPath`
    /// from a test body go through the actor, no data races possible.
    private actor RecordingBackend: AdapterInferenceBackend {
        private(set) var lastPrompt: String = ""
        private(set) var lastAdapterPath: String = ""
        private(set) var lastStopSequences: [String] = []
        private var response: String
        private var shouldThrow: Bool

        init(response: String, shouldThrow: Bool = false) {
            self.response = response
            self.shouldThrow = shouldThrow
        }

        // `nonisolated` matches the Sendable protocol conformance shape
        // used by `ScriptedBackend` — keeps behavior consistent across
        // test stubs and stays correct under Swift strict concurrency.
        nonisolated func generate(
            messages: [AdapterChatMessage],
            adapterPath: String,
            maxTokens: Int,
            stopSequences: [String]
        ) async throws -> String {
            // Flatten to the existing substring-based assertions. The
            // LFMChatProvider tests run through the `prompt:` path so
            // this is only exercised by callers that adopt the
            // chat-template path (classifier, tool selector).
            let flat = messages.map(\.content).joined(separator: "\n\n")
            return try await self.record(prompt: flat, adapterPath: adapterPath, stops: stopSequences)
        }

        nonisolated func generate(
            prompt: String,
            adapterPath: String,
            maxTokens: Int,
            stopSequences: [String]
        ) async throws -> String {
            try await self.record(prompt: prompt, adapterPath: adapterPath, stops: stopSequences)
        }

        private func record(prompt: String, adapterPath: String, stops: [String]) throws -> String {
            self.lastPrompt = prompt
            self.lastAdapterPath = adapterPath
            self.lastStopSequences = stops
            if shouldThrow {
                throw NSError(domain: "RecordingBackend", code: 1, userInfo: nil)
            }
            return response
        }
    }

    // MARK: - Grounded QA

    func test_groundedQA_prompt_includesRetrievedEntryAndQuery() async throws {
        let backend = RecordingBackend(response: "Open Equipment → Restart router, then confirm.")
        let provider = LFMChatProvider(backend: backend)
        let entry = KBEntry(
            id: "restart-router",
            topic: "Restart Router",
            aliases: [],
            category: "Troubleshooting",
            answer: "Equipment → Restart router → Restart router.",
            deepLinks: [DeepLink(label: "Restart", url: "telco://restart-router")],
            tags: [],
            requiresToolExecution: true
        )
        let response = try await provider.generate(
            query: "how do I restart my router",
            mode: .groundedQA(topEntry: entry)
        )

        let lastPrompt = await backend.lastPrompt
        let lastAdapterPath = await backend.lastAdapterPath
        XCTAssertTrue(lastPrompt.contains("Customer: how do I restart my router"))
        XCTAssertTrue(lastPrompt.contains("Reference: Restart Router"))
        XCTAssertTrue(lastPrompt.contains("Equipment → Restart router"))
        // Base model, no adapter swap for chat generation.
        XCTAssertEqual(lastAdapterPath, "")
        XCTAssertEqual(response.text, "Open Equipment → Restart router, then confirm.")
        XCTAssertEqual(response.usedContextIDs, ["restart-router"])
        XCTAssertEqual(response.deepLinks.first?.url, "telco://restart-router")
    }

    // MARK: - Tool proposal

    func test_toolProposal_prompt_includesToolIdAndArguments() async throws {
        let backend = RecordingBackend(response: "I'll pause Son's Tablet. Confirm?")
        let provider = LFMChatProvider(backend: backend)
        let context = CustomerContext()
        let tool = ToggleParentalControlsTool(customerContext: context)
        let response = try await provider.generate(
            query: "block my son's tablet",
            mode: .toolProposal(tool: tool, arguments: [
                "action": "pause_internet",
                "target_device": "son's tablet",
            ])
        )

        let lastPrompt = await backend.lastPrompt
        XCTAssertTrue(lastPrompt.contains("Tool: toggle-parental-controls"))
        XCTAssertTrue(lastPrompt.contains("action=pause_internet"))
        XCTAssertTrue(lastPrompt.contains("target_device=son's tablet"))
        XCTAssertEqual(response.text, "I'll pause Son's Tablet. Confirm?")
    }

    // MARK: - Tool confirmation

    func test_toolConfirmation_prompt_includesStructuredPayload() async throws {
        let backend = RecordingBackend(response: "Son's Tablet is now paused.")
        let provider = LFMChatProvider(backend: backend)
        let context = CustomerContext()
        let tool = ToggleParentalControlsTool(customerContext: context)
        let result = ToolResult(
            toolID: tool.id,
            status: .success,
            humanSummary: "Internet access paused for Son's Tablet.",
            structuredPayload: ["target_device": "Son's Tablet", "access_state": "paused"],
            latencyMS: 12
        )
        let response = try await provider.generate(
            query: "(tool confirmation)",
            mode: .toolConfirmation(tool: tool, result: result)
        )

        let lastPrompt = await backend.lastPrompt
        XCTAssertTrue(lastPrompt.contains("Tool: toggle-parental-controls"))
        XCTAssertTrue(lastPrompt.contains("Status: success"))
        XCTAssertTrue(lastPrompt.contains("target_device=Son's Tablet"))
        XCTAssertEqual(response.text, "Son's Tablet is now paused.")
    }

    // MARK: - Profile summary

    /// The `.profileSummary` Mode is dead code as of build 16 —
    /// ChatViewModel.runPersonalizedSummary now always takes a
    /// deterministic path (personalSummaryResponse / billingResponse).
    /// The prompt is retained for F8's LFM-generated summarizer but
    /// no live call site uses it. The contract the test still enforces:
    /// the plan name reaches the prompt so a future F8 summarizer can
    /// ground on it.
    func test_profileSummary_prompt_includesPlan() async throws {
        let backend = RecordingBackend(response: "You're on Fiber Gigabit.")
        let provider = LFMChatProvider(backend: backend)
        let response = try await provider.generate(
            query: "summarize my home network",
            mode: .profileSummary(profile: CustomerProfile.demo)
        )

        let lastPrompt = await backend.lastPrompt
        XCTAssertTrue(lastPrompt.contains("Fiber Gigabit Connection"),
                      "profileSummary prompt must include plan name")
        XCTAssertFalse(response.text.isEmpty)
    }

    // MARK: - Out of scope

    func test_outOfScope_prompt_includesQueryAndDoesNotAttemptAnswer() async throws {
        let backend = RecordingBackend(response: "I only handle home internet support. Nothing from this query left your phone.")
        let provider = LFMChatProvider(backend: backend)
        let response = try await provider.generate(
            query: "what is the weather in new york",
            mode: .outOfScope(query: "what is the weather in new york")
        )

        let lastPrompt = await backend.lastPrompt
        XCTAssertTrue(lastPrompt.contains("Customer: what is the weather in new york"))
        XCTAssertTrue(lastPrompt.contains("ONE short, friendly sentence"))
        XCTAssertTrue(response.text.contains("on-device") || response.text.contains("home internet"))
    }

    // MARK: - Stop sequences
    //
    // Regression for session-048 screenshot: "how do I restart my
    // router" was replied to correctly, then the 350M base model
    // emitted "Reply:\n\nReply:\n\nReply:\n\nReply:…" as trailing
    // garbage because the backend ran to `maxTokens` with no stop
    // markers. The chat provider must pipe its curated stops through
    // to the backend on every call.

    func test_generate_passesChatStopSequencesToBackend_forGroundedQA() async throws {
        let backend = RecordingBackend(response: "Open Equipment → Restart router, then confirm.")
        let provider = LFMChatProvider(backend: backend)
        let entry = KBEntry(
            id: "restart-router",
            topic: "Restart Router",
            aliases: [],
            category: "Troubleshooting",
            answer: "Steps.",
            deepLinks: [],
            tags: [],
            requiresToolExecution: true
        )
        _ = try await provider.generate(
            query: "how do I restart my router",
            mode: .groundedQA(topEntry: entry)
        )
        let stops = await backend.lastStopSequences
        XCTAssertTrue(stops.contains("\nReply:"), "Missing '\\nReply:' stop — this is the exact sequence the 350M base model emits as fake-turn garbage after the answer")
        XCTAssertTrue(stops.contains("\nCustomer:"))
        XCTAssertTrue(stops.contains("\nUser:"))
        XCTAssertTrue(stops.contains("\nAssistant:"))
        XCTAssertTrue(stops.contains("\nQuery:"))
    }

    func test_generate_passesChatStopSequencesToBackend_forProfileSummary() async throws {
        let backend = RecordingBackend(response: "You're on Fiber Gigabit; E3200 extender is unhealthy.")
        let provider = LFMChatProvider(backend: backend)
        _ = try await provider.generate(
            query: "summarize my home network",
            mode: .profileSummary(profile: CustomerProfile.demo)
        )
        let stops = await backend.lastStopSequences
        XCTAssertFalse(stops.isEmpty, "Chat calls must pass stop sequences to prevent runaway generation")
        XCTAssertTrue(stops.contains("\nCustomer:"))
    }

    func test_generate_passesChatStopSequencesToBackend_forOutOfScope() async throws {
        let backend = RecordingBackend(response: "I only handle home internet support.")
        let provider = LFMChatProvider(backend: backend)
        _ = try await provider.generate(
            query: "what is the weather",
            mode: .outOfScope(query: "what is the weather")
        )
        let stops = await backend.lastStopSequences
        XCTAssertTrue(stops.contains("\nReply:"))
        XCTAssertTrue(stops.contains("\nCustomer:"))
    }

    // MARK: - N-gram loop truncation
    //
    // Regression for session-048 screenshot 2: "summarize my home
    // network" locked into "…the router is connected to the router
    // G3100 and the router is connected to the router G3100…" on
    // greedy decoding. Stop sequences don't catch this because the
    // loop is within the answer body (no "\nCustomer:" boundary).
    // Post-generation n-gram truncation is the backstop.

    func test_truncateAtRepeatedNgram_truncatesRouterG3100Loop() {
        // Condensed version of the session-048 screenshot 2 output.
        let collapsed = """
        Alex Rivera's home network is in good health. The Fiber Gigabit \
        connection is online and the router is functioning. The router \
        is connected to the router G3100 and the router is connected to \
        the router G3100 and the router is connected to the router G3100.
        """
        let cleaned = LFMChatProvider.truncateAtRepeatedNgram(collapsed, ngramSize: 6)
        XCTAssertTrue(cleaned.hasPrefix("Alex Rivera's home network is in good health."))
        XCTAssertFalse(
            cleaned.contains("connected to the router G3100 and the router is connected"),
            "Second occurrence of the repeating span should have been stripped; got: \(cleaned)"
        )
    }

    func test_truncateAtRepeatedNgram_leavesCleanTextAlone() {
        let clean = "Open Equipment, choose Restart router, and confirm. That will reboot the Fiber G3100 in under a minute."
        XCTAssertEqual(LFMChatProvider.truncateAtRepeatedNgram(clean, ngramSize: 6), clean)
    }

    func test_truncateAtRepeatedNgram_shortTextIsUnchanged() {
        let short = "Pause Son's Tablet now."
        XCTAssertEqual(LFMChatProvider.truncateAtRepeatedNgram(short, ngramSize: 6), short)
    }

    // MARK: - truncateAtRepeatedSentenceStart (new in build 19)

    /// The pattern the 6-gram guard couldn't catch: fixed 5-word
    /// sentence stem with a varying 6th word. Taken verbatim from the
    /// TestFlight build 17 screenshot.
    func test_truncateAtRepeatedSentenceStart_cutsTopRightButtonLoop() {
        let loop = """
        The Home page is the main landing page of the app. It opens automatically when you log in. The top-right button is Notifications. The top-left dropdown lets you switch between addresses. The top-right button is Troubleshoot. The top-right button is Speed Test. The top-right button is Restart Router.
        """
        let cleaned = LFMChatProvider.truncateAtRepeatedSentenceStart(loop)
        XCTAssertLessThan(cleaned.count, loop.count, "guard did not cut the loop")
        XCTAssertFalse(cleaned.contains("Speed Test"),
                       "guard kept the 3rd-occurrence sentence (should cut before it)")
        XCTAssertTrue(cleaned.contains("Notifications"),
                      "guard over-cut — first repeated sentence should remain")
    }

    /// Two occurrences of the same sentence-start are normal English
    /// structure (parallelism). Guard must not cut on threshold=2.
    func test_truncateAtRepeatedSentenceStart_preservesParallelism() {
        let parallel = "If your wifi is slow, restart the router. If your wifi stays slow, open a ticket."
        XCTAssertEqual(LFMChatProvider.truncateAtRepeatedSentenceStart(parallel), parallel)
    }

    /// Short interjections ("OK.", "Sure.") are below the stem-word
    /// threshold and must be ignored when counting sentences — they
    /// mustn't shift the cut point.
    func test_truncateAtRepeatedSentenceStart_ignoresShortInterjections() {
        let text = "Sure. OK. The router is online. The router is offline."
        // Only 2 qualifying sentences, below the 3-threshold. Unchanged.
        XCTAssertEqual(LFMChatProvider.truncateAtRepeatedSentenceStart(text), text)
    }

    // MARK: - isTerseGeneration (grounded-QA fallback gate)

    func test_isTerseGeneration_rejectsTwoWordEcho() {
        XCTAssertTrue(ChatViewModel.isTerseGeneration("Restart router."))
    }

    func test_isTerseGeneration_acceptsRealAnswer() {
        let real = "To restart your router, open Equipment then tap Restart router and confirm."
        XCTAssertFalse(ChatViewModel.isTerseGeneration(real))
    }

    func test_isTerseGeneration_acceptsShortButMeaningful() {
        // 6 words / >40 chars — passes the threshold because the char
        // budget is met even though the word count is low.
        let short = "Your router rebooted; you're back online now."
        XCTAssertFalse(ChatViewModel.isTerseGeneration(short))
    }

    func test_cleanResponseText_stripsTrailingReplyMarker() {
        // Mirrors session-048 screenshot 5: the model ended its real
        // answer then started a fake "Reply:" turn. Stop sequences
        // are what stops decoding on device, but post-cleanup must
        // also strip the marker if it slipped through the check
        // window (multi-byte UTF-8 boundary).
        let withGarbage = """
        You can restart your router from Equipment → Restart router.

        Reply:

        Reply:
        """
        let cleaned = LFMChatProvider.cleanResponseText(withGarbage)
        XCTAssertEqual(cleaned, "You can restart your router from Equipment → Restart router.")
    }

    // MARK: - Text cleanup

    func test_cleanResponseText_stripsCodeFences() {
        let cleaned = LFMChatProvider.cleanResponseText("""
        ```
        Here is the answer.
        ```
        """)
        XCTAssertEqual(cleaned, "Here is the answer.")
    }

    func test_cleanResponseText_cutsAtFakeTurnMarker() {
        let cleaned = LFMChatProvider.cleanResponseText("Restart the router.\nCustomer: but what if it doesn't work?\nAssistant: …")
        XCTAssertEqual(cleaned, "Restart the router.")
    }

    // MARK: - Error paths

    func test_backendThrows_surfacesLFMChatError() async {
        let backend = RecordingBackend(response: "", shouldThrow: true)
        let provider = LFMChatProvider(backend: backend)
        do {
            _ = try await provider.generate(
                query: "x",
                mode: .outOfScope(query: "x")
            )
            XCTFail("Expected LFMChatError.backendFailed")
        } catch let err as LFMChatError {
            if case .backendFailed = err { return }
            XCTFail("Wrong LFMChatError case: \(err)")
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    func test_emptyResponse_surfacesEmptyResponseError() async {
        let backend = RecordingBackend(response: "   \n  ")
        let provider = LFMChatProvider(backend: backend)
        do {
            _ = try await provider.generate(
                query: "x",
                mode: .outOfScope(query: "x")
            )
            XCTFail("Expected LFMChatError.emptyResponse")
        } catch let err as LFMChatError {
            if case .emptyResponse = err { return }
            XCTFail("Wrong LFMChatError case: \(err)")
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }
}
