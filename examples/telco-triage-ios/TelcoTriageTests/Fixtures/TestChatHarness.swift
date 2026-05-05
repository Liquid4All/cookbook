import Foundation
import UIKit
import XCTest
@testable import TelcoTriage

/// Assembles a `ChatViewModel` with every dependency stubbed, so
/// integration tests can drive a query end-to-end and assert on the
/// resulting `messages` list + customer state without pulling in the
/// bundled GGUFs or the LFMEngine actor.
///
/// Fluent API for scripting backend responses lets each test read
/// top-to-bottom as a little scenario (setup → send → assert).
///
/// Usage:
///   let harness = TestChatHarness()
///   await harness
///       .whenModeIs(.kbQuestion, matching: "how do I restart")
///       .whenChatPrompt(contains: "Reference:", returns: "Open Equipment → Restart router.")
///       .send("how do I restart my router")
///   XCTAssertEqual(harness.lastAssistantMessage?.routing?.path, .answerWithRAG)
@MainActor
final class TestChatHarness {
    let vm: ChatViewModel
    let customerContext: CustomerContext
    let tokenLedger: TokenLedger
    let sessionStats: SessionStats
    let backend: ScriptedBackend
    let chatModeRouter: ScriptedChatModeRouter
    let kbExtractor: ScriptedKBExtractor

    /// Pulled out for assertion access — the harness owns it so tests
    /// can script rules + inspect recorded prompts.
    let knowledgeBase: KnowledgeBase

    init(
        kb: KnowledgeBase = .loadFromBundle(),
        profile: CustomerProfile = .demo,
        useFastGroundedQA: Bool = false,
        decisionEngine: (any TelcoDecisionEngine)? = nil
    ) {
        self.knowledgeBase = kb
        let pii = PIIAnalyzer()
        let backend = ScriptedBackend()
        self.backend = backend

        let provider = LFMChatProvider(backend: backend)
        let context = CustomerContext(profile: profile)
        self.customerContext = context
        let registry = ToolRegistry.default(customerContext: context)
        let ledger = TokenLedger()
        self.tokenLedger = ledger
        let stats = SessionStats()
        self.sessionStats = stats
        let nba = NextBestActionEngine(registry: .default, customerContext: context)
        let tool = LFMToolSelector(backend: backend, adapterPath: "/stub/tool.gguf")
        let executor = ToolExecutor(chatProvider: provider)

        // Default routing: everything routes to `.kbQuestion` with a
        // `.noMatch` citation unless the test scripts otherwise.
        // That matches the Phase A.2 "lean on the KB by default"
        // posture and keeps tests reading top-to-bottom.
        let modeRouter = ScriptedChatModeRouter(
            fallback: ChatModePrediction(
                mode: .kbQuestion,
                confidence: 1.0,
                reasoning: "test fallback",
                runtimeMS: 0
            )
        )
        let kbExtractor = ScriptedKBExtractor()
        self.chatModeRouter = modeRouter
        self.kbExtractor = kbExtractor

        self.vm = ChatViewModel(
            decisionEngine: decisionEngine,
            chatModeRouter: modeRouter,
            kbExtractor: kbExtractor,
            provider: provider,
            piiAnalyzer: pii,
            kb: kb,
            tokenLedger: ledger,
            sessionStats: stats,
            toolRegistry: registry,
            visionAnalyzer: StubVisionAnalyzer(),
            customerContext: context,
            nbaEngine: nba,
            toolSelector: tool,
            toolExecutor: executor,
            useFastGroundedQA: useFastGroundedQA,
            welcomeGreetingProvider: { "Welcome" }
        )
    }

    // MARK: - Fluent script API

    /// Script the chat-mode router to return `mode` for any query
    /// containing the given substring. First matching rule wins. Use
    /// this instead of the legacy `whenIntentPrompt` — mode routing
    /// no longer goes through the scripted LFM backend.
    @discardableResult
    func whenModeIs(_ mode: ChatMode, matching substring: String, confidence: Double = 0.95) async -> Self {
        await chatModeRouter.script(.init(
            matches: substring,
            prediction: ChatModePrediction(
                mode: mode,
                confidence: confidence,
                reasoning: "test",
                runtimeMS: 0
            )
        ))
        return self
    }

    /// Script the KB extractor to cite the given entry id + passage
    /// for any query containing `substring`. First matching rule
    /// wins; unmatched queries return `.noMatch` by default.
    @discardableResult
    func whenKBCitation(
        entryID: String,
        passage: String,
        matching substring: String,
        confidence: Double = 0.9
    ) async -> Self {
        await kbExtractor.script(.init(
            matches: substring,
            citation: KBCitation(
                entryId: entryID,
                passage: passage,
                confidence: confidence,
                runtimeMS: 0
            )
        ))
        return self
    }

    /// Script a response for the tool selector (matches on
    /// `"Select the correct tool"`).
    @discardableResult
    func whenToolPrompt(returns json: String) async -> Self {
        await backend.script(.init(matches: "Select the correct tool", response: json))
        return self
    }

    /// Script a response for any chat-provider prompt matching the
    /// given substring. Use substrings that reliably differentiate the
    /// five prompt modes (`"Reference:"`, `"Tool: toggle-"`, etc.).
    @discardableResult
    func whenChatPrompt(contains substring: String, returns text: String) async -> Self {
        await backend.script(.init(matches: substring, response: text))
        return self
    }

    // MARK: - Drive

    /// Enqueue a query and wait for the pipeline to finish. Fails the
    /// test if processing doesn't converge within `timeout` seconds.
    func send(_ query: String, timeout: TimeInterval = 3.0) async {
        vm.inputText = query
        vm.send()
        await waitForIdle(timeout: timeout)
    }

    /// Confirm a tool decision on the latest assistant message that
    /// carries one. No-op (for the test's benefit) if none exists —
    /// callers should assert on `lastToolDecision` first.
    func confirmLatestTool(timeout: TimeInterval = 3.0) async {
        guard let target = vm.messages.last(where: { $0.toolDecision != nil }) else {
            return
        }
        vm.confirmTool(messageID: target.id)
        await waitForIdle(timeout: timeout)
    }

    func declineLatestTool() {
        guard let target = vm.messages.last(where: { $0.toolDecision != nil }) else {
            return
        }
        vm.declineTool(messageID: target.id)
    }

    // MARK: - Assertion helpers

    /// The most recent assistant message (skips the welcome seed on
    /// fresh harnesses by virtue of being "most recent after a send").
    var lastAssistantMessage: ChatMessage? {
        vm.messages.reversed().first { $0.role == .assistant }
    }

    var lastToolDecision: ToolDecision? {
        vm.messages.reversed().compactMap { $0.toolDecision }.first
    }

    /// Every assistant message after the welcome seed, in order.
    var assistantReplies: [ChatMessage] {
        Array(vm.messages.filter { $0.role == .assistant }.dropFirst())
    }

    /// Fail the calling test if any prompt the pipeline produced went
    /// unmatched by the scripted rules. Call at the end of a fully-
    /// scripted test so missing entries surface with a clear message
    /// instead of as an empty-response / inference-error bubble.
    /// Skip for tests that intentionally leave a path unscripted
    /// (e.g. the provider-failure regression).
    func assertAllPromptsMatched(
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        let unmatched = await backend.unmatchedPrompts
        if !unmatched.isEmpty {
            let summary = unmatched
                .map { String($0.prefix(80)) }
                .joined(separator: "\n  - ")
            XCTFail(
                "ScriptedBackend had \(unmatched.count) unmatched prompt(s):\n  - \(summary)",
                file: file,
                line: line
            )
        }
    }

    // MARK: - Private helpers

    /// Wait for the pipeline to finish. Two-phase loop:
    ///
    ///  1. **Spin-up**: the Task spawned in `vm.send()` hasn't run yet
    ///     when we return here, so `isProcessing` may still be false.
    ///     Yield until the task starts (or until a short budget elapses
    ///     — a genuinely-instant path is valid, e.g. early guards in
    ///     `send()`).
    ///  2. **Wait**: once `isProcessing` is observably true, poll at
    ///     5ms intervals until the Task finishes or `timeout` expires.
    ///
    /// Fails the test (`XCTFail`) rather than warn-and-continue on
    /// timeout, so stale-data assertions can't mask a hung pipeline.
    private func waitForIdle(
        timeout: TimeInterval,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        // Phase 1: let the spawned Task start. Budget 50ms — plenty for
        // a Task kickoff on any simulator, short enough that legitimate
        // instant returns don't hang the test.
        let spinupDeadline = Date().addingTimeInterval(0.05)
        while !vm.isProcessing && Date() < spinupDeadline {
            await Task.yield()
        }

        // Phase 2: wait for completion.
        let deadline = Date().addingTimeInterval(timeout)
        while vm.isProcessing && Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000) // 5ms
        }
        if vm.isProcessing {
            XCTFail(
                "ChatViewModel still processing after \(timeout)s",
                file: file,
                line: line
            )
        }
    }
}

// MARK: - Vision stub

/// Minimal vision stub so the harness can build a ChatViewModel
/// without pulling in the MockVisionAnalyzer pack plumbing. All
/// integration tests exercise the text path; vision is tested
/// separately in MockVisionAnalyzerTests.
private struct StubVisionAnalyzer: VisionAnalyzer {
    func analyze(image: UIImage, prompt: String) async throws -> VisionResult {
        throw NSError(domain: "StubVisionAnalyzer", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "vision path not exercised by TestChatHarness"
        ])
    }
}
