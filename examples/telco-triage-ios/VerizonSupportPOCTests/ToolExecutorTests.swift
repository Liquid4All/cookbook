import XCTest
@testable import VerizonSupportPOC

/// Tests for `ToolExecutor` — the glue between a confirmed
/// `ToolDecision` and a real `Tool.execute(...)` plus the follow-up
/// LFM confirmation summary.
///
/// The backend is stubbed so tests run in milliseconds without the
/// bundled GGUF.
@MainActor
final class ToolExecutorTests: XCTestCase {

    /// Records prompts + returns canned responses. Actor-based so
    /// strict concurrency sees the recorded state as properly isolated.
    private actor RecordingBackend: AdapterInferenceBackend {
        private(set) var lastPrompt: String = ""
        private var response: String
        private var shouldThrow: Bool

        init(response: String, shouldThrow: Bool = false) {
            self.response = response
            self.shouldThrow = shouldThrow
        }

        nonisolated func generate(
            messages: [AdapterChatMessage],
            adapterPath: String,
            maxTokens: Int,
            stopSequences: [String]
        ) async throws -> String {
            let flat = messages.map(\.content).joined(separator: "\n\n")
            return try await self.record(prompt: flat)
        }

        nonisolated func generate(
            prompt: String,
            adapterPath: String,
            maxTokens: Int,
            stopSequences: [String]
        ) async throws -> String {
            try await self.record(prompt: prompt)
        }

        private func record(prompt: String) throws -> String {
            self.lastPrompt = prompt
            if shouldThrow {
                throw NSError(domain: "RecordingBackend", code: 1, userInfo: nil)
            }
            return response
        }
    }

    /// Minimal Tool conformance for tests — lets us inject success,
    /// failure, and thrown-error outcomes without pulling in the real
    /// ToggleParentalControlsTool / RebootExtenderTool state
    /// mutations.
    ///
    /// `@unchecked Sendable` is acceptable here because the test
    /// harness only exercises FakeTool from `@MainActor`-isolated test
    /// methods, and mutation of `receivedArguments` happens on the
    /// main actor before any assertion reads it. If this pattern
    /// expands (e.g. multiple concurrent tools) convert to an actor.
    private final class FakeTool: Tool, @unchecked Sendable {
        let id: String
        let displayName: String
        let description: String
        let icon: String
        let requiresConfirmation: Bool
        let isDestructive: Bool
        let intent: ToolIntent
        let deepLink: DeepLink?

        enum Outcome {
            case success(ToolResult)
            case throwError(Error)
        }

        var outcome: Outcome
        private(set) var receivedArguments: [String: String] = [:]

        init(
            id: String = "reboot-extender",
            displayName: String = "Reboot Extender",
            intent: ToolIntent = .rebootExtender,
            outcome: Outcome
        ) {
            self.id = id
            self.displayName = displayName
            self.description = "fake"
            self.icon = "gear"
            self.requiresConfirmation = true
            self.isDestructive = true
            self.intent = intent
            self.deepLink = nil
            self.outcome = outcome
        }

        func execute(arguments: ToolArguments) async throws -> ToolResult {
            receivedArguments = arguments.values
            switch outcome {
            case .success(let r): return r
            case .throwError(let e): throw e
            }
        }
    }

    // MARK: - Successful execution

    func test_execute_forwardsMappedArgumentsAndSummarizesResult() async throws {
        let backend = RecordingBackend(response: "Extender rebooted.")
        let provider = LFMChatProvider(backend: backend)
        let tool = FakeTool(outcome: .success(ToolResult(
            toolID: "reboot-extender",
            status: .success,
            humanSummary: "Reboot started.",
            structuredPayload: ["target_extender": "upstairs"],
            latencyMS: 42
        )))
        let executor = ToolExecutor(chatProvider: provider)

        // ToolDecisionArgument labels go through ChatViewModel.formatArguments
        // which transforms snake_case → Space Case. ToolExecutor must
        // reverse that cleanly.
        let decision = ToolDecision(
            intent: .rebootExtender,
            toolID: tool.id,
            displayName: tool.displayName,
            icon: tool.icon,
            description: tool.description,
            arguments: [
                ToolDecisionArgument(label: "Extender Name", value: "upstairs"),
            ],
            confidence: 0.88,
            reasoning: nil,
            requiresConfirmation: true,
            isDestructive: true
        )

        let outcome = try await executor.execute(tool: tool, decision: decision)

        XCTAssertEqual(tool.receivedArguments["extender_name"], "upstairs")
        XCTAssertEqual(outcome.toolResult.status, .success)
        XCTAssertEqual(outcome.assistantText, "Extender rebooted.")
        XCTAssertGreaterThanOrEqual(outcome.summaryLatencyMS, 0)
        // The LFM confirmation prompt received the structured payload.
        let lastPrompt = await backend.lastPrompt
        XCTAssertTrue(lastPrompt.contains("target_extender=upstairs"))
        XCTAssertTrue(lastPrompt.contains("Status: success"))
    }

    // MARK: - Argument round-trip for multi-word + underscore-heavy keys

    func test_argumentsDict_roundTripsAllTrainedTelcoArgumentKeys() async throws {
        // Every argument key the 8 trained tools can emit must survive
        // the snake_case → displayed label → snake_case round-trip.
        // If someone adds a tool argument with CamelCase or unusual
        // capitalization, this test is the early-warning signal.
        let roundTripCases: [(snake: String, expected: String)] = [
            ("action", "action"),
            ("target_device", "target_device"),
            ("issue_summary", "issue_summary"),
            ("preferred_date", "preferred_date"),
            ("extender_name", "extender_name"),
        ]

        let backend = RecordingBackend(response: "ok")
        let provider = LFMChatProvider(backend: backend)
        let executor = ToolExecutor(chatProvider: provider)

        for (snake, expected) in roundTripCases {
            let tool = FakeTool(outcome: .success(ToolResult(
                toolID: "reboot-extender",
                status: .success,
                humanSummary: "ok",
                latencyMS: 1
            )))
            let args = ChatViewModel.formatArguments(ToolArguments([snake: "value"]))
            let decision = ToolDecision(
                intent: .rebootExtender,
                toolID: tool.id,
                displayName: tool.displayName,
                icon: tool.icon,
                description: tool.description,
                arguments: args,
                confidence: 0.9,
                reasoning: nil,
                requiresConfirmation: true,
                isDestructive: true
            )

            _ = try await executor.execute(tool: tool, decision: decision)

            XCTAssertEqual(
                tool.receivedArguments[expected],
                "value",
                "\(snake) did not round-trip cleanly to \(expected)"
            )
        }
    }

    // MARK: - Tool execution failure

    func test_execute_toolThrows_surfacesToolExecutionFailed() async {
        let backend = RecordingBackend(response: "ok")
        let provider = LFMChatProvider(backend: backend)
        let tool = FakeTool(outcome: .throwError(
            NSError(domain: "fake", code: 1, userInfo: nil)
        ))
        let executor = ToolExecutor(chatProvider: provider)
        let decision = ToolDecision(
            intent: .rebootExtender,
            toolID: tool.id,
            displayName: tool.displayName,
            icon: tool.icon,
            description: tool.description,
            arguments: [],
            confidence: 0.9,
            reasoning: nil,
            requiresConfirmation: true,
            isDestructive: true
        )

        do {
            _ = try await executor.execute(tool: tool, decision: decision)
            XCTFail("Expected ExecutorError.toolExecutionFailed")
        } catch let err as ToolExecutor.ExecutorError {
            if case .toolExecutionFailed = err { return }
            XCTFail("Wrong ExecutorError case: \(err)")
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    // MARK: - Summary LFM failure

    func test_execute_summaryBackendThrows_surfacesSummaryFailed() async {
        let backend = RecordingBackend(response: "", shouldThrow: true)
        let provider = LFMChatProvider(backend: backend)
        let tool = FakeTool(outcome: .success(ToolResult(
            toolID: "reboot-extender",
            status: .success,
            humanSummary: "ok",
            latencyMS: 1
        )))
        let executor = ToolExecutor(chatProvider: provider)
        let decision = ToolDecision(
            intent: .rebootExtender,
            toolID: tool.id,
            displayName: tool.displayName,
            icon: tool.icon,
            description: tool.description,
            arguments: [],
            confidence: 0.9,
            reasoning: nil,
            requiresConfirmation: true,
            isDestructive: true
        )

        do {
            _ = try await executor.execute(tool: tool, decision: decision)
            XCTFail("Expected ExecutorError.summaryFailed")
        } catch let err as ToolExecutor.ExecutorError {
            if case .summaryFailed = err { return }
            XCTFail("Wrong ExecutorError case: \(err)")
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }
}
