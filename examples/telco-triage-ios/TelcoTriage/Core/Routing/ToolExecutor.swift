import Foundation

/// Executes a confirmed `Tool` and produces an LFM-composed
/// confirmation message from the tool's structured output. Sits
/// between ChatViewModel and the Tool protocol so the chat flow
/// doesn't have to know about argument unpacking or prompt
/// construction for the follow-up summary.
///
/// Execution contract:
///  - Caller passes a `ToolDecision` (what the LFM selector already
///    produced) plus the concrete `Tool` instance resolved from
///    `ToolRegistry`.
///  - `execute(...)` runs `tool.execute(arguments:)` with the
///    extracted arguments, then calls `LFMChatProvider` with
///    `.toolConfirmation` mode to summarize the `ToolResult`.
///  - Returns the composed message text, the latency breakdown, and
///    the structured `ToolResult` so ChatViewModel can thread it
///    through the call trace.
public struct ToolExecutor: Sendable {
    public struct Outcome: Sendable {
        public let assistantText: String
        public let toolResult: ToolResult
        public let toolLatencyMS: Int
        public let summaryLatencyMS: Int
        public let inputTokens: Int
        public let outputTokens: Int
    }

    public enum ExecutorError: Error, LocalizedError {
        case toolExecutionFailed(underlying: Error)
        case summaryFailed(underlying: Error)

        public var errorDescription: String? {
            switch self {
            case .toolExecutionFailed(let u): return "Tool execution failed: \(u.localizedDescription)"
            case .summaryFailed(let u): return "On-device summary failed: \(u.localizedDescription)"
            }
        }
    }

    private let chatProvider: LFMChatProvider

    public init(chatProvider: LFMChatProvider) {
        self.chatProvider = chatProvider
    }

    public func execute(
        tool: Tool,
        decision: ToolDecision
    ) async throws -> Outcome {
        let toolStart = Date()
        let args = Self.argumentsDict(from: decision.arguments)

        let result: ToolResult
        do {
            result = try await tool.execute(arguments: ToolArguments(args))
        } catch {
            throw ExecutorError.toolExecutionFailed(underlying: error)
        }
        let toolMS = Int(Date().timeIntervalSince(toolStart) * 1000)

        let summaryStart = Date()
        let summary: ChatModelResponse
        do {
            summary = try await chatProvider.generate(
                query: "(tool confirmation)",
                mode: .toolConfirmation(tool: tool, result: result)
            )
        } catch {
            throw ExecutorError.summaryFailed(underlying: error)
        }
        let summaryMS = Int(Date().timeIntervalSince(summaryStart) * 1000)

        return Outcome(
            assistantText: summary.text,
            toolResult: result,
            toolLatencyMS: toolMS,
            summaryLatencyMS: summaryMS,
            inputTokens: summary.inputTokens,
            outputTokens: summary.outputTokens
        )
    }

    /// Flatten `[ToolDecisionArgument]` → `[String: String]`. The
    /// LFM tool selector emits args as a dict but ChatMessage renders
    /// them as a sorted array; we round-trip back to a dict here.
    private static func argumentsDict(from list: [ToolDecisionArgument]) -> [String: String] {
        var dict: [String: String] = [:]
        for arg in list {
            // `label` is the humanized form (e.g. "Target Device");
            // we need the original snake_case key the tool expects.
            // ChatViewModel.formatArguments is the sole producer, so
            // we reverse its transform here: lowercase + replace
            // spaces with underscores.
            let key = arg.label
                .lowercased()
                .replacingOccurrences(of: " ", with: "_")
            dict[key] = arg.value
        }
        return dict
    }
}
