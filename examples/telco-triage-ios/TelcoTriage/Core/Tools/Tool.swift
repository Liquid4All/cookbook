import Foundation

/// A callable tool the assistant can invoke. Tools are the bridge between
/// conversational intent ("restart my router") and app action (calling the
/// router API). The registry is plugin-style: new tools register themselves
/// with zero changes to the chat or router code.
///
/// Execution contract:
///  - `requiresConfirmation == true` means the UI shows a confirm sheet
///    before `execute` runs. Use for anything destructive or user-visible.
///  - `execute` is async, returns a `ToolResult` carrying both a structured
///    payload (for the model to paraphrase) and a `humanSummary` (for the
///    chat bubble).
public protocol Tool: Sendable {
    var id: String { get }
    var displayName: String { get }
    var description: String { get }
    var icon: String { get }                   // SF Symbol
    var requiresConfirmation: Bool { get }
    var isDestructive: Bool { get }
    var intent: ToolIntent { get }
    var deepLink: DeepLink? { get }

    func execute(arguments: ToolArguments) async throws -> ToolResult
}

/// Loose key-value bag of call arguments — kept simple for the alpha since
/// the tools are parameterless. Extending to typed arguments is a
/// per-tool concern.
public struct ToolArguments: Sendable, Equatable {
    public let values: [String: String]
    public init(_ values: [String: String] = [:]) { self.values = values }
    public static let empty = ToolArguments()

    public subscript(key: String) -> String? { values[key] }

    public var isEmpty: Bool { values.isEmpty }
}

public struct ToolResult: Sendable {
    public enum Status: String, Sendable { case success, failure, cancelled }

    public let toolID: String
    public let status: Status
    public let humanSummary: String         // "Router restarted. Back online in ~45s."
    public let structuredPayload: [String: String]  // for the model if it wants to elaborate
    public let latencyMS: Int
    public let executedAt: Date

    public init(
        toolID: String,
        status: Status,
        humanSummary: String,
        structuredPayload: [String: String] = [:],
        latencyMS: Int,
        executedAt: Date = Date()
    ) {
        self.toolID = toolID
        self.status = status
        self.humanSummary = humanSummary
        self.structuredPayload = structuredPayload
        self.latencyMS = latencyMS
        self.executedAt = executedAt
    }
}

public enum ToolError: Error {
    case notFound(String)
    case executionFailed(String)
    case cancelled
}
