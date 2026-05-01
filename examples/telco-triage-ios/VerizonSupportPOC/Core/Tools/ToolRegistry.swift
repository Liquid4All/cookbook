import Foundation

/// Plugin-style registry. Lookup by id or by `ToolIntent` — the router
/// already classifies intent, so it hands off directly without caring
/// which concrete Tool satisfies the intent.
///
/// `set-downtime` is intentionally NOT registered. `SetDowntimeTool`
/// was removed from the repo because the fine-tuned tool-selector
/// adapter was never trained on time-bounded pauses; exposing it here
/// would cause the LFM to hallucinate a tool id it couldn't emit
/// reliably. See docs/FUTURE_SCOPE.md for the retrain plan.
public final class ToolRegistry: Sendable {
    private let tools: [Tool]

    public init(tools: [Tool]) {
        self.tools = tools
    }

    public func tool(for intent: ToolIntent) -> Tool? {
        tools.first { $0.intent == intent }
    }

    public func tool(id: String) -> Tool? {
        tools.first { $0.id == id }
    }

    public var all: [Tool] { tools }

    /// The production registry — every tool this POC ships.
    @MainActor
    public static func `default`(customerContext: CustomerContext) -> ToolRegistry {
        ToolRegistry(tools: [
            RestartRouterTool(customerContext: customerContext),
            RunSpeedTestTool(),
            CheckConnectionTool(customerContext: customerContext),
            EnableWPSTool(),
            RunDiagnosticsTool(customerContext: customerContext),
            ScheduleTechnicianTool(customerContext: customerContext),
            ToggleParentalControlsTool(customerContext: customerContext),
            RebootExtenderTool(customerContext: customerContext),
        ])
    }
}
