import Foundation

/// Reboots a Wi-Fi extender or mesh node. This is the "avoid a truck roll"
/// self-service path: identify the failing extender from free-form text,
/// confirm the action, and recover coverage without an agent.
public final class RebootExtenderTool: Tool {
    public let id = "reboot-extender"
    public let displayName = "Reboot Extender"
    public let description = "Reboots a specific Wi-Fi extender or mesh node. Devices on that node will briefly disconnect."
    public let icon = "dot.radiowaves.up.forward"
    public let requiresConfirmation = true
    public let isDestructive = true
    public let intent: ToolIntent = .rebootExtender
    public let deepLink: DeepLink? = DeepLink(label: "Mesh Health", url: "telco://mesh-health")

    private let customerContext: CustomerContext

    public init(customerContext: CustomerContext) {
        self.customerContext = customerContext
    }

    public func execute(arguments: ToolArguments) async throws -> ToolResult {
        let start = Date()
        try? await Task.sleep(nanoseconds: 1_200_000_000)

        let requestedName = arguments["extender_name"]
        guard let extender = await MainActor.run(body: {
            customerContext.markExtenderRebooted(requestedName: requestedName)
        }) else {
            return ToolResult(
                toolID: id,
                status: .failure,
                humanSummary: "I couldn't find an extender on this account to reboot.",
                latencyMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        let resolvedName = requestedName ?? extender.model
        return ToolResult(
            toolID: id,
            status: .success,
            humanSummary: "Extender reboot started for \(resolvedName). Coverage should stabilize in about 30 seconds.",
            structuredPayload: [
                "target_extender": resolvedName,
                "equipment_model": extender.model,
                "estimated_recovery_seconds": "30",
            ],
            latencyMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }
}
