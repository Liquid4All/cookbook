import Foundation

/// Restart the primary router. Destructive (disconnects every device on the
/// network) so it demands confirmation. In Phase 2 this would call the
/// carrier home internet API; the alpha simulates the 45-second router boot with
/// realistic latency and updates `CustomerContext.lastReboot` so the
/// Equipment view reflects it.
public final class RestartRouterTool: Tool {
    public let id = "restart-router"
    public let displayName = "Restart Router"
    public let description = "Reboots the primary router. All connected devices will briefly lose internet."
    public let icon = "arrow.clockwise.circle.fill"
    public let requiresConfirmation = true
    public let isDestructive = true
    public let intent: ToolIntent = .restartRouter
    public let deepLink: DeepLink? = DeepLink(label: "Restart Router", url: "telco://restart-router")

    private let customerContext: CustomerContext

    public init(customerContext: CustomerContext) {
        self.customerContext = customerContext
    }

    public func execute(arguments: ToolArguments) async throws -> ToolResult {
        let start = Date()
        // Alpha: simulate the restart. In Phase 2 this is a real API call
        // to the carrier router-control endpoint, guarded by the
        // app's customer auth.
        try? await Task.sleep(nanoseconds: 1_500_000_000)  // 1.5s "command accepted"

        guard let router = await MainActor.run(body: { customerContext.profile.equipment.first(where: { $0.kind == .router }) }) else {
            return ToolResult(
                toolID: id,
                status: .failure,
                humanSummary: "I couldn't find a primary router on your account to restart.",
                latencyMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        await MainActor.run {
            customerContext.markRouterRebooted(serial: router.serial)
        }

        return ToolResult(
            toolID: id,
            status: .success,
            humanSummary: "Router restart initiated on \(router.model). Devices will reconnect in about 45 seconds.",
            structuredPayload: [
                "equipment_serial": router.serial,
                "equipment_model": router.model,
                "estimated_recovery_seconds": "45",
            ],
            latencyMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }
}
