import Foundation

/// Check connection status across the customer's equipment. Read-only —
/// runs without confirmation. Surfaces the first unhealthy device in the
/// human summary so the assistant can proactively offer a fix.
public final class CheckConnectionTool: Tool {
    public let id = "check-connection"
    public let displayName = "Check Connection Status"
    public let description = "Pings each piece of equipment and reports health."
    public let icon = "wifi"
    public let requiresConfirmation = false
    public let isDestructive = false
    public let intent: ToolIntent = .checkConnection
    public let deepLink: DeepLink? = DeepLink(label: "Connections", url: "telco://tab-home")

    private let customerContext: CustomerContext

    public init(customerContext: CustomerContext) {
        self.customerContext = customerContext
    }

    public func execute(arguments: ToolArguments) async throws -> ToolResult {
        let start = Date()
        try? await Task.sleep(nanoseconds: 900_000_000)

        let equipment = await MainActor.run { customerContext.profile.equipment }
        let online = equipment.filter { $0.status == .online }
        let unhealthy = equipment.filter { $0.status == .unhealthy }
        let offline = equipment.filter { $0.status == .offline }

        let summary: String
        if offline.isEmpty && unhealthy.isEmpty {
            summary = "All \(online.count) devices online and healthy."
        } else if let bad = unhealthy.first ?? offline.first {
            summary = "\(online.count)/\(equipment.count) online. \(bad.model) is \(bad.status.rawValue) — want me to restart it?"
        } else {
            summary = "\(online.count)/\(equipment.count) online."
        }

        return ToolResult(
            toolID: id,
            status: .success,
            humanSummary: summary,
            structuredPayload: [
                "total": String(equipment.count),
                "online": String(online.count),
                "unhealthy": String(unhealthy.count),
                "offline": String(offline.count),
            ],
            latencyMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }
}
