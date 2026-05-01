import Foundation

/// Put the router into WPS pairing mode. Destructive in the sense that it
/// exposes the network to any WPS-capable device for the next 120s, so
/// confirm first.
public final class EnableWPSTool: Tool {
    public let id = "enable-wps"
    public let displayName = "Start WPS Pairing"
    public let description = "Opens a 2-minute window where a WPS device can join Wi-Fi without the password."
    public let icon = "lock.open.rotation"
    public let requiresConfirmation = true
    public let isDestructive = true
    public let intent: ToolIntent = .wpsPair
    public let deepLink: DeepLink? = DeepLink(label: "WPS", url: "telco://equipment-wps")

    public init() {}

    public func execute(arguments: ToolArguments) async throws -> ToolResult {
        let start = Date()
        try? await Task.sleep(nanoseconds: 800_000_000)

        return ToolResult(
            toolID: id,
            status: .success,
            humanSummary: "WPS enabled on your router for the next 2 minutes. Press the WPS button on the device you're pairing.",
            structuredPayload: [
                "pairing_window_seconds": "120",
            ],
            latencyMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }
}
