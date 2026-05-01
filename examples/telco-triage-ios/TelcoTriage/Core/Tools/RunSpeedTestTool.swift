import Foundation

/// Run a router speed test. Non-destructive — runs without confirmation.
/// Alpha returns a realistic-looking result relative to the customer's
/// plan cap; in Phase 2 this calls the real speed-test endpoint.
public final class RunSpeedTestTool: Tool {
    public let id = "run-speed-test"
    public let displayName = "Run Speed Test"
    public let description = "Measures download and upload speeds at the router."
    public let icon = "speedometer"
    public let requiresConfirmation = false
    public let isDestructive = false
    public let intent: ToolIntent = .runSpeedTest
    public let deepLink: DeepLink? = DeepLink(label: "Speed Test", url: "telco://speed-test")

    public init() {}

    public func execute(arguments: ToolArguments) async throws -> ToolResult {
        let start = Date()
        try? await Task.sleep(nanoseconds: 3_500_000_000)  // typical speed test ~3–5s

        // Simulate: 82–95% of plan cap, typical for home fiber.
        let down = Int.random(in: 770...900)
        let up = Int.random(in: 720...870)
        let ping = Int.random(in: 4...12)

        return ToolResult(
            toolID: id,
            status: .success,
            humanSummary: "Speed test complete. Down: \(down) Mbps • Up: \(up) Mbps • Latency: \(ping) ms.",
            structuredPayload: [
                "down_mbps": String(down),
                "up_mbps": String(up),
                "ping_ms": String(ping),
            ],
            latencyMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }
}
