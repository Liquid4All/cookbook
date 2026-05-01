import Foundation

/// Deep home-network diagnostic. Simulates a higher-confidence local health
/// check that can explain whether the issue is router-side, extender-side,
/// or likely line-side before the user escalates.
public final class RunDiagnosticsTool: Tool {
    public let id = "run-diagnostics"
    public let displayName = "Run Diagnostics"
    public let description = "Runs a whole-home diagnostic: latency, packet loss, gateway reachability, and extender health."
    public let icon = "waveform.path.ecg.rectangle"
    public let requiresConfirmation = false
    public let isDestructive = false
    public let intent: ToolIntent = .runDiagnostics
    public let deepLink: DeepLink? = DeepLink(label: "Home Health", url: "telco://home-health")

    private let customerContext: CustomerContext

    public init(customerContext: CustomerContext) {
        self.customerContext = customerContext
    }

    public func execute(arguments: ToolArguments) async throws -> ToolResult {
        let start = Date()
        try? await Task.sleep(nanoseconds: 1_800_000_000)

        let scope = arguments["scope"] ?? "whole_home"
        let issueSummary = await MainActor.run { () -> String in
            let hasUnhealthyExtender = customerContext.profile.equipment.contains {
                $0.kind == .extender && $0.status != .online
            }
            let summary = hasUnhealthyExtender
                ? "Diagnostics found packet loss concentrated on the extender path."
                : "Diagnostics found no line-side failure; issue appears isolated to in-home Wi-Fi conditions."
            customerContext.appendRecentIssue(summary)
            return summary
        }

        return ToolResult(
            toolID: id,
            status: .success,
            humanSummary: issueSummary,
            structuredPayload: [
                "scope": scope,
                "gateway_latency_ms": "11",
                "packet_loss_percent": "2.1",
                "recommendation": "reboot extender",
            ],
            latencyMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }
}
