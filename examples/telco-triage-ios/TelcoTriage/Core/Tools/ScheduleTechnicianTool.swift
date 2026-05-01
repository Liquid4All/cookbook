import Foundation

/// Schedules a visit when the model determines the issue is unlikely to be
/// resolved with one more scripted troubleshooting loop.
public final class ScheduleTechnicianTool: Tool {
    public let id = "schedule-technician"
    public let displayName = "Schedule Technician"
    public let description = "Books an in-home technician visit for issues that need a field dispatch."
    public let icon = "calendar.badge.clock"
    public let requiresConfirmation = true
    public let isDestructive = true
    public let intent: ToolIntent = .scheduleTechnician
    public let deepLink: DeepLink? = DeepLink(label: "Visits", url: "telco://support-visits")

    private let customerContext: CustomerContext

    public init(customerContext: CustomerContext) {
        self.customerContext = customerContext
    }

    public func execute(arguments: ToolArguments) async throws -> ToolResult {
        let start = Date()
        try? await Task.sleep(nanoseconds: 1_000_000_000)

        let window = arguments["preferred_date"] ?? "next available"
        let issue = arguments["issue_summary"] ?? "Home network issue"

        await MainActor.run {
            customerContext.scheduleTechnician(
                windowLabel: window,
                note: issue
            )
        }

        return ToolResult(
            toolID: id,
            status: .success,
            humanSummary: "Technician visit scheduled for \(window). We'll send a reminder before arrival.",
            structuredPayload: [
                "window": window,
                "issue_summary": issue,
                "dispatch_type": "in-home network support",
            ],
            latencyMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }
}
