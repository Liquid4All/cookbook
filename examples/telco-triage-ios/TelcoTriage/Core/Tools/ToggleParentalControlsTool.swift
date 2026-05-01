import Foundation

/// Natural-language parental controls. Converts phrases like "block my son's
/// tablet" into a concrete account action and reflects the change in the
/// family-device UI immediately.
public final class ToggleParentalControlsTool: Tool {
    public let id = "toggle-parental-controls"
    public let displayName = "Parental Controls"
    public let description = "Pauses or restores internet access for a selected device."
    public let icon = "hand.raised.square.on.square"
    public let requiresConfirmation = true
    public let isDestructive = true
    public let intent: ToolIntent = .toggleParentalControls
    public let deepLink: DeepLink? = DeepLink(label: "Parental Controls", url: "telco://parental-controls")

    private let customerContext: CustomerContext

    public init(customerContext: CustomerContext) {
        self.customerContext = customerContext
    }

    public func execute(arguments: ToolArguments) async throws -> ToolResult {
        let start = Date()
        try? await Task.sleep(nanoseconds: 900_000_000)

        let targetDevice = arguments["target_device"] ?? "all devices"
        let action = arguments["action"] ?? "pause_internet"

        if targetDevice == "all devices" {
            return ToolResult(
                toolID: id,
                status: .failure,
                humanSummary: "I need a specific device name before I pause internet access.",
                latencyMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        guard let updatedDevice = await MainActor.run(body: {
            customerContext.updateParentalControls(deviceName: targetDevice, action: action)
        }) else {
            return ToolResult(
                toolID: id,
                status: .failure,
                humanSummary: "I couldn't find \(targetDevice) on this home network.",
                latencyMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        let actionLabel = action == "resume_internet" ? "restored" : "paused"
        return ToolResult(
            toolID: id,
            status: .success,
            humanSummary: "Internet access \(actionLabel) for \(updatedDevice.name).",
            structuredPayload: [
                "target_device": updatedDevice.name,
                "location": updatedDevice.location,
                "access_state": updatedDevice.accessState.rawValue,
            ],
            latencyMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }
}
