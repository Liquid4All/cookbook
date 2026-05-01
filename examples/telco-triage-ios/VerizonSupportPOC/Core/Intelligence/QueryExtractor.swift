import Foundation

/// Pulls structured fields out of a raw customer query. This is the
/// **Telco Extraction** slot in the roadmap — a ~350M fine-tune trained
/// on the schema below. Until the fine-tune ships, `RegexQueryExtractor`
/// handles the common patterns.
///
/// Downstream consumers:
/// - Tool selector (to fill `ToolArguments` with extracted device/error)
/// - NBA engine (urgency + device type influence scoring)
/// - Chat UI (surfaces the extracted fields in the call trace)
public protocol QueryExtractor: Sendable {
    func extract(from query: String) -> ExtractionResult
}

/// Schema mirrors the fine-tune target described in the training plan.
public struct ExtractionResult: Sendable, Equatable {
    public let device: String?          // "Fiber Router G3100", "Stream TV", …
    public let errorCode: String?       // "blinking orange", "error 1302", …
    public let planName: String?        // "Fiber Gigabit", "500/500", …
    public let requestedAction: String? // "restart", "upgrade", "cancel", …
    public let targetDevice: String?    // "son's tablet", "living room tv", …
    public let locationHint: String?    // "upstairs", "basement", …
    public let requestedTime: String?   // "next week", "tomorrow morning", …
    public let urgency: Urgency
    public let runtimeMS: Int

    public enum Urgency: String, Sendable, Codable { case low, medium, high }

    public init(
        device: String? = nil,
        errorCode: String? = nil,
        planName: String? = nil,
        requestedAction: String? = nil,
        targetDevice: String? = nil,
        locationHint: String? = nil,
        requestedTime: String? = nil,
        urgency: Urgency = .low,
        runtimeMS: Int = 0
    ) {
        self.device = device
        self.errorCode = errorCode
        self.planName = planName
        self.requestedAction = requestedAction
        self.targetDevice = targetDevice
        self.locationHint = locationHint
        self.requestedTime = requestedTime
        self.urgency = urgency
        self.runtimeMS = runtimeMS
    }

    public static let empty = ExtractionResult()

    public var hasAnyField: Bool {
        device != nil
            || errorCode != nil
            || planName != nil
            || requestedAction != nil
            || targetDevice != nil
            || locationHint != nil
            || requestedTime != nil
    }

    public var compactDescription: String {
        var parts: [String] = []
        if let d = device { parts.append("device: \(d)") }
        if let e = errorCode { parts.append("error: \(e)") }
        if let p = planName { parts.append("plan: \(p)") }
        if let a = requestedAction { parts.append("action: \(a)") }
        if let targetDevice { parts.append("target: \(targetDevice)") }
        if let locationHint { parts.append("location: \(locationHint)") }
        if let requestedTime { parts.append("time: \(requestedTime)") }
        parts.append("urgency: \(urgency.rawValue)")
        return "{ \(parts.joined(separator: ", ")) }"
    }
}
