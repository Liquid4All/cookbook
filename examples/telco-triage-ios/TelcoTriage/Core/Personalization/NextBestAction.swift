import Foundation

/// A personalization recommendation the assistant can surface. NBAs are the
/// bridge between on-device intelligence and bottom-line impact — every
/// card here maps to a number Telco finance will actually book: upsell
/// revenue, retention-driven churn avoidance, cost savings for the
/// customer that buys loyalty, or proactive support that keeps the
/// NPS number up.
///
/// Protocol-based so new actions register without changes to the engine,
/// the Plan view, or the chat. See `NextBestActionRegistry.default`.
public protocol NextBestAction: Sendable {
    var id: String { get }
    var category: NBACategory { get }
    var headline: String { get }
    var body: String { get }
    var icon: String { get }
    var acceptLabel: String { get }
    var declineLabel: String { get }
    /// One-line value tag rendered on the card, e.g. "+$12/mo" or
    /// "Save $96/yr" or "No action needed".
    var impactTag: String? { get }

    /// True if this NBA is relevant for the given customer right now.
    func isEligible(for profile: CustomerProfile) -> Bool

    /// Higher = more prominent. Engine sorts descending. Signals that
    /// affect ARPU more (upsell accepted, retention saved) score higher.
    func priorityScore(for profile: CustomerProfile) -> Double

    /// Optional: query keywords that make this NBA contextually relevant
    /// when attached to a chat reply. Nil means no chat attachment.
    var chatAttachmentKeywords: [String]? { get }
}

public enum NBACategory: String, Sendable, Codable, CaseIterable {
    case upsell
    case retention
    case planOptimize
    case boltOn
    case proactiveSupport

    public var displayName: String {
        switch self {
        case .upsell: return "Upsell"
        case .retention: return "Retention"
        case .planOptimize: return "Plan fit"
        case .boltOn: return "Add-on"
        case .proactiveSupport: return "Proactive"
        }
    }
}

public struct NBAOutcome: Sendable, Equatable {
    public enum Verdict: String, Sendable { case accepted, declined, snoozed }
    public let actionID: String
    public let verdict: Verdict
    public let recordedAt: Date

    public init(actionID: String, verdict: Verdict, recordedAt: Date = Date()) {
        self.actionID = actionID
        self.verdict = verdict
        self.recordedAt = recordedAt
    }
}
