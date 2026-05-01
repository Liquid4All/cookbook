import Foundation

// MARK: - Plan optimization (downsell when overpaying / upsell when undersized)

/// Compares usage to plan cap. Flags the customer as a plan-fit case if
/// usage pattern suggests a cheaper plan would cover them (loyalty play)
/// OR a pricier plan would clearly pay for itself (upsell).
public struct PlanOptimizeNBA: NextBestAction, MonetaryNBA {
    public let id = "plan-optimize"
    public let category: NBACategory = .planOptimize
    public let icon = "chart.line.uptrend.xyaxis"
    public let acceptLabel = "See plan options"
    public let declineLabel = "Keep current plan"
    public let chatAttachmentKeywords: [String]? = ["bill", "plan", "price", "charge", "upgrade", "downgrade"]

    public init() {}

    // Headlining copy computed once; framing shifts based on usage fit.
    public var headline: String { "You're well-matched to a different plan" }
    public var body: String {
        "Your average speed use (~680 Mbps down) is close to your Gigabit plan cap. If you hit the cap often enough, an upgrade could actually save you cycles of slowdown. If not, the step-down 500/500 plan lines up with your actual use."
    }
    public var impactTag: String? { "Save up to $8/mo" }
    public var estimatedMonthlyDollars: Double? { 8.0 }

    public func isEligible(for profile: CustomerProfile) -> Bool {
        // Eligible when usage is meaningfully below cap (savings path) or
        // frequently at/over cap (upgrade path).
        let usage = profile.usage
        let fractionOfCap = Double(usage.avgDownMbps) / Double(max(profile.plan.downSpeedMbps, 1))
        return fractionOfCap < 0.85 || usage.billCyclesAtOrOverCap >= 2
    }

    public func priorityScore(for profile: CustomerProfile) -> Double {
        // Strong signal: speed test failures (frustration) or cycles at cap.
        let usage = profile.usage
        return 0.6 + Double(usage.speedTestFailures) * 0.05 + Double(usage.billCyclesAtOrOverCap) * 0.1
    }
}

// MARK: - Mesh Wi-Fi upsell (device count + peak growth)

public struct MeshUpgradeUpsellNBA: NextBestAction, MonetaryNBA {
    public let id = "mesh-upgrade"
    public let category: NBACategory = .upsell
    public let icon = "dot.radiowaves.up.forward"
    public let acceptLabel = "Add mesh Wi-Fi"
    public let declineLabel = "Not now"
    // "slow" intentionally NOT in the keywords: it overlaps with the
    // retention NBA's trigger, and slow != coverage — slow is usually a
    // speed-test / line-side issue that retention credit addresses.
    public let chatAttachmentKeywords: [String]? = ["signal", "weak", "extender", "mesh", "dead zone", "bedroom", "basement"]

    public init() {}

    public var headline: String { "Upgrade to Mesh Wi-Fi 6E" }
    public var body: String { "You've got 18 devices connected this month, peaking at 24. Your current extender is going unhealthy. Mesh Wi-Fi 6E adds two tri-band nodes and holds Gig speeds in every room." }
    public var impactTag: String? { "+$10/mo" }
    public var estimatedMonthlyDollars: Double? { 10.0 }

    public func isEligible(for profile: CustomerProfile) -> Bool {
        let hasExtenderIssue = profile.equipment.contains { $0.kind == .extender && $0.status != .online }
        return profile.usage.peakDeviceCount >= 15 || hasExtenderIssue
    }

    public func priorityScore(for profile: CustomerProfile) -> Double {
        let extenderBroken = profile.equipment.contains { $0.kind == .extender && $0.status != .online }
        return 0.75 + (extenderBroken ? 0.2 : 0.0) + Double(profile.usage.peakDeviceCount) * 0.005
    }
}

// MARK: - TravelPass bolt-on (behavioral trigger example)

public struct TravelPassBoltOnNBA: NextBestAction, MonetaryNBA {
    public let id = "travel-pass"
    public let category: NBACategory = .boltOn
    public let icon = "airplane"
    public let acceptLabel = "Add TravelPass"
    public let declineLabel = "Not now"
    public let chatAttachmentKeywords: [String]? = ["travel", "international", "roaming", "abroad", "trip"]

    public init() {}

    public var headline: String { "Add TravelPass before your next trip" }
    public var body: String { "TravelPass gives you your plan's data, talk, and text in 210+ countries for $12/day — only charged on days you use it abroad. You don't have it on your account yet." }
    public var impactTag: String? { "$12/day" }
    // TravelPass is charged per day abroad. For the ARPU roll-up we
    // amortize an industry-typical 5 travel-days/month expectation for
    // users who opt in. This keeps the "Surfaced $" headline dimensionally
    // consistent with the other monthly NBAs. Configurable constant —
    // tune to whatever a carrier's data team tells us the real number is.
    private static let expectedTravelDaysPerMonth: Double = 5
    public var estimatedMonthlyDollars: Double? { 12.0 * Self.expectedTravelDaysPerMonth }

    public func isEligible(for profile: CustomerProfile) -> Bool {
        !profile.usage.activeBoltOns.contains("TravelPass")
    }

    public func priorityScore(for profile: CustomerProfile) -> Double { 0.5 }
}

// MARK: - Retention credit (churn risk signal)

public struct SlowSpeedRetentionNBA: NextBestAction, MonetaryNBA {
    public let id = "slow-speed-retention"
    public let category: NBACategory = .retention
    public let icon = "hand.raised.circle"
    public let acceptLabel = "Apply credit"
    public let declineLabel = "Skip"
    public let chatAttachmentKeywords: [String]? = ["slow", "bad", "terrible", "complaint", "refund", "cancel"]

    public init() {}

    public var headline: String { "We've noticed the recent speed issues" }
    public var body: String { "Your speed tests have come in below plan three times this month. We'd like to credit your account $15 this cycle while our network team investigates the degradation at your address." }
    public var impactTag: String? { "−$15 one-time" }
    /// Negative: this is money the Telco spends to retain. Shown in
    /// parentheses in the ARPU tile.
    public var estimatedMonthlyDollars: Double? { -15.0 }

    public func isEligible(for profile: CustomerProfile) -> Bool {
        profile.usage.speedTestFailures >= 3 || profile.usage.outageMinutes >= 30
    }

    public func priorityScore(for profile: CustomerProfile) -> Double {
        // Retention is the highest-stakes NBA — boost it hard when triggered.
        0.9 + Double(profile.usage.speedTestFailures) * 0.02
    }
}

// MARK: - Proactive: unhealthy extender

public struct ExtenderProactiveSupportNBA: NextBestAction {
    public let id = "extender-proactive"
    public let category: NBACategory = .proactiveSupport
    public let icon = "exclamationmark.triangle"
    public let acceptLabel = "Troubleshoot now"
    public let declineLabel = "Dismiss"
    public let chatAttachmentKeywords: [String]? = nil   // Not a chat attachment — banner/tile only.

    public init() {}

    public var headline: String { "Your extender needs attention" }
    public var body: String { "The E3200 extender in your account has been reporting unhealthy for a few days. I can walk through a one-minute troubleshoot, or we can schedule a free swap if it keeps dropping." }
    public var impactTag: String? { "Prevent outage" }

    public func isEligible(for profile: CustomerProfile) -> Bool {
        profile.equipment.contains { $0.kind == .extender && $0.status == .unhealthy }
    }

    public func priorityScore(for profile: CustomerProfile) -> Double { 0.95 }
}
