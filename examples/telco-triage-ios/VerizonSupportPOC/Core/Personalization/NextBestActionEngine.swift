import Foundation
import Combine

/// Scores every registered NBA against the current customer profile,
/// filters to those eligible, and exposes the ranked list. Also tracks
/// outcomes (accept/decline) so the ARPU tile on Savings can quote a
/// running "revenue surfaced" number.
///
/// Keeping the engine deterministic + synchronous makes testing trivial.
/// Real production would run this as an async scoring pass triggered by
/// telemetry events (usage over cap, equipment degradation, etc.).
@MainActor
public final class NextBestActionEngine: ObservableObject {
    @Published public private(set) var topActions: [any NextBestAction] = []
    @Published public private(set) var outcomes: [NBAOutcome] = []

    private let registry: NextBestActionRegistry
    private let customerContext: CustomerContext
    private var observers: Set<AnyCancellable> = []

    public init(registry: NextBestActionRegistry, customerContext: CustomerContext) {
        self.registry = registry
        self.customerContext = customerContext
        refresh()

        // Re-score whenever the profile mutates (equipment state change,
        // plan change, a tool call that flips a flag, etc.).
        customerContext.$profile
            .sink { [weak self] _ in
                self?.refresh()
            }
            .store(in: &observers)
    }

    /// Re-run scoring against the current profile.
    public func refresh() {
        let profile = customerContext.profile
        topActions = registry.all
            .filter { $0.isEligible(for: profile) }
            .sorted { $0.priorityScore(for: profile) > $1.priorityScore(for: profile) }
    }

    /// Find the single best NBA whose `chatAttachmentKeywords` match the
    /// given query. Used by the chat layer to attach a contextual card
    /// after answering a user question.
    public func bestMatchForChat(query: String) -> (any NextBestAction)? {
        let q = query.lowercased()
        let profile = customerContext.profile
        return topActions.first { action in
            guard action.isEligible(for: profile),
                  let keywords = action.chatAttachmentKeywords else { return false }
            return keywords.contains(where: { q.contains($0) })
        }
    }

    public func record(outcome: NBAOutcome) {
        outcomes.append(outcome)
    }

    public func hasOutcome(for actionID: String) -> Bool {
        outcomes.contains { $0.actionID == actionID }
    }

    // MARK: - ARPU impact aggregation (for Savings tile)

    /// Gross monthly revenue the Telco could book if every positive-value
    /// NBA were accepted. Excludes retention credits (those are Telco
    /// cost, surfaced separately as `surfacedRetentionCostDollars`).
    public var surfacedMonthlyValueDollars: Double {
        topActions
            .compactMap { $0.estimatedMonthlyDollars }
            .filter { $0 > 0 }
            .reduce(0, +)
    }

    /// Retention credits the assistant is willing to offer (Telco cost,
    /// surfaced as absolute value). Shown as a paired stat with the
    /// revenue number so the ARPU card tells the whole story.
    public var surfacedRetentionCostDollars: Double {
        topActions
            .compactMap { $0.estimatedMonthlyDollars }
            .filter { $0 < 0 }
            .reduce(0, +)
            .magnitude
    }

    public var acceptedCount: Int {
        outcomes.filter { $0.verdict == .accepted }.count
    }

    /// Net monthly dollars from NBAs the customer actually accepted:
    /// revenue minus retention credits. Positive = net ARPU up.
    public var acceptedMonthlyValueDollars: Double {
        let acceptedIDs = Set(
            outcomes.filter { $0.verdict == .accepted }.map(\.actionID)
        )
        return registry.all
            .filter { acceptedIDs.contains($0.id) }
            .compactMap { $0.estimatedMonthlyDollars }
            .reduce(0, +)
    }
}

/// Plugin registry for NBAs. Same shape as `ToolRegistry` / `BrandRegistry`
/// — add a new concrete action to `default` and it lights up everywhere.
public final class NextBestActionRegistry: Sendable {
    public let all: [any NextBestAction]

    public init(actions: [any NextBestAction]) {
        self.all = actions
    }

    public static let `default`: NextBestActionRegistry = NextBestActionRegistry(actions: [
        PlanOptimizeNBA(),
        MeshUpgradeUpsellNBA(),
        TravelPassBoltOnNBA(),
        SlowSpeedRetentionNBA(),
        ExtenderProactiveSupportNBA(),
    ])
}

/// Most NBAs have a monetary value we can surface. The protocol doesn't
/// require it (proactive support might not), so we expose it as an
/// optional protocol extension.
public protocol MonetaryNBA {
    /// Signed: positive = revenue for the Telco; negative = credit / cost
    /// for the Telco (a retention win the Telco pays for). Use net-
    /// expected-value framing.
    var estimatedMonthlyDollars: Double? { get }
}

public extension NextBestAction {
    var estimatedMonthlyDollars: Double? {
        (self as? MonetaryNBA)?.estimatedMonthlyDollars
    }
}
