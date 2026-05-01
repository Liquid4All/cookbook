import Foundation
import Combine

/// Running tally of tokens the user did NOT send to a cloud LLM this session,
/// plus a dollar estimate. The number a carrier cares about most in the demo:
/// "here's what you saved vs. cloud-only."
///
/// Pricing assumption documented as a constant so it's easy to change when
/// If a carrier asks "what price are you quoting?" — default is GPT-4-class, which
/// is the honest comparison for a support assistant today.
@MainActor
public final class TokenLedger: ObservableObject {
    /// Pricing in dollars per 1M tokens.
    ///
    /// ⚙️ **Change pricing here.** This is the single source of truth.
    /// Two prices are tracked:
    ///  - `competitorCloud` — what a carrier would pay running this on a
    ///    generic cloud LLM. Used to estimate savings on on-device queries.
    ///  - `liquidCloud` — what a Phase-2+ Liquid cloud LFM escalation
    ///    actually costs. Used for the spent-in-cloud side of the ledger.
    ///  The net-savings story on the dashboard is
    ///  `saved_at_competitor_rates − spent_at_liquid_rates` — a true
    ///  apples-to-apples "what you would have paid vs what you actually
    ///  paid" number.
    public struct Pricing: Sendable {
        public let inputPerM: Double
        public let outputPerM: Double

        /// Representative GPT-4-class pricing — the honest comparison for
        /// a support assistant today. The customer would pay these rates
        /// if they used a generic cloud LLM instead of an on-device LFM.
        public static let competitorCloud: Pricing = .init(inputPerM: 5.0, outputPerM: 15.0)

        /// Liquid's own cloud-class inference (LFM2-24B-class). Used for
        /// the tokens-spent side when a query escalates. Meaningfully
        /// cheaper than frontier competitor pricing — part of the Liquid
        /// cost advantage story.
        public static let liquidCloud: Pricing = .init(inputPerM: 0.50, outputPerM: 1.50)

        /// Legacy alias — equivalent to `competitorCloud`. Kept for
        /// callers that haven't been migrated yet. Prefer the explicit
        /// names above.
        public static let gpt4: Pricing = .competitorCloud
    }

    @Published public private(set) var inputTokensSaved: Int = 0
    @Published public private(set) var outputTokensSaved: Int = 0
    @Published public private(set) var inputTokensSpentInCloud: Int = 0
    @Published public private(set) var outputTokensSpentInCloud: Int = 0
    @Published public private(set) var messagesOnDevice: Int = 0
    @Published public private(set) var messagesDeflected: Int = 0
    @Published public private(set) var messagesCloudEscalated: Int = 0

    /// Rate at which savings are quoted — what the customer would pay
    /// running a competitor cloud LLM.
    public var competitorPricing: Pricing = .competitorCloud

    /// Rate at which escalations actually bill — Liquid's own cloud LFM.
    public var liquidPricing: Pricing = .liquidCloud

    /// Back-compat alias for callers / tests still referring to `pricing`.
    public var pricing: Pricing {
        get { competitorPricing }
        set { competitorPricing = newValue }
    }

    public init() {}

    public var totalTokensSaved: Int {
        inputTokensSaved + outputTokensSaved
    }

    public var totalTokensSpentInCloud: Int {
        inputTokensSpentInCloud + outputTokensSpentInCloud
    }

    public var estimatedDollarsSaved: Double {
        let inputCost = Double(inputTokensSaved) / 1_000_000.0 * competitorPricing.inputPerM
        let outputCost = Double(outputTokensSaved) / 1_000_000.0 * competitorPricing.outputPerM
        return inputCost + outputCost
    }

    public var estimatedDollarsSpentInCloud: Double {
        let inputCost = Double(inputTokensSpentInCloud) / 1_000_000.0 * liquidPricing.inputPerM
        let outputCost = Double(outputTokensSpentInCloud) / 1_000_000.0 * liquidPricing.outputPerM
        return inputCost + outputCost
    }

    public var netDollarsSaved: Double {
        estimatedDollarsSaved - estimatedDollarsSpentInCloud
    }

    public var percentOnDevice: Double {
        let total = messagesOnDevice + messagesDeflected + messagesCloudEscalated
        guard total > 0 else { return 0 }
        return Double(messagesOnDevice + messagesDeflected) / Double(total) * 100.0
    }

    /// Record an on-device answer. These are tokens the user did not send to
    /// cloud — pure savings.
    public func recordOnDevice(inputTokens: Int, outputTokens: Int) {
        inputTokensSaved += inputTokens
        outputTokensSaved += outputTokens
        messagesOnDevice += 1
    }

    /// Record a deflection. The tokens for the query *would* have been sent
    /// to cloud in Phase 2, so they don't count as savings — but we track
    /// the message count to show the on-device/deflection mix honestly.
    public func recordDeflection() {
        messagesDeflected += 1
    }

    /// Record a cloud escalation. These tokens DID leave the device — they
    /// show up on the "spent in cloud" side of the ledger. Net savings is
    /// the delta between what we kept local and what we had to escalate.
    public func recordCloudEscalation(inputTokens: Int, outputTokens: Int) {
        inputTokensSpentInCloud += inputTokens
        outputTokensSpentInCloud += outputTokens
        messagesCloudEscalated += 1
    }

    public func reset() {
        inputTokensSaved = 0
        outputTokensSaved = 0
        inputTokensSpentInCloud = 0
        outputTokensSpentInCloud = 0
        messagesOnDevice = 0
        messagesDeflected = 0
        messagesCloudEscalated = 0
    }
}
