import XCTest
@testable import VerizonSupportPOC

@MainActor
final class TokenLedgerTests: XCTestCase {
    func test_recordOnDevice_accumulatesTokens() {
        let ledger = TokenLedger()
        ledger.recordOnDevice(inputTokens: 100, outputTokens: 50)
        ledger.recordOnDevice(inputTokens: 40, outputTokens: 20)

        XCTAssertEqual(ledger.inputTokensSaved, 140)
        XCTAssertEqual(ledger.outputTokensSaved, 70)
        XCTAssertEqual(ledger.totalTokensSaved, 210)
        XCTAssertEqual(ledger.messagesOnDevice, 2)
    }

    func test_estimatedDollarsSaved_usesGPT4Pricing() {
        let ledger = TokenLedger()
        ledger.recordOnDevice(inputTokens: 1_000_000, outputTokens: 1_000_000)
        XCTAssertEqual(ledger.estimatedDollarsSaved, 20.0, accuracy: 0.01)
    }

    func test_percentOnDevice_countsDeflectionsAsLocal() {
        // Deflections run tool calls on-device (no network). The only
        // thing that actually leaves the phone is `cloudEscalated`.
        let ledger = TokenLedger()
        ledger.recordOnDevice(inputTokens: 10, outputTokens: 10)
        ledger.recordOnDevice(inputTokens: 10, outputTokens: 10)
        ledger.recordOnDevice(inputTokens: 10, outputTokens: 10)
        ledger.recordDeflection()

        XCTAssertEqual(ledger.percentOnDevice, 100.0, accuracy: 0.1)
    }

    func test_percentOnDevice_mixesCloudEscalations() {
        let ledger = TokenLedger()
        ledger.recordOnDevice(inputTokens: 10, outputTokens: 10)
        ledger.recordOnDevice(inputTokens: 10, outputTokens: 10)
        ledger.recordOnDevice(inputTokens: 10, outputTokens: 10)
        ledger.recordCloudEscalation(inputTokens: 100, outputTokens: 200)

        XCTAssertEqual(ledger.percentOnDevice, 75.0, accuracy: 0.1)
    }

    func test_recordCloudEscalation_tracksTokensSpentAtLiquidPricing() {
        let ledger = TokenLedger()
        ledger.recordCloudEscalation(inputTokens: 1_000_000, outputTokens: 1_000_000)

        XCTAssertEqual(ledger.totalTokensSpentInCloud, 2_000_000)
        XCTAssertEqual(ledger.messagesCloudEscalated, 1)
        // Liquid cloud pricing: $0.50 in + $1.50 out per 1M → $2.00 total
        // for 1M in + 1M out. Meaningfully cheaper than competitor rates.
        XCTAssertEqual(ledger.estimatedDollarsSpentInCloud, 2.0, accuracy: 0.01)
    }

    func test_netDollarsSaved_usesCompetitorPricingOnSaved_LiquidPricingOnSpent() {
        let ledger = TokenLedger()
        ledger.recordOnDevice(inputTokens: 1_000_000, outputTokens: 1_000_000)      // $20 saved @ competitor rates
        ledger.recordCloudEscalation(inputTokens: 1_000_000, outputTokens: 1_000_000) // $2 spent @ Liquid rates
        // Net = $20 saved minus $2 actually paid to Liquid cloud = $18.
        // This is the "real dollars" customer story.
        XCTAssertEqual(ledger.netDollarsSaved, 18.0, accuracy: 0.01)
    }

    func test_reset_zeroesCounters() {
        let ledger = TokenLedger()
        ledger.recordOnDevice(inputTokens: 100, outputTokens: 50)
        ledger.recordDeflection()
        ledger.reset()

        XCTAssertEqual(ledger.totalTokensSaved, 0)
        XCTAssertEqual(ledger.messagesOnDevice, 0)
        XCTAssertEqual(ledger.messagesDeflected, 0)
    }
}
