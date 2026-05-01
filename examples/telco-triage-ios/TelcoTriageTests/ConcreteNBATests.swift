import XCTest
@testable import TelcoTriage

@MainActor
final class ConcreteNBATests: XCTestCase {
    private var profile: CustomerProfile!

    override func setUp() {
        super.setUp()
        profile = .demo
    }

    // MARK: - PlanOptimizeNBA

    func test_planOptimize_eligibleWhenUsageBelowCap() {
        let nba = PlanOptimizeNBA()
        XCTAssertTrue(nba.isEligible(for: profile))
    }

    func test_planOptimize_hasMonetaryValue() {
        let nba = PlanOptimizeNBA()
        XCTAssertEqual(nba.estimatedMonthlyDollars, 8.0)
    }

    // MARK: - MeshUpgradeUpsellNBA

    func test_meshUpgrade_eligibleWhenExtenderUnhealthy() {
        let nba = MeshUpgradeUpsellNBA()
        XCTAssertTrue(nba.isEligible(for: profile))
    }

    func test_meshUpgrade_priorityBoostedByBrokenExtender() {
        let nba = MeshUpgradeUpsellNBA()
        let score = nba.priorityScore(for: profile)
        XCTAssertGreaterThan(score, 0.85)
    }

    // MARK: - TravelPassBoltOnNBA

    func test_travelPass_eligibleWhenNotActive() {
        let nba = TravelPassBoltOnNBA()
        XCTAssertTrue(nba.isEligible(for: profile))
    }

    func test_travelPass_notEligibleWhenActive() {
        let nba = TravelPassBoltOnNBA()
        var p = profile!
        p = CustomerProfile(
            customerID: p.customerID,
            firstName: p.firstName,
            lastName: p.lastName,
            plan: p.plan,
            address: p.address,
            equipment: p.equipment,
            recentIssues: p.recentIssues,
            usage: CustomerProfile.UsageSnapshot(
                periodDays: p.usage.periodDays,
                downloadedGB: p.usage.downloadedGB,
                uploadedGB: p.usage.uploadedGB,
                connectedDeviceCount: p.usage.connectedDeviceCount,
                peakDeviceCount: p.usage.peakDeviceCount,
                troubleshootCount: p.usage.troubleshootCount,
                avgDownMbps: p.usage.avgDownMbps,
                avgUpMbps: p.usage.avgUpMbps,
                speedTestFailures: p.usage.speedTestFailures,
                outageMinutes: p.usage.outageMinutes,
                billCyclesAtOrOverCap: p.usage.billCyclesAtOrOverCap,
                activeBoltOns: ["TravelPass"]
            )
        )
        XCTAssertFalse(nba.isEligible(for: p))
    }

    // MARK: - SlowSpeedRetentionNBA

    func test_slowSpeedRetention_eligibleWithThreeFailures() {
        let nba = SlowSpeedRetentionNBA()
        XCTAssertTrue(nba.isEligible(for: profile))
    }

    func test_slowSpeedRetention_monetaryValueIsNegative() {
        let nba = SlowSpeedRetentionNBA()
        // Negative = Telco spends on retention credit.
        XCTAssertEqual(nba.estimatedMonthlyDollars, -15.0)
    }

    func test_slowSpeedRetention_highPriority() {
        let nba = SlowSpeedRetentionNBA()
        XCTAssertGreaterThan(nba.priorityScore(for: profile), 0.9)
    }

    // MARK: - ExtenderProactiveSupportNBA

    func test_extenderProactive_eligibleWhenExtenderUnhealthy() {
        let nba = ExtenderProactiveSupportNBA()
        XCTAssertTrue(nba.isEligible(for: profile))
    }

    func test_extenderProactive_noChatKeywords() {
        let nba = ExtenderProactiveSupportNBA()
        // This one is banner/tile only, not a chat attachment.
        XCTAssertNil(nba.chatAttachmentKeywords)
    }

    func test_extenderProactive_noMonetaryValue() {
        let nba = ExtenderProactiveSupportNBA()
        XCTAssertNil(nba.estimatedMonthlyDollars)
    }

    // MARK: - Categories

    func test_categories_oneOfEachType() {
        let categories = Set(NextBestActionRegistry.default.all.map(\.category))
        XCTAssertEqual(categories, Set(NBACategory.allCases))
    }
}
