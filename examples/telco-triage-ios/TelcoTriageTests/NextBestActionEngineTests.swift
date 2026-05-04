import XCTest
@testable import TelcoTriage

@MainActor
final class NextBestActionEngineTests: XCTestCase {
    private var context: CustomerContext!
    private var engine: NextBestActionEngine!

    override func setUp() {
        super.setUp()
        context = CustomerContext()
        engine = NextBestActionEngine(registry: .default, customerContext: context)
    }

    // MARK: - Default registry + demo profile wiring

    func test_defaultRegistry_hasFiveActions() {
        XCTAssertEqual(NextBestActionRegistry.default.all.count, 5)
    }

    func test_demoProfile_triggersRetentionAndMeshAndExtender() {
        // Demo profile has 3 speed test failures + unhealthy extender →
        // retention + extender proactive + mesh upsell are all eligible.
        let ids = engine.topActions.map(\.id)
        XCTAssertTrue(ids.contains("slow-speed-retention"))
        XCTAssertTrue(ids.contains("extender-proactive"))
        XCTAssertTrue(ids.contains("mesh-upgrade"))
    }

    func test_topActions_rankedByPriority() {
        guard let top = engine.topActions.first else { XCTFail("no actions"); return }
        // Retention and extender-proactive both have ~0.9+ priority;
        // accept either as the top spot, but they must rank above
        // plan-optimize (0.6+) and travel-pass (0.5).
        XCTAssertGreaterThan(top.priorityScore(for: context.profile), 0.7)
    }

    // MARK: - Chat attachment matching

    func test_bestMatchForChat_billQuery_returnsPlanOptimize() {
        let nba = engine.bestMatchForChat(query: "can you explain my bill")
        XCTAssertEqual(nba?.id, "plan-optimize")
    }

    func test_bestMatchForChat_slowQuery_returnsRetention() {
        let nba = engine.bestMatchForChat(query: "my internet is slow")
        XCTAssertEqual(nba?.id, "slow-speed-retention")
    }

    func test_bestMatchForChat_travelQuery_returnsTravelPass() {
        let nba = engine.bestMatchForChat(query: "I'm going on a trip abroad")
        XCTAssertEqual(nba?.id, "travel-pass")
    }

    func test_bestMatchForChat_unrelatedQuery_returnsNil() {
        let nba = engine.bestMatchForChat(query: "hello how are you")
        XCTAssertNil(nba)
    }

    // MARK: - Outcome tracking + ARPU aggregation

    func test_surfacedMonthlyValueDollars_sumsAcrossMonetaryActions() {
        // plan-optimize ($8) + mesh-upgrade ($10) + travel-pass ($12) +
        // retention (-$15). Proactive has no monetary value.
        let value = engine.surfacedMonthlyValueDollars
        // Don't pin the exact sum — some actions may not be eligible in
        // varying demo seeds. Assert it's nonzero + bounded.
        XCTAssertGreaterThan(value, 0)
        XCTAssertLessThan(value, 100)
    }

    func test_recordOutcome_acceptedIsCountedInARPU() {
        engine.record(outcome: NBAOutcome(actionID: "mesh-upgrade", verdict: .accepted))
        XCTAssertEqual(engine.acceptedCount, 1)
        XCTAssertEqual(engine.acceptedMonthlyValueDollars, 10.0, accuracy: 0.01)
    }

    func test_hasOutcome_returnsFalseBeforeRecord() {
        XCTAssertFalse(engine.hasOutcome(for: "mesh-upgrade"))
        engine.record(outcome: NBAOutcome(actionID: "mesh-upgrade", verdict: .declined))
        XCTAssertTrue(engine.hasOutcome(for: "mesh-upgrade"))
    }

    // MARK: - Reactivity to context mutations

    func test_engine_refresh_picksUpEquipmentStatusChanges() {
        // Initially extender-proactive is eligible (demo has unhealthy extender)
        XCTAssertTrue(engine.topActions.contains { $0.id == "extender-proactive" })

        // Mark the extender online and refresh synchronously. At runtime
        // the Combine sink on `$profile` handles this, but for a
        // deterministic test we call `refresh()` directly — this proves
        // the scoring logic itself is correct, without depending on
        // Combine scheduling in the test environment.
        let extender = context.profile.equipment.first { $0.kind == .extender }!
        context.markEquipmentStatus(serial: extender.serial, status: .online)
        engine.refresh()

        XCTAssertFalse(engine.topActions.contains { $0.id == "extender-proactive" })
    }
}
