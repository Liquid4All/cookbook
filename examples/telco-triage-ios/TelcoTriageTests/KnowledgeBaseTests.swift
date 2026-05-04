import XCTest
@testable import TelcoTriage

final class KnowledgeBaseTests: XCTestCase {
    func test_bundledKB_loadsAndHasEntries() {
        let kb = KnowledgeBase.loadFromBundle()
        XCTAssertFalse(kb.entries.isEmpty, "bundled knowledge-base.json should load with entries")
        XCTAssertGreaterThanOrEqual(
            kb.entries.count, 14,
            "expect at least the original 14 telco sample FAQs seeded"
        )
    }

    func test_bundledKB_containsTelcoSampleFAQs() {
        let kb = KnowledgeBase.loadFromBundle()
        let ids = Set(kb.entries.map(\.id))
        // IDs sourced from the original telco sample FAQ sheet.
        XCTAssertTrue(ids.contains("restart-router"))
        XCTAssertTrue(ids.contains("router-speed-test"))
        XCTAssertTrue(ids.contains("wps-connect"))
        XCTAssertTrue(ids.contains("router-model"))
        XCTAssertTrue(ids.contains("activate-wifi-network"))
        XCTAssertTrue(ids.contains("home-page"))
    }

    func test_bundledKB_coversEveryRegisteredToolIntent() async {
        // W10 regression: every tool intent should be reachable via at
        // least one KB entry with `requiresToolExecution = true`. Otherwise
        // the router only picks up the tool via keyword fallback, which is
        // fragile.
        let kb = KnowledgeBase.loadFromBundle()
        let toolKBIDs = Set(kb.entries.filter(\.requiresToolExecution).map(\.id))

        // Must align with the tool-selector adapter's training catalog
        // (see `scripts/generate_telco_tool_selector.py::TELCO_TOOLS`).
        XCTAssertTrue(toolKBIDs.contains("restart-router"), "restart-router missing")
        XCTAssertTrue(toolKBIDs.contains("router-speed-test"), "router-speed-test missing")
        XCTAssertTrue(toolKBIDs.contains("check-connection"), "check-connection missing")
        XCTAssertTrue(toolKBIDs.contains("wps-connect"), "wps-connect missing")
    }

    func test_toolExecutionEntries_haveDeepLinks() {
        let kb = KnowledgeBase.loadFromBundle()
        for entry in kb.entries where entry.requiresToolExecution {
            XCTAssertFalse(
                entry.deepLinks.isEmpty,
                "Tool-execution entry \(entry.id) should have a deep link for Phase 2 routing"
            )
        }
    }
}
