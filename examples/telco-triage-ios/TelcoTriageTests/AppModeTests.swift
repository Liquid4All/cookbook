import XCTest
@testable import TelcoTriage

/// Tests for the AppMode enum, RoutingStage enum, and the customer-mode
/// ConversationStarter subset. Pure logic — no SwiftUI rendering.
final class AppModeTests: XCTestCase {

    // MARK: - AppMode

    func test_appMode_defaultsToCustomer() {
        // UserDefaults won't have the key in a fresh test target.
        let mode = AppMode(rawValue: "customer")
        XCTAssertEqual(mode, .customer)
    }

    func test_appMode_rawValues() {
        XCTAssertEqual(AppMode.customer.rawValue, "customer")
        XCTAssertEqual(AppMode.engineering.rawValue, "engineering")
    }

    func test_appMode_allCases() {
        XCTAssertEqual(AppMode.allCases.count, 2)
        XCTAssertTrue(AppMode.allCases.contains(.customer))
        XCTAssertTrue(AppMode.allCases.contains(.engineering))
    }

    func test_appMode_invalidRawValueReturnsNil() {
        XCTAssertNil(AppMode(rawValue: "debug"))
        XCTAssertNil(AppMode(rawValue: ""))
    }

    // MARK: - RoutingStage

    func test_routingStage_displayText_understanding() {
        XCTAssertEqual(RoutingStage.understanding.displayText, "Understanding...")
    }

    func test_routingStage_displayText_searching() {
        XCTAssertEqual(RoutingStage.searching.displayText, "Searching knowledge base...")
    }

    func test_routingStage_displayText_preparingAction() {
        XCTAssertEqual(RoutingStage.preparingAction.displayText, "Preparing action...")
    }

    func test_routingStage_displayText_composing() {
        XCTAssertEqual(RoutingStage.composing.displayText, "Composing response...")
    }

    func test_routingStage_rawValues() {
        XCTAssertEqual(RoutingStage.understanding.rawValue, "understanding")
        XCTAssertEqual(RoutingStage.searching.rawValue, "searching")
        XCTAssertEqual(RoutingStage.preparingAction.rawValue, "preparingAction")
        XCTAssertEqual(RoutingStage.composing.rawValue, "composing")
    }

    // MARK: - ConversationStarter.customer

    func test_customerStarters_hasSixChips() {
        XCTAssertEqual(ConversationStarter.customer.count, 6)
    }

    func test_customerStarters_idsAreUnique() {
        let ids = ConversationStarter.customer.map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count)
    }

    func test_customerStarters_fieldsNonEmpty() {
        for starter in ConversationStarter.customer {
            XCTAssertFalse(starter.prompt.isEmpty, "\(starter.id) missing prompt")
            XCTAssertFalse(starter.label.isEmpty, "\(starter.id) missing label")
            XCTAssertFalse(starter.icon.isEmpty, "\(starter.id) missing icon")
        }
    }

    func test_customerStarters_idsDoNotCollideWithAll() {
        let customerIDs = Set(ConversationStarter.customer.map(\.id))
        let allIDs = Set(ConversationStarter.all.map(\.id))
        // Customer IDs are prefixed c- to avoid identity collision
        XCTAssertTrue(customerIDs.isDisjoint(with: allIDs),
                      "Customer and engineering starters share IDs: \(customerIDs.intersection(allIDs))")
    }

    func test_customerStarters_coverEveryRouterMode() {
        let ids = Set(ConversationStarter.customer.map(\.id))
        // One chip per ChatModeRouter label + the KB/tool modality pair
        // that proves the router reads intent, not keywords.
        XCTAssertTrue(ids.contains("c-wifi-slow-kb"),
                      "Missing kb_question chip (KBExtractor best-covered entry)")
        XCTAssertTrue(ids.contains("c-restart-router-kb"),
                      "Missing kb_question chip (half of the KB/tool pair)")
        XCTAssertTrue(ids.contains("c-restart-router-tool"),
                      "Missing tool_action chip (half of the KB/tool pair)")
        XCTAssertTrue(ids.contains("c-parental-pause"),
                      "Missing tool_action chip with arg extraction")
        XCTAssertTrue(ids.contains("c-connected-devices"),
                      "Missing personal_summary chip")
        XCTAssertTrue(ids.contains("c-out-of-scope"),
                      "Missing out_of_scope chip")
    }

    // MARK: - ConversationStarter.all (engineering mode)

    func test_allStarters_hasSixChips() {
        XCTAssertEqual(ConversationStarter.all.count, 6)
    }

    func test_allStarters_idsAreUnique() {
        let ids = ConversationStarter.all.map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count)
    }

    func test_allStarters_fieldsNonEmpty() {
        for starter in ConversationStarter.all {
            XCTAssertFalse(starter.prompt.isEmpty, "\(starter.id) missing prompt")
            XCTAssertFalse(starter.label.isEmpty, "\(starter.id) missing label")
            XCTAssertFalse(starter.icon.isEmpty, "\(starter.id) missing icon")
            XCTAssertFalse(starter.primitive.isEmpty, "\(starter.id) missing primitive")
        }
    }

    /// Guards against regressions like the "block my son's tablet from
    /// the internet" wording that caused the tool-selector LoRA to
    /// hallucinate a non-existent tool_id (`block-parental-controls`).
    /// Any prompt with a verb outside the v3 training manifold should
    /// be tested against the local harness before landing here.
    func test_allStarters_coverEveryPrimitive() {
        let ids = Set(ConversationStarter.all.map(\.id))
        XCTAssertTrue(ids.contains("rag-restart-router"),
                      "Missing grounded-QA primitive")
        XCTAssertTrue(ids.contains("tool-parental-pause"),
                      "Missing tool-call + device-extraction primitive")
        XCTAssertTrue(ids.contains("agentic-diagnostics"),
                      "Missing agentic-diagnostics primitive")
        XCTAssertTrue(ids.contains("tool-reboot-extender-upstairs"),
                      "Missing location-extraction primitive")
        XCTAssertTrue(ids.contains("personalized-summary"),
                      "Missing personalization primitive")
        XCTAssertTrue(ids.contains("privacy-out-of-scope"),
                      "Missing privacy/out-of-scope primitive")
    }

    /// The parental-controls engineering chip must use a verb dense in
    /// the tool-selector v3 training distribution (`pause`). The older
    /// "block …" wording caused a hallucinated tool_id on the local
    /// harness. Pin the prompt until tool-selector v4 ships.
    func test_allStarters_parentalChipUsesTrainedVerb() {
        guard let chip = ConversationStarter.all.first(where: { $0.id == "tool-parental-pause" }) else {
            return XCTFail("tool-parental-pause chip missing from engineering set")
        }
        XCTAssertTrue(chip.prompt.localizedCaseInsensitiveContains("pause"),
                      "Engineering parental-controls chip must use 'pause' verb; got \(chip.prompt)")
        XCTAssertFalse(chip.prompt.localizedCaseInsensitiveContains("block "),
                       "'block …' wording regressed — hallucinates tool_id on tool-selector v3")
    }

    // MARK: - Persistence

    func test_appMode_persistsToUserDefaults() {
        // Write engineering mode directly to UserDefaults
        UserDefaults.standard.set("engineering", forKey: "appMode")
        let restored = AppMode(rawValue: UserDefaults.standard.string(forKey: "appMode") ?? "customer")
        XCTAssertEqual(restored, .engineering)
        // Cleanup
        UserDefaults.standard.removeObject(forKey: "appMode")
    }

    func test_appMode_missingKeyDefaultsToCustomer() {
        UserDefaults.standard.removeObject(forKey: "appMode")
        let raw = UserDefaults.standard.string(forKey: "appMode") ?? "customer"
        XCTAssertEqual(AppMode(rawValue: raw), .customer)
    }

    // MARK: - Customer starters content

    func test_customerStarters_noEngineeringJargon() {
        // Customer-facing labels should not contain engineering terms
        let jargon = ["primitive", "extraction", "classifier", "LoRA", "adapter",
                      "out-of-scope", "intent", "inference"]
        for starter in ConversationStarter.customer {
            for term in jargon {
                XCTAssertFalse(
                    starter.label.localizedCaseInsensitiveContains(term),
                    "Customer label '\(starter.label)' contains jargon: \(term)"
                )
            }
        }
    }
}
