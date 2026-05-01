import XCTest
@testable import VerizonSupportPOC

final class ConversationStarterTests: XCTestCase {
    /// Locks the 6 proven-capability chips the telco pitch demos.
    /// Adding or renaming a chip requires updating this test AND
    /// docs/FUTURE_SCOPE.md (the retrain plan for any chip that
    /// lands on unproven capability).
    func test_all_coversSixProvenCapabilities() {
        let ids = Set(ConversationStarter.all.map(\.id))
        let expected: Set<String> = [
            "rag-restart-router",
            "tool-parental-pause",
            "agentic-diagnostics",
            "tool-reboot-extender-upstairs",
            "personalized-summary",
            "privacy-out-of-scope",
        ]
        XCTAssertEqual(ids, expected)
    }

    func test_all_idsAreUnique() {
        let ids = ConversationStarter.all.map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count)
    }

    func test_all_fieldsNonEmpty() {
        for starter in ConversationStarter.all {
            XCTAssertFalse(starter.prompt.isEmpty, "\(starter.id) missing prompt")
            XCTAssertFalse(starter.label.isEmpty, "\(starter.id) missing label")
            XCTAssertFalse(starter.icon.isEmpty, "\(starter.id) missing icon")
            XCTAssertFalse(starter.primitive.isEmpty, "\(starter.id) missing primitive")
        }
    }

    func test_privacyBoundaryChip_rendersOffTopicPrompt() {
        let chip = ConversationStarter.all.first { $0.id == "privacy-out-of-scope" }
        XCTAssertNotNil(chip)
        // The prompt must be clearly off-topic so the intent classifier
        // emits `unknown`. Matches training data in
        // `scripts/generate_telco_intent.py` unknown scenarios.
        XCTAssertEqual(chip?.prompt, "what is the weather in new york")
    }

    func test_personalizationChip_exists() {
        // In the TF-IDF era a dedicated lexical check
        // (`SupportRouter.isPersonalizationQuery`) confirmed this
        // starter's phrasing would fire the personalization branch.
        // After Phase A.3, routing is an LFM decision against the
        // full query — the starter chip only promises the prompt
        // text itself. Mode validation happens in the scenario
        // harness with a ScriptedChatModeRouter.
        let chip = ConversationStarter.all.first { $0.id == "personalized-summary" }
        XCTAssertNotNil(chip)
        XCTAssertFalse(chip?.prompt.isEmpty ?? true)
    }
}
