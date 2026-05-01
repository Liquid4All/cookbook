import XCTest
@testable import VerizonSupportPOC

/// Pin-down tests for the deterministic primitives that replaced the
/// brittle ML approaches:
///   - `TelcoTopicGate` — off-topic refusal
///   - `KeywordKBExtractor` — KB selection over curated aliases
///   - `ImperativeToolDetector` — iOS-only tool overrides
///
/// Every failing scenario the user reported gets a test here. If a
/// regression sneaks in, these tests catch it before the build ships.
final class TelcoFirstPrinciplePrimitivesTests: XCTestCase {
    // MARK: - Topic gate

    func test_topicGate_refusesWeather() {
        let gate = TelcoTopicGate()
        if case .outOfScope = gate.decide("what is the weather in new york") {
            // expected
        } else {
            XCTFail("weather should be refused as out_of_scope")
        }
    }

    func test_topicGate_passesRouterRestart() {
        XCTAssertEqual(TelcoTopicGate().decide("how do I restart my router"), .inScope)
    }

    func test_topicGate_passesPauseInternetForKid() {
        XCTAssertEqual(
            TelcoTopicGate().decide("pause internet for my son's tablet"),
            .inScope
        )
    }

    func test_topicGate_refusesGeneralChat() {
        if case .outOfScope = TelcoTopicGate().decide("tell me a joke") { return }
        XCTFail("general chat should be refused")
    }

    // MARK: - Keyword KB extractor — the share-wifi/restart-router bug

    func test_kbExtractor_restartRouterPicksRestartRouter() async {
        let kb = TelcoFirstPrinciplePrimitivesTests.fixtureKB()
        let citation = await KeywordKBExtractor().extract(
            query: "how do I restart my router",
            kb: kb
        )
        // The bug we shipped a fix for: stopwords ("how", "do", "my")
        // overlapped with share-wifi's aliases ("share my wifi",
        // "how do I share wifi"), pulling share-wifi over restart-router.
        // After filtering stopwords on both sides + adding ID-as-signal,
        // restart-router wins cleanly.
        XCTAssertEqual(citation.entryId, "restart-router",
                       "stopwords must not drive scoring; restart-router must win")
    }

    func test_kbExtractor_pauseInternetPicksParentalControls() async {
        let citation = await KeywordKBExtractor().extract(
            query: "pause internet for my son's tablet",
            kb: TelcoFirstPrinciplePrimitivesTests.fixtureKB()
        )
        XCTAssertEqual(citation.entryId, "parental-controls")
    }

    func test_kbExtractor_ssidPicksFindWifiName() async {
        let citation = await KeywordKBExtractor().extract(
            query: "what is the ssid",
            kb: TelcoFirstPrinciplePrimitivesTests.fixtureKB()
        )
        XCTAssertEqual(citation.entryId, "find-wifi-name")
    }

    func test_kbExtractor_unknownQueryReturnsNoMatch() async {
        let citation = await KeywordKBExtractor().extract(
            query: "blah blah lorem ipsum",
            kb: TelcoFirstPrinciplePrimitivesTests.fixtureKB()
        )
        XCTAssertEqual(citation.entryId, KBCitation.noMatchID)
    }

    // MARK: - Imperative tool detector — schema-missing tools

    func test_imperativeDetector_pauseInternetForKid() {
        XCTAssertEqual(
            ImperativeToolDetector.detect("pause internet for my son's tablet"),
            .toggleParentalControls
        )
    }

    func test_imperativeDetector_blockKidsWifi() {
        XCTAssertEqual(
            ImperativeToolDetector.detect("block wifi for the kids"),
            .toggleParentalControls
        )
    }

    func test_imperativeDetector_restartExtender() {
        XCTAssertEqual(
            ImperativeToolDetector.detect("restart the wifi extender upstairs"),
            .rebootExtender
        )
    }

    func test_imperativeDetector_resetMeshNode() {
        XCTAssertEqual(
            ImperativeToolDetector.detect("reset the mesh node in the bedroom"),
            .rebootExtender
        )
    }

    func test_imperativeDetector_doesNotOverrideHowToQuestion() {
        // "How do I pause internet" is a real KB lookup question.
        // The override must NOT fire — classifier output stands.
        XCTAssertNil(ImperativeToolDetector.detect("how do I pause internet for my kid"))
    }

    func test_imperativeDetector_doesNotOverrideWhatIsQuestion() {
        XCTAssertNil(ImperativeToolDetector.detect("what is parental controls"))
    }

    func test_imperativeDetector_ignoresUnrelatedQueries() {
        XCTAssertNil(ImperativeToolDetector.detect("how do I share wifi with a guest"))
        XCTAssertNil(ImperativeToolDetector.detect("what is the ssid"))
        XCTAssertNil(ImperativeToolDetector.detect("restart my router"))   // restart_gateway is in-schema
    }

    // MARK: - Personal summary detector — "show me my data"

    func test_personalSummaryDetector_summarizeMyHomeNetwork() {
        // The exact failure the user flagged: "summarize my home
        // network" was being routed to run_diagnostics. It's a
        // status read, not a fix-it action.
        XCTAssertTrue(PersonalSummaryDetector.detect("summarize my home network"))
    }

    func test_personalSummaryDetector_showMyConnectedDevices() {
        XCTAssertTrue(PersonalSummaryDetector.detect("show my connected devices"))
        XCTAssertTrue(PersonalSummaryDetector.detect("show me my connected devices"))
    }

    func test_personalSummaryDetector_whatDevices() {
        XCTAssertTrue(PersonalSummaryDetector.detect("what devices are on my network"))
        XCTAssertTrue(PersonalSummaryDetector.detect("which devices are connected"))
    }

    func test_personalSummaryDetector_doesNotFireOnUnrelatedSummary() {
        // No personal-data noun → not a personalSummary signal.
        XCTAssertFalse(PersonalSummaryDetector.detect("summarize this article"))
        XCTAssertFalse(PersonalSummaryDetector.detect("give me a summary of news"))
    }

    func test_personalSummaryDetector_doesNotFireOnHowToQuestion() {
        // "how do I view my devices" is a KB question, not a
        // personalSummary read — the user wants steps, not data.
        // The detector keys on phrases like "show my" / "summarize"
        // / "what devices", which "how do I view my devices" doesn't
        // contain.
        XCTAssertFalse(PersonalSummaryDetector.detect("how do I view my devices"))
    }

    func test_personalSummaryDetector_ignoresActionRequests() {
        XCTAssertFalse(PersonalSummaryDetector.detect("restart my router"))
        XCTAssertFalse(PersonalSummaryDetector.detect("pause internet for my son's tablet"))
        XCTAssertFalse(PersonalSummaryDetector.detect("what is the ssid"))
    }

    // MARK: - Fixtures

    /// Subset of the production KB sufficient to exercise the bug
    /// classes: aliased entries (find-wifi-name), heavy-stopword aliases
    /// (share-wifi), and the regression target (restart-router).
    private static func fixtureKB() -> [KBEntry] {
        [
            KBEntry(
                id: "restart-router",
                topic: "Restart Router",
                aliases: ["Turn off and on again router"],
                category: "WiFi",
                answer: "To restart your router: tap Equipment, then Restart.",
                deepLinks: [],
                tags: ["router", "restart"],
                requiresToolExecution: false
            ),
            KBEntry(
                id: "share-wifi",
                topic: "Share Wi-Fi",
                aliases: [
                    "share wifi", "share my wifi", "how do I share wifi",
                    "send wifi password", "wifi qr code",
                    "guest wifi access", "share network with friend",
                ],
                category: "WiFi",
                answer: "Tap Share Wi-Fi from the Home page.",
                deepLinks: [],
                tags: ["share", "wifi", "guest", "qr", "password"],
                requiresToolExecution: false
            ),
            KBEntry(
                id: "find-wifi-name",
                topic: "Find Your WiFi Name (SSID)",
                aliases: [
                    "ssid", "wifi name", "network name", "what is my ssid",
                    "find my ssid", "wireless network name", "my wifi name",
                    "show ssid",
                ],
                category: "WiFi",
                answer: "Your SSID is shown at the top of the Network tile.",
                deepLinks: [],
                tags: ["wifi", "ssid", "network-name"],
                requiresToolExecution: false
            ),
            KBEntry(
                id: "parental-controls",
                topic: "Parental Controls",
                aliases: [
                    "parental controls", "block websites", "time limits",
                    "pause internet", "content filter", "kids",
                ],
                category: "Security",
                answer: "Open Parental Controls from the Home page.",
                deepLinks: [],
                tags: ["parental", "controls", "kids"],
                requiresToolExecution: false
            ),
        ]
    }
}
