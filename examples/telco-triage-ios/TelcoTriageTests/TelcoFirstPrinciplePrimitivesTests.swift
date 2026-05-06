import XCTest
@testable import TelcoTriage

    /// Pin-down tests for deterministic primitives that stay outside the
    /// learned routing path:
    ///   - `TelcoTopicGate` — off-topic refusal
    ///   - `KeywordKBExtractor` — KB selection over curated aliases
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
