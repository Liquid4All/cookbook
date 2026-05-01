import XCTest
@testable import VerizonSupportPOC

/// Phase B validation harness for the LFM-backed routing + retrieval
/// primitives. Measures whether the base LFM2.5-350M (no LoRA) is
/// discriminative enough for the 4-way ChatMode classification and
/// the 32-entry KB generative retrieval.
///
/// Gated behind `LFM_ON_DEVICE_SMOKE=1` — the full 30-query mode
/// fixture plus the KB-extractor sweep takes ~2-5 minutes on the
/// simulator CPU. Default CI skips.
///
/// Each test attaches a per-query result table (correct/incorrect +
/// model output) via XCTAttachment so failures are debuggable without
/// a re-run.
final class LFMValidationTests: XCTestCase {

    private var isEnabled: Bool {
        ProcessInfo.processInfo.environment["LFM_ON_DEVICE_SMOKE"] == "1"
    }

    private static var sharedBackend: LlamaBackend?

    override class func setUp() {
        super.setUp()
        guard ProcessInfo.processInfo.environment["LFM_ON_DEVICE_SMOKE"] == "1" else { return }
        guard let basePath = TelcoModelBundle.basePath() else {
            XCTFail("Bundled base GGUF missing")
            return
        }
        let gpuLayers: Int32
        #if targetEnvironment(simulator)
        gpuLayers = 0
        #else
        gpuLayers = 99
        #endif
        let backend = LlamaBackend()
        let g = DispatchGroup(); g.enter()
        Task {
            defer { g.leave() }
            do {
                try await backend.loadModel(
                    path: basePath,
                    contextLength: 8192,
                    gpuLayers: gpuLayers,
                    temperature: 0
                )
            } catch { XCTFail("loadModel failed: \(error)") }
        }
        g.wait()
        sharedBackend = backend
    }

    override class func tearDown() {
        sharedBackend = nil
        super.tearDown()
    }

    // MARK: - ChatModeRouter validation

    /// 30-query fixture covering all 4 modes with realistic phrasings
    /// (how-to questions, imperative actions, profile summaries,
    /// off-topic).
    ///
    /// **Iteration 2 ship (2026-04-22, chat-mode-router-v2 LoRA):**
    ///   - eval_mean_token_accuracy (held-out 20%): 96.55 %
    ///   - telco KB sample harness broken-rate: 47 % → 13 %
    ///   - Cluster A (possessive field lookups): 0/13 → 12/13 fixed
    ///   - Chip regression (13 chips): 13/13 pass, zero regression
    ///
    /// The 0.90 threshold below is the on-device acceptance gate. A drop below
    /// that usually means the wrong base model was bundled, the adapter was
    /// not applied, or the prompt drifted from the training template.
    func test_chatModeRouter_fixture_meetsAccuracyThreshold() async throws {
        try XCTSkipUnless(isEnabled, "Set LFM_ON_DEVICE_SMOKE=1 to run")
        guard let backend = Self.sharedBackend else {
            XCTFail("shared backend not initialized"); return
        }
        guard let chatModeRouterAdapter = TelcoModelBundle.chatModeRouterAdapterPath() else {
            XCTFail("chat-mode-router-v2.gguf missing from Resources/Models — run bootstrap-models.sh")
            return
        }
        let bridge = LlamaAdapterBackend(backend: backend)
        let router = LFMChatModeRouter(backend: bridge, adapterPath: chatModeRouterAdapter)

        var correct = 0
        var table: [String] = ["expected\tgot\tconfidence\tquery"]
        for item in Self.modeFixture {
            let prediction = await router.classify(query: item.query)
            let isCorrect = prediction.mode == item.expected
            if isCorrect { correct += 1 }
            table.append(String(
                format: "%@\t%@\t%.2f\t%@",
                item.expected.rawValue,
                prediction.mode.rawValue,
                prediction.confidence,
                item.query
            ))
        }
        let accuracy = Double(correct) / Double(Self.modeFixture.count)
        table.insert("accuracy: \(correct)/\(Self.modeFixture.count) = \(String(format: "%.1f%%", accuracy * 100))",
                     at: 0)

        let att = XCTAttachment(string: table.joined(separator: "\n"))
        att.name = "mode_router_results.tsv"
        att.lifetime = .keepAlways
        add(att)

        // On-device acceptance gate: >= 90%.
        XCTAssertGreaterThanOrEqual(
            accuracy, 0.90,
            "ChatModeRouter on-device accuracy \(correct)/\(Self.modeFixture.count) below ship gate — see attached TSV"
        )
    }

    // MARK: - KBExtractor validation

    /// For a handful of canonical KB queries, measure whether the
    /// fine-tuned KB extractor picks the expected entry id.
    ///
    /// **Phase C:** the prompt is ~40 tokens (no inline KB). The LoRA
    /// memorizes the 32-entry KB. The simulator-crash issue from Phase B
    /// (3.2K-token inline prompt) no longer applies — this test runs on
    /// both simulator and device.
    ///
    /// Threshold: 0.80 post-LoRA (lower than ChatModeRouter's 0.90 because
    /// the 4-query fixture is deliberately hard — representative of the
    /// worst 10% of real traffic).
    func test_kbExtractor_canonicalQueries_citeExpectedEntries() async throws {
        try XCTSkipUnless(isEnabled, "Set LFM_ON_DEVICE_SMOKE=1 to run")
        guard let backend = Self.sharedBackend else {
            XCTFail("shared backend not initialized"); return
        }
        guard let kbExtractorAdapter = TelcoModelBundle.kbExtractorAdapterPath() else {
            XCTFail("kb-extractor-v1.gguf missing from Resources/Models — run bootstrap-models.sh")
            return
        }
        let bridge = LlamaAdapterBackend(backend: backend)
        let extractor = LFMKBExtractor(backend: bridge, adapterPath: kbExtractorAdapter)
        let kb = KnowledgeBase.loadFromBundle().entries

        var correct = 0
        var table: [String] = ["expected_entry\tgot_entry\tconfidence\tquery"]
        for item in Self.kbFixture {
            let citation = await extractor.extract(query: item.query, kb: kb)
            let isCorrect = citation.entryId == item.expectedEntryID
            if isCorrect { correct += 1 }
            table.append(String(
                format: "%@\t%@\t%.2f\t%@",
                item.expectedEntryID,
                citation.entryId,
                citation.confidence,
                item.query
            ))
        }
        let accuracy = Double(correct) / Double(Self.kbFixture.count)
        table.insert("accuracy: \(correct)/\(Self.kbFixture.count) = \(String(format: "%.1f%%", accuracy * 100))",
                     at: 0)

        let att = XCTAttachment(string: table.joined(separator: "\n"))
        att.name = "kb_extractor_results.tsv"
        att.lifetime = .keepAlways
        add(att)

        // On-device acceptance gate: >= 80%.
        XCTAssertGreaterThanOrEqual(
            accuracy, 0.80,
            "KBExtractor on-device accuracy \(correct)/\(Self.kbFixture.count) below ship gate — see attached TSV"
        )
    }

    // MARK: - Fixtures

    private struct ModeCase: Sendable {
        let query: String
        let expected: ChatMode
    }

    private struct KBCase: Sendable {
        let query: String
        let expectedEntryID: String
    }

    /// 30 queries, balanced across the 4 modes with realistic home
    /// internet support phrasings. Spread = 8 + 8 + 6 + 8.
    private static let modeFixture: [ModeCase] = [
        // kb_question — how-to / informational
        .init(query: "how do I restart my router", expected: .kbQuestion),
        .init(query: "what are the LED colors on my router", expected: .kbQuestion),
        .init(query: "where is the reset button on the G3100", expected: .kbQuestion),
        .init(query: "can you explain WPS", expected: .kbQuestion),
        .init(query: "what is a mesh extender", expected: .kbQuestion),
        .init(query: "how do I change my wifi password", expected: .kbQuestion),
        .init(query: "why is my connection slow sometimes", expected: .kbQuestion),
        .init(query: "how do parental controls work on fiber internet", expected: .kbQuestion),

        // tool_action — imperatives
        .init(query: "restart my router", expected: .toolAction),
        .init(query: "run a speed test", expected: .toolAction),
        .init(query: "pause internet for my son's tablet", expected: .toolAction),
        .init(query: "check the connection status", expected: .toolAction),
        .init(query: "reboot the living room extender", expected: .toolAction),
        .init(query: "schedule a technician for tomorrow", expected: .toolAction),
        .init(query: "enable WPS now", expected: .toolAction),
        .init(query: "run network diagnostics", expected: .toolAction),

        // personal_summary
        .init(query: "summarize my home network", expected: .personalSummary),
        .init(query: "tell me about my plan", expected: .personalSummary),
        .init(query: "what is my current setup", expected: .personalSummary),
        .init(query: "give me an overview of my account", expected: .personalSummary),
        .init(query: "how is my wifi doing overall", expected: .personalSummary),
        .init(query: "status of my home internet", expected: .personalSummary),

        // out_of_scope
        .init(query: "what is the weather today", expected: .outOfScope),
        .init(query: "tell me a joke", expected: .outOfScope),
        .init(query: "who won the lakers game", expected: .outOfScope),
        .init(query: "what's a good pasta recipe", expected: .outOfScope),
        .init(query: "translate hello to spanish", expected: .outOfScope),
        .init(query: "how do I change a tire", expected: .outOfScope),
        .init(query: "what's the capital of france", expected: .outOfScope),
        .init(query: "write a poem about dogs", expected: .outOfScope),
    ]

    /// 4-query KB-extractor validation. Deliberately small because
    /// each call runs ~3,200-token prompts through simulator CPU,
    /// ~30-60 s each. Extend on a real device or a sidecar harness.
    /// All 4 IDs verified against the bundled knowledge-base.json.
    private static let kbFixture: [KBCase] = [
        .init(query: "how do I restart my router", expectedEntryID: "restart-router"),
        .init(query: "change my wifi password", expectedEntryID: "change-wifi-password"),
        .init(query: "my wifi signal is very weak", expectedEntryID: "weak-signal-troubleshoot"),
        .init(query: "test my internet speed", expectedEntryID: "router-speed-test"),
    ]
}
