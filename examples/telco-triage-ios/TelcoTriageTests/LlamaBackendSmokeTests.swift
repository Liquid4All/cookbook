import XCTest
@testable import TelcoTriage

/// BUG-022 regression + on-device smoke for the bundled LFM2.5-350M
/// base GGUF. Gated behind `LFM_ON_DEVICE_SMOKE=1` (set via
/// `TEST_RUNNER_LFM_ON_DEVICE_SMOKE=1` when invoked through
/// xcodebuild) because the 219 MB model load + inference runs takes
/// ~60 s per run — too expensive for default CI.
///
/// The three tests cover, in order of diagnostic value:
///  1. Base model generates non-pad output for a canonical prompt.
///     Regression for BUG-022 (symptom: all `<|pad|>` tokens).
///  2. Tool-selector LoRA round-trips through the ChatML-templated
///     adapter path and emits JSON-shaped output.
///  3. ChatModeRouter produces a parseable ChatMode prediction for
///     a canonical question query.
final class LlamaBackendSmokeTests: XCTestCase {

    private var isEnabled: Bool {
        ProcessInfo.processInfo.environment["LFM_ON_DEVICE_SMOKE"] == "1"
    }

    private func makeLoadedBackend() async throws -> LlamaBackend {
        guard let basePath = TelcoModelBundle.basePath() else {
            throw XCTSkip("Bundled base GGUF missing — see bootstrap-models.sh")
        }
        let backend = LlamaBackend()
        // The iOS simulator reports 0 MiB free on MTL0 — requesting
        // GPU offload there silently produces garbage (argmax lands on
        // pad every step). Real devices have real Metal memory and
        // the production path uses `gpuLayers: 99`. For the smoke
        // tests we force CPU to get meaningful output in the sim.
        let gpuLayers: Int32
        #if targetEnvironment(simulator)
        gpuLayers = 0
        #else
        gpuLayers = 99
        #endif
        try await backend.loadModel(
            path: basePath,
            contextLength: 8192,
            gpuLayers: gpuLayers,
            temperature: 0
        )
        return backend
    }

    func test_baseModel_generatesNonPadOutput() async throws {
        try XCTSkipUnless(isEnabled, "Set LFM_ON_DEVICE_SMOKE=1 to run")
        let backend = try await makeLoadedBackend()

        // Deliberately simple arithmetic question — the base LFM2.5-350M
        // should handle this in 1–10 tokens. If the output is still all
        // `<|pad|>`, BUG-022 is not fully resolved (version bump didn't
        // catch the full fix — probably need the refreshed GGUF too).
        let (raw, tokenCount, _) = try await backend.generate(
            messages: [.user("What is 2 plus 2? Answer with just the number.")],
            maxTokens: 40,
            temperature: 0,
            stopSequences: [],
            clearCache: true,
            outputMode: .text
        )

        XCTAssertGreaterThan(tokenCount, 0, "no tokens emitted")
        XCTAssertFalse(
            raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "empty output after trimming: raw=\(raw.prefix(200))"
        )
        // The specific bug signature was the detokenized string being
        // literal `<|pad|>` repeated. Assert the pad-token sentinel is
        // absent (or at most trailing).
        let padCount = raw.components(separatedBy: "<|pad|>").count - 1
        XCTAssertLessThan(
            padCount, tokenCount,
            "output is still dominated by <|pad|>: raw=\(raw.prefix(200))"
        )
        // Attach the raw output so future BUG-022-style investigations
        // have ground truth without recreating the diagnostic.
        let attach = XCTAttachment(string: raw)
        attach.name = "base_model_output"
        attach.lifetime = .keepAlways
        add(attach)
    }

    func test_toolSelectorAdapter_generatesJSON() async throws {
        try XCTSkipUnless(isEnabled, "Set LFM_ON_DEVICE_SMOKE=1 to run")
        guard let toolAdapter = TelcoModelBundle.toolAdapterPath() else {
            throw XCTSkip("Tool-selector adapter missing")
        }
        let backend = try await makeLoadedBackend()
        try await backend.setAdapter(path: toolAdapter, scale: 1.0)

        let prompt = LFMToolSelector.buildPrompt(query: "restart my router")
        let (raw, tokenCount, _) = try await backend.generate(
            messages: [.user(prompt)],
            maxTokens: 120,
            temperature: 0,
            stopSequences: [],
            clearCache: true,
            outputMode: .text
        )
        XCTAssertGreaterThan(tokenCount, 0)
        // Adapter was trained to emit a JSON object with a `tool_id`
        // field. We don't assert a specific id — just that the output
        // contains JSON shape.
        XCTAssertTrue(
            raw.contains("{") && raw.contains("tool_id"),
            "tool-selector adapter output isn't JSON-shaped: \(raw.prefix(200))"
        )
        let attach = XCTAttachment(string: raw)
        attach.name = "tool_selector_output"
        attach.lifetime = .keepAlways
        add(attach)
    }

    func test_chatModeRouter_producesParseablePrediction() async throws {
        try XCTSkipUnless(isEnabled, "Set LFM_ON_DEVICE_SMOKE=1 to run")
        let backend = try await makeLoadedBackend()
        let bridge = LlamaAdapterBackend(backend: backend)
        // Apply the shipping LoRA if bundled; otherwise fall back to the
        // base model so this smoke test still produces a parseable
        // prediction on a fresh clone without the LoRA artifact.
        let adapter = TelcoModelBundle.chatModeRouterAdapterPath() ?? ""
        let router = LFMChatModeRouter(backend: bridge, adapterPath: adapter)

        let prediction = await router.classify(query: "how do I restart my router")

        // Canonical "how do I …" phrasing is paradigmatic kb_question.
        // With the v1 LoRA we expect a high-confidence kb_question;
        // without the adapter, we accept any non-fallback mode to avoid
        // asserting on the base model's (zero-shot) judgment.
        XCTAssertGreaterThan(
            prediction.confidence, 0.0,
            "fallback confidence (0.0) means the router couldn't parse model output: reasoning=\(prediction.reasoning)"
        )
    }
}
