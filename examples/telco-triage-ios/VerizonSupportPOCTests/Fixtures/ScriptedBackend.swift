import Foundation
@testable import VerizonSupportPOC

/// A single `AdapterInferenceBackend` that serves all three LFM paths
/// (intent classifier, tool selector, chat provider) by matching
/// prompt substrings to pre-scripted responses.
///
/// The real backends in the app share one `AdapterInferenceBackend`
/// because they all point to the same llama.cpp actor; we preserve
/// that shape in tests so the harness exercises the same plumbing.
///
/// Matcher semantics:
///  - Rules are evaluated in insertion order; first match wins.
///  - A rule matches if the prompt contains the configured substring.
///  - Unmatched prompts raise a test failure via the harness so tests
///    fail loudly on a missing script entry instead of silently
///    returning an empty string (which the ChatViewModel would then
///    report as `LFMChatError.emptyResponse` — confusing).
///
/// Actor-isolated so the script can be mutated from the test body
/// (e.g. script-then-send) without strict-concurrency warnings.
actor ScriptedBackend: AdapterInferenceBackend {
    struct Rule {
        let matches: String
        let response: String
    }

    private var rules: [Rule] = []
    private(set) var recordedPrompts: [String] = []
    private(set) var recordedStops: [[String]] = []
    private(set) var unmatchedPrompts: [String] = []
    private var forcedThrow: Bool = false

    func script(_ rule: Rule) {
        rules.append(rule)
    }

    func forceThrow(_ enabled: Bool) {
        forcedThrow = enabled
    }

    nonisolated func generate(
        messages: [AdapterChatMessage],
        adapterPath: String,
        maxTokens: Int,
        stopSequences: [String]
    ) async throws -> String {
        // Flatten the message sequence so existing string-substring
        // scripts (rules that look for the classifier/tool-selector
        // buildPrompt output) continue to match unchanged.
        let flat = messages.map(\.content).joined(separator: "\n\n")
        return try await self.dispatch(prompt: flat, stopSequences: stopSequences)
    }

    nonisolated func generate(
        prompt: String,
        adapterPath: String,
        maxTokens: Int,
        stopSequences: [String]
    ) async throws -> String {
        try await self.dispatch(prompt: prompt, stopSequences: stopSequences)
    }

    private func dispatch(prompt: String, stopSequences: [String]) throws -> String {
        recordedPrompts.append(prompt)
        recordedStops.append(stopSequences)
        if forcedThrow {
            throw NSError(domain: "ScriptedBackend", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "forced throw for test",
            ])
        }
        for rule in rules where prompt.contains(rule.matches) {
            return rule.response
        }
        unmatchedPrompts.append(prompt)
        // Return an empty string so the caller's error path runs. The
        // harness will surface `unmatchedPrompts` at assertion time so
        // the test fails with a clear message about which script entry
        // was missing.
        return ""
    }
}
