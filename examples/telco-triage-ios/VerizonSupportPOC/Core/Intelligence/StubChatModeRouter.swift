import Foundation

/// Deterministic stub `ChatModeRouter`. Returns a caller-configured
/// prediction regardless of input. Two intended uses:
///
///  1. **Tests** — wire a stub into `ChatViewModel` and script each
///     test's expected mode without standing up the LFM stack.
///  2. **Scaffolding under BUG-022** — while on-device LFM inference
///     emits only `<|pad|>` tokens, the production app cannot rely
///     on `LFMChatModeRouter`. `StubChatModeRouter(mode: .kbQuestion)`
///     wired as the default lets the rest of the pipeline (KB
///     extraction, tool selection, handlers) stay runnable and
///     testable end-to-end.
///
/// This stub does NOT perform keyword matching, TF-IDF, or any lexical
/// primitive — per the architectural directive that routing is a pure
/// LFM decision. If you need behavior-per-query in a test, use
/// `ScriptedChatModeRouter` (below) which matches by prompt substring.
public struct StubChatModeRouter: ChatModeRouter {
    private let prediction: ChatModePrediction

    public init(
        mode: ChatMode = .kbQuestion,
        confidence: Double = 1.0,
        reasoning: String = "stub"
    ) {
        self.prediction = ChatModePrediction(
            mode: mode,
            confidence: confidence,
            reasoning: reasoning,
            runtimeMS: 0
        )
    }

    public func classify(query: String) async -> ChatModePrediction {
        prediction
    }
}

/// Test-only mode router that maps query substrings to pre-configured
/// predictions. Useful when a single test exercises multiple modes
/// (e.g. TelcoScenarioPipelineTests running both question and action
/// flows). First matching rule wins; on no match, returns the
/// `fallback` prediction.
///
/// Actor-isolated so rules can be mutated mid-test without strict
/// concurrency warnings — same shape as `ScriptedBackend`.
public actor ScriptedChatModeRouter: ChatModeRouter {
    public struct Rule: Sendable {
        public let matches: String
        public let prediction: ChatModePrediction

        public init(matches: String, prediction: ChatModePrediction) {
            self.matches = matches
            self.prediction = prediction
        }
    }

    private var rules: [Rule] = []
    private let fallback: ChatModePrediction

    public init(
        fallback: ChatModePrediction = ChatModePrediction(
            mode: .outOfScope,
            confidence: 0.0,
            reasoning: "no rule matched",
            runtimeMS: 0
        )
    ) {
        self.fallback = fallback
    }

    public func script(_ rule: Rule) {
        rules.append(rule)
    }

    public nonisolated func classify(query: String) async -> ChatModePrediction {
        await self.dispatch(query: query)
    }

    private func dispatch(query: String) -> ChatModePrediction {
        for rule in rules where query.lowercased().contains(rule.matches.lowercased()) {
            return rule.prediction
        }
        return fallback
    }
}
