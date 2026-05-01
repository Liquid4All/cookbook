import Foundation

/// Deterministic stub `KBExtractor`. Returns a caller-configured
/// citation regardless of input. Two intended uses:
///
///  1. **Tests** — wire a stub into `ChatViewModel` and assert on
///     citation handling without standing up the LFM stack.
///  2. **Scaffolding under BUG-022** — while on-device LFM inference
///     is broken, this lets the rest of the chat pipeline
///     (routing → extract → grounded-answer formatter) be exercised
///     end-to-end with a predictable citation.
///
/// This stub does NOT perform TF-IDF, embedding lookup, or any lexical
/// matching — per the architectural directive that retrieval is a
/// pure generative decision. If you need per-query behavior in tests,
/// use `ScriptedKBExtractor` (below).
public struct StubKBExtractor: KBExtractor {
    private let citation: KBCitation

    public init(citation: KBCitation = .noMatch(runtimeMS: 0)) {
        self.citation = citation
    }

    public func extract(query: String, kb: [KBEntry]) async -> KBCitation {
        citation
    }
}

/// Test-only extractor that maps query substrings to pre-configured
/// citations. First matching rule wins; on no match, returns the
/// fallback (`.noMatch` by default).
public actor ScriptedKBExtractor: KBExtractor {
    public struct Rule: Sendable {
        public let matches: String
        public let citation: KBCitation

        public init(matches: String, citation: KBCitation) {
            self.matches = matches
            self.citation = citation
        }
    }

    private var rules: [Rule] = []
    private let fallback: KBCitation

    public init(fallback: KBCitation = .noMatch(runtimeMS: 0)) {
        self.fallback = fallback
    }

    public func script(_ rule: Rule) {
        rules.append(rule)
    }

    public nonisolated func extract(query: String, kb: [KBEntry]) async -> KBCitation {
        await self.dispatch(query: query)
    }

    private func dispatch(query: String) -> KBCitation {
        for rule in rules where query.lowercased().contains(rule.matches.lowercased()) {
            return rule.citation
        }
        return fallback
    }
}
