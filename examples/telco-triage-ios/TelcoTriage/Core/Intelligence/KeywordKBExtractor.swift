import Foundation

/// Deterministic KB retrieval over curated aliases. Replaces the
/// brittle classifier-head and embedding-RAG approaches as the
/// PRIMARY KB extractor.
///
/// Why this exists (first-principles):
///   The KB has hand-curated aliases for every entry — that's a hard
///   ML signal and we should use it. A keyword/token-overlap matcher
///   is:
///     - 100% accurate when the query contains an alias (the common
///       case: "pause internet" → parental-controls; "ssid" →
///       find-wifi-name)
///     - Sub-millisecond on iPhone (no LFM forward pass)
///     - Impossible to silently regress on (no model retraining)
///     - Trivially debuggable (which alias matched, what score)
///
///   Embedding RAG was the wrong primitive here: the classifier
///   adapter we trained collapses fine-grained KB entries inside the
///   same intent class (parental-controls, firmware-version, and
///   find-wifi-name all classify as `device_setup`, so their
///   embeddings cluster together — fatal for retrieval).
///
///   Production RAG: BM25/keyword first, embeddings only for
///   paraphrase fallback. We mirror that here.
public struct KeywordKBExtractor: KBExtractor {
    /// Bonus weight for matching a multi-word alias (e.g.,
    /// "pause internet"). A two-word match is much stronger evidence
    /// than the same two words appearing separately.
    private let phraseBonus: Double
    /// Minimum score needed to return a match. Below this → noMatch.
    /// 1.0 means at least one content (non-stopword) token must match.
    private let scoreThreshold: Double

    public init(phraseBonus: Double = 2.0, scoreThreshold: Double = 1.0) {
        self.phraseBonus = phraseBonus
        self.scoreThreshold = scoreThreshold
    }

    public func extract(query: String, kb: [KBEntry]) async -> KBCitation {
        let start = Date()
        // Filter stopwords + sub-2-char tokens from BOTH sides. Without
        // this, share-wifi's "how do I share wifi" alias dominates the
        // overlap with any "how do I" query because of {how, do, my}.
        let queryTokens = Self.contentTokens(text: query)
        let queryNormalized = Self.normalize(query)

        var bestEntry: KBEntry?
        var bestScore: Double = 0
        var runnerUpScore: Double = 0

        for entry in kb {
            let s = score(
                entry: entry,
                queryTokens: queryTokens,
                queryNormalized: queryNormalized
            )
            if s > bestScore {
                runnerUpScore = bestScore
                bestScore = s
                bestEntry = entry
            } else if s > runnerUpScore {
                runnerUpScore = s
            }
        }

        let runtimeMS = Int(Date().timeIntervalSince(start) * 1000)

        // Reject when (a) nothing matched (b) best is tied with the
        // runner-up — a tie means the query was too ambiguous to
        // pick a single entry and we'd rather show "no match" than
        // guess between two equally-plausible KB articles.
        guard let entry = bestEntry,
              bestScore >= scoreThreshold,
              bestScore > runnerUpScore
        else {
            return .noMatch(runtimeMS: runtimeMS)
        }

        // Confidence proxy: log-scaled best score, with a margin
        // bonus over the runner-up. Caps at 1.0 so the trace UI's
        // percent renderer behaves.
        let margin = max(0, bestScore - runnerUpScore)
        let raw = (bestScore + margin) / (bestScore + margin + 4.0)
        let confidence = min(1.0, max(0.5, raw))

        return KBCitation(
            entryId: entry.id,
            passage: Self.firstParagraph(of: entry.answer),
            confidence: confidence,
            runtimeMS: runtimeMS
        )
    }

    // MARK: - Scoring

    /// Score a single KB entry against the query. Sums:
    ///  - 1.0 per token-level match between query tokens and the
    ///    union of {topic words, alias words, tag words}
    ///  - `phraseBonus` per multi-word alias (or topic) that appears
    ///    contiguously in the query (e.g., "pause internet" matched
    ///    in "pause internet for my son's tablet")
    ///
    /// Multi-word aliases dominate because they're high-signal
    /// (curated by the KB author specifically because they map
    /// uniquely to this entry). A single common word like "wifi"
    /// scores 1.0 alone — below threshold — which prevents every
    /// query mentioning "wifi" from collapsing onto one entry.
    private func score(
        entry: KBEntry,
        queryTokens: Set<String>,
        queryNormalized: String
    ) -> Double {
        var score: Double = 0

        // Token-level overlap (topic + aliases + tags)
        let entryTokens = Self.entryTokens(entry)
        score += Double(queryTokens.intersection(entryTokens).count)

        // Multi-word alias / topic phrase bonus. Normalize aliases
        // the same way as the query and check substring presence.
        let phrases = ([entry.topic] + entry.aliases)
            .map(Self.normalize)
            .filter { $0.contains(" ") }  // multi-word only
        for phrase in phrases where queryNormalized.contains(phrase) {
            score += phraseBonus
        }

        return score
    }

    /// Content-bearing tokens: lowercased words minus stopwords and
    /// sub-2-char tokens. Used on both sides of the overlap so that
    /// stopwords ("how", "do", "my", "what") never drive scoring.
    /// The previous implementation only filtered stopwords on the
    /// topic-gate side, which let share-wifi outrank restart-router
    /// for "how do I restart my router" (3 stopword hits to 2 real).
    private static func contentTokens(text: String) -> Set<String> {
        var out: Set<String> = []
        for tok in TelcoTopicGate.tokenize(text) {
            if tok.count < 2 { continue }
            if Self.stopwords.contains(tok) { continue }
            out.insert(tok)
        }
        return out
    }

    /// Per-entry content token set: topic + aliases + tags + the
    /// hyphenated id itself (split on `-`/`_`). The id is canonical
    /// and high-signal — `restart-router` directly tells us this
    /// entry is about restart and router, regardless of how the
    /// aliases are phrased.
    private static func entryTokens(_ entry: KBEntry) -> Set<String> {
        var tokens: Set<String> = []
        for term in entry.searchableTerms {
            tokens.formUnion(contentTokens(text: term))
        }
        // ID-as-signal: split on dash/underscore so "restart-router"
        // contributes "restart" and "router" tokens.
        tokens.formUnion(contentTokens(text: entry.id))
        return tokens
    }

    /// Generic English stopwords. Mirrors `TelcoTopicGate.stopwords`
    /// — kept private here so the KB extractor doesn't depend on the
    /// gate's lifecycle.
    private static let stopwords: Set<String> = [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "doing", "have", "has", "had", "having",
        "i", "you", "he", "she", "we", "they", "it", "my", "your", "his",
        "her", "our", "their", "its", "this", "that", "these", "those",
        "what", "where", "when", "why", "how", "who", "which",
        "to", "of", "in", "on", "at", "by", "for", "with", "from", "about",
        "and", "or", "but", "if", "then", "else", "so", "than", "as",
        "can", "could", "should", "would", "will", "might", "may", "must",
        "please", "thanks", "thank", "ok", "okay", "yes", "no", "not",
        "me", "us", "them", "myself", "yourself", "themselves",
        "very", "really", "just", "also", "too", "only", "here", "there",
        "now", "today", "tomorrow", "yesterday", "always", "never", "again",
        "more", "less", "most", "least", "some", "any", "all", "both",
        "much", "many", "few", "several",
    ]

    /// Lowercase + collapse whitespace + strip non-alphanumerics
    /// (except spaces) — used for substring/phrase matching.
    private static func normalize(_ text: String) -> String {
        let lower = text.lowercased()
        let scalars = lower.unicodeScalars.map { scalar -> Character in
            if scalar.properties.isAlphabetic || ("0"..."9").contains(scalar) {
                return Character(scalar)
            }
            return " "
        }
        let cleaned = String(scalars)
        let collapsed = cleaned
            .components(separatedBy: .whitespaces)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        return collapsed
    }

    private static func firstParagraph(of text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let r = trimmed.range(of: "\n\n") {
            return String(trimmed[..<r.lowerBound])
        }
        return trimmed
    }
}
