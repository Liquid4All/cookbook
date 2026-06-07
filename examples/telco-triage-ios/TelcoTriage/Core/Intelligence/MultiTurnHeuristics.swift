import Foundation

/// Swift port of the Step 5b pure-function dispatcher heuristics
/// (`scripts/vz/eval/multi_turn_acceptance.py`).
///
/// These six functions are the spec for `VerizonChatDispatcher`'s
/// multi-turn behaviour:
///
/// * `isAffirmative(_:)`            — bare "yes" / "ok" / "do it" detector.
/// * `isDidntWork(_:)`              — "tried that, still not working" family.
/// * `hasTopicSwitchPrefix(_:)`     — "actually", "instead", "never mind", "wait".
/// * `isShortFollowup(_:)`          — bare wh-word OR anaphoric pronoun /
///                                    slot-accrual prepositional phrase with
///                                    ≤ 2 content tokens.
/// * `isCrossSectionShift(_:_:)`    — prior vs new `page_id` in different
///                                    top-level section (cheap topic-change proxy).
/// * `inferQueryMood(_:)`           — re-export of the four-bucket query mood
///                                    used by the retriever, kept here so the
///                                    dispatcher and the retriever stay
///                                    parameter-compatible.
///
/// All six are pure and value-typed — no instance state, no globals,
/// no side effects. Tests live in
/// `TelcoTriageTests/MultiTurnHeuristicsTests.swift`.
///
/// Composer stays stateless per the locked Step 6 guardrails. State lives in
/// `ConversationState`; these helpers feed the dispatcher's decisions on
/// when to reuse prior page, when to fire pending tools, when to escalate
/// frustration, and when to clear stale context.

// MARK: - Affirmative / didn't-work / topic-switch detectors

/// True for bare-affirmative phrasings. Intentionally narrow: phrases
/// like "yes please restart" do NOT qualify because they carry their own
/// new directive — those route through normal retrieval.
public func isAffirmative(_ text: String) -> Bool {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return false }
    // Strip a single trailing `.` or `!`.
    let stripped = trimmed.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: ".!"))
    let normalised = stripped.trimmingCharacters(in: .whitespaces)
    return MultiTurnHeuristicsConstants.affirmativeBareForms.contains(normalised)
}

/// True when the user is reporting that a prior suggestion failed. Increments
/// `ConversationState.didntWorkCount`. Second strike escalates to live agent.
public func isDidntWork(_ text: String) -> Bool {
    let lower = text.lowercased()
    for pattern in MultiTurnHeuristicsConstants.didntWorkPatterns {
        if pattern.firstMatch(in: lower, range: NSRange(lower.startIndex..., in: lower)) != nil {
            return true
        }
    }
    return false
}

/// True when the query opens with a "you know what, change topic" prefix.
/// The dispatcher pairs this with the new top-1 evidence ≠ pending evidence
/// rule to decide to clear pending without firing.
public func hasTopicSwitchPrefix(_ text: String) -> Bool {
    let lower = text.lowercased()
    return MultiTurnHeuristicsConstants.topicSwitchPrefix.firstMatch(
        in: lower, range: NSRange(lower.startIndex..., in: lower)
    ) != nil
}

// MARK: - Short-followup detector

/// Dispatcher-level short-followup detector. True when the query is too
/// sparse to retrieve fresh evidence and clearly refers to prior context.
///
/// Three patterns qualify (the caller must additionally check that
/// `prior_page_id` is set on the conversation state):
///
/// 1. Bare wh-word — `"How?"`, `"Why?"`, `"Where?"`, `"What?"`.
/// 2. Anaphoric pronoun + ≤ ``maxContentTokens`` content tokens — e.g.
///    `"how do I turn it off?"`.
/// 3. Slot-accrual preposition prefix + ≤ ``maxContentTokens`` content
///    tokens — e.g. `"for my son's tablet"`, `"with the extender"`.
///
/// Empty content tokens after stopword/wh-word stripping also qualify.
public func isShortFollowup(_ text: String, maxContentTokens: Int = 2) -> Bool {
    let lower = text.lowercased()
    let range = NSRange(lower.startIndex..., in: lower)
    if MultiTurnHeuristicsConstants.bareWhQuery.firstMatch(in: lower, range: range) != nil {
        return true
    }
    let hasPronoun = MultiTurnHeuristicsConstants.anaphoricPronoun.firstMatch(
        in: lower, range: range
    ) != nil
    let hasSlotPrefix = MultiTurnHeuristicsConstants.slotPrefix.firstMatch(
        in: lower, range: range
    ) != nil
    let contentTokens = MultiTurnHeuristicsTokenizer.contentTokens(text)
    let n = contentTokens.count
    if n == 0 { return true }
    if hasPronoun && n <= maxContentTokens { return true }
    if hasSlotPrefix && n <= maxContentTokens { return true }
    return false
}

// MARK: - Cross-section topic shift

/// True when the new page is in a different top-level section (first two
/// digits of `page_id`) than the prior page. Cheap proxy for "the
/// dispatcher recognized a topic change". Used to set
/// `TurnTelemetry.clearedPriorContext` on the normal path even when no
/// pending tool was involved.
public func isCrossSectionShift(priorPageID: String?, newPageID: String?) -> Bool {
    guard let prior = priorPageID, let new = newPageID else { return false }
    guard let pd = prior.split(separator: ".").first,
          let nd = new.split(separator: ".").first else { return false }
    return pd != nd
}

// MARK: - Internals (constants + tokenizer)

/// Lazily-compiled regex + bare-string tables. Internal — public callers
/// go through the top-level functions above.
enum MultiTurnHeuristicsConstants {
    /// Bare affirmative forms — exact match after lowercase + trim of
    /// trailing `.`/`!`. Intentionally tight so "yes please do it"
    /// does NOT match (that one carries its own directive).
    static let affirmativeBareForms: Set<String> = [
        "y", "yes", "yep", "yeah", "yup",
        "ok", "okay", "sure",
        "do it", "go ahead", "confirm", "please do",
    ]

    /// Five regexes that capture the "I tried that, it didn't work"
    /// family. The dispatcher only treats `is_didnt_work=true` as a
    /// frustration signal when a pending tool exists — without one,
    /// it's just a complaint and the retriever still gets a fresh pass.
    static let didntWorkPatterns: [NSRegularExpression] = {
        [
            #"\b(didn'?t|did not) work\b"#,
            #"\b(still|even) (not working|broken|same issue|no luck)\b"#,
            #"\b(tried that|already tried)\b"#,
            #"\bstill not working\b"#,
            #"\bno luck\b"#,
        ].map { try! NSRegularExpression(pattern: $0, options: [.caseInsensitive]) }
    }()

    static let topicSwitchPrefix: NSRegularExpression = {
        try! NSRegularExpression(
            pattern:
                #"^\s*(actually|instead|never ?mind|wait|forget|no,?\s+show|no,?\s+do|let'?s\s+(switch|do)|on second thought|change of plans|hold on)\b"#,
            options: [.caseInsensitive]
        )
    }()

    static let anaphoricPronoun: NSRegularExpression = {
        try! NSRegularExpression(
            pattern: #"\b(it|that|this|those|these|them|its)\b"#,
            options: [.caseInsensitive]
        )
    }()

    static let slotPrefix: NSRegularExpression = {
        try! NSRegularExpression(
            pattern: #"^\s*(for|with|at|in|on|from|to|by|about)\s+"#,
            options: [.caseInsensitive]
        )
    }()

    static let bareWhQuery: NSRegularExpression = {
        try! NSRegularExpression(
            pattern: #"^\s*(how|why|where|what|which|when|who)\s*[?.]?\s*$"#,
            options: [.caseInsensitive]
        )
    }()

}

/// Minimal Verizon-domain tokenizer — keeps the dispatcher's
/// short-followup detector self-contained without dragging in the
/// retriever's `tokenize` (which lives in `BM25HierarchyRetriever` and
/// would force a circular import at the Swift level).
///
/// Behaviour: lowercase → strip punctuation → drop stopwords (incl.
/// common wh-words). The token list returned here is intentionally
/// content-only — the dispatcher uses it to count "how much new
/// material is the user introducing this turn?".
enum MultiTurnHeuristicsTokenizer {
    /// Stopwords stripped before counting content tokens. The set
    /// includes the wh-words (`how`, `why`, `where`, `what`, `who`,
    /// `which`, `when`) because the short-followup detector handles
    /// bare wh-word queries via its own regex branch — once that
    /// branch fires, the token-count branch shouldn't double-count.
    static let stopwords: Set<String> = [
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
        "have", "how", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "please",
        "that", "the", "this", "to", "what", "when", "where", "which", "will", "with", "you",
        "your", "u", "ur", "im", "ive", "id", "why", "who", "whose", "whom",
    ]

    static let tokenRegex: NSRegularExpression = {
        try! NSRegularExpression(pattern: #"[A-Za-z0-9]+"#)
    }()

    /// Returns the lowercase content tokens (stopwords stripped, no
    /// stemming). The dispatcher only cares about how many content
    /// words the user typed — it doesn't need full retriever-grade
    /// tokenisation.
    static func contentTokens(_ text: String) -> [String] {
        let lower = text.lowercased()
        var out: [String] = []
        let range = NSRange(lower.startIndex..., in: lower)
        tokenRegex.enumerateMatches(in: lower, range: range) { match, _, _ in
            guard let m = match,
                  let r = Range(m.range, in: lower) else { return }
            let raw = String(lower[r])
            if stopwords.contains(raw) { return }
            if raw.count == 1, !raw.allSatisfy({ $0.isNumber }) { return }
            out.append(raw)
        }
        return out
    }
}
