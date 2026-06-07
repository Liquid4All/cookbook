import Foundation

/// Swift port of `scripts/vz/retriever.py` (Step 4a, post-4c alias
/// improvements). The behaviour MUST match the Python reference
/// byte-for-byte on identical input; a parity test
/// (`BM25HierarchyRetrieverSwiftParityTests`) pins this.
///
/// Public surface (mirrors Python):
///
/// * `BM25Tokenizer.tokenize(_:)`
/// * `inferQueryMood(_:)`
/// * `BM25Hit`
/// * `BM25HierarchyRetriever.init(corpus:)`
/// * `retriever.rank(query:history:k:)`
/// * `retriever.rank(query:context:k:)`
///
/// Internals (private):
///
/// * `TitleAliasRetriever`
/// * `BM25Retriever`
///
/// The retriever does **no** model calls. Tokenisation + IDF + BM25
/// + a small set of hand-tuned priors. Per Step 5 the answer layer
/// will be the composer (also pure); the retriever picks the unit
/// and stops.
public enum BM25Tokenizer {
    /// Same stopword set as `scripts/vz/retriever.py::_STOPWORDS`.
    private static let stopwords: Set<String> = [
        "a", "an", "and", "are", "as", "at", "be", "by", "can",
        "do", "does", "for", "from", "have", "how", "i", "in", "is",
        "it", "its", "me", "my", "of", "on", "or", "please", "that",
        "the", "this", "to", "what", "when", "where", "which", "will",
        "with", "you", "your", "u", "ur", "im", "ive", "id", "ill",
    ]

    /// Light plural / verb-tense normalisation. Matches Python `_stem`.
    static func stem(_ token: String) -> String {
        if token.count >= 7 && token.hasSuffix("ing") {
            return String(token.dropLast(3))
        }
        if token.count >= 6 && token.hasSuffix("ed") {
            return String(token.dropLast(2))
        }
        if token.count >= 5 && token.hasSuffix("s") && !token.hasSuffix("ss") {
            return String(token.dropLast(1))
        }
        return token
    }

    /// Lowercase + digit/letter boundary split + alphanumeric token
    /// extraction + stopword drop + light stem. Mirrors Python
    /// `tokenize()`.
    public static func tokenize(_ text: String) -> [String] {
        let lower = text.lowercased()
        let split = splitDigitLetterBoundaries(lower)
        var out: [String] = []
        var current = ""
        for scalar in split.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) {
                current.unicodeScalars.append(scalar)
            } else {
                if !current.isEmpty {
                    appendIfValid(current, into: &out)
                    current = ""
                }
            }
        }
        if !current.isEmpty {
            appendIfValid(current, into: &out)
        }
        return out
    }

    private static func appendIfValid(_ raw: String, into out: inout [String]) {
        if stopwords.contains(raw) { return }
        if raw.count == 1 && !raw.allSatisfy({ $0.isNumber }) { return }
        out.append(stem(raw))
    }

    /// Insert a space between digit↔letter transitions so `5GHz`,
    /// `2.4GHz`, `GBUsage` tokenise the same as `5 ghz`, etc.
    static func splitDigitLetterBoundaries(_ text: String) -> String {
        var out = ""
        var prev: Character?
        for ch in text {
            if let p = prev {
                let pIsDigit = p.isNumber
                let cIsDigit = ch.isNumber
                let pIsAlpha = p.isLetter
                let cIsAlpha = ch.isLetter
                if (pIsDigit && cIsAlpha) || (pIsAlpha && cIsDigit) {
                    out.append(" ")
                }
            }
            out.append(ch)
            prev = ch
        }
        return out
    }
}

// MARK: - Query mood

public enum QueryMood: String, Sendable, Equatable {
    case actionImperative = "action_imperative"
    case navigateImperative = "navigate_imperative"
    case question
    case statement
}

/// Mirrors Python `infer_query_mood`. Used by the dispatcher (not the
/// retriever directly) to drive composer route + ToolRegistry gating.
public func inferQueryMood(_ query: String) -> QueryMood {
    let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if q.isEmpty { return .statement }
    if q.hasSuffix("?") { return .question }
    for prefix in questionPrefixes {
        if q.hasPrefix(prefix) { return .question }
    }
    for verb in actionVerbs {
        if q == verb || q.hasPrefix(verb + " ") {
            return .actionImperative
        }
    }
    for verb in navigateVerbs {
        if q == verb || q.hasPrefix(verb + " ") {
            return .navigateImperative
        }
    }
    return .statement
}

private let actionVerbs: [String] = [
    "restart", "reboot", "reset", "turn off", "turn on", "block",
    "unblock", "pause", "enable", "disable", "set up", "set", "add",
    "remove", "delete", "share", "change", "rename", "run", "start",
    "stop", "test", "create", "toggle", "power cycle", "cycle",
]

private let navigateVerbs: [String] = [
    "show", "see", "find", "look", "look up", "view", "open", "check",
    "go to", "take me to", "navigate", "list",
]

private let questionPrefixes: [String] = [
    "how", "what", "where", "when", "why", "which", "who",
    "can ", "could ", "do ", "does ", "did ", "is ", "are ",
    "should ", "would ", "will ",
]

// MARK: - History page hints

/// One conversation turn — minimal shape the retriever needs.
public struct ConversationTurnSnippet: Sendable, Equatable {
    public let role: String  // "USER" or "ASSISTANT"
    public let body: String

    public init(role: String, body: String) {
        self.role = role
        self.body = body
    }
}

/// Mirrors Python `extract_history_page_hints`. Mines prior assistant
/// turns for `vzhome://link-id` deep links and resolves them through
/// `linkIndex` to a set of candidate page IDs.
public func extractHistoryPageHints(
    history: [ConversationTurnSnippet],
    linkIndex: [String: [String]]
) -> Set<String> {
    var hints = Set<String>()
    guard let regex = try? NSRegularExpression(pattern: "vzhome://([A-Za-z0-9\\-]+)", options: []) else {
        return hints
    }
    for turn in history where turn.role == "ASSISTANT" {
        let body = turn.body
        let ns = body as NSString
        let range = NSRange(location: 0, length: ns.length)
        regex.enumerateMatches(in: body, options: [], range: range) { match, _, _ in
            guard let m = match, m.numberOfRanges > 1 else { return }
            let linkID = ns.substring(with: m.range(at: 1)).lowercased()
            if let pids = linkIndex[linkID] {
                for pid in pids { hints.insert(pid) }
            }
        }
    }
    return hints
}

// MARK: - Public hit type

public struct BM25Hit: Sendable, Equatable {
    public let pageID: String
    public let score: Double
    public let linkID: String
    public let actionAffordance: String?

    public init(pageID: String, score: Double, linkID: String, actionAffordance: String?) {
        self.pageID = pageID
        self.score = score
        self.linkID = linkID
        self.actionAffordance = actionAffordance
    }
}

// MARK: - Title/alias retriever (internal)

/// Mirrors Python `TitleAliasRetriever`. Phrase-aware overlap on title +
/// aliases only. Used as a precision-heavy second signal that the
/// hierarchy retriever consults for the alias-confirmation boost.
final class TitleAliasRetriever {
    private let units: [String: RAGUnit]
    let unitTokens: [String: Set<String>]
    private let unitTitleTokens: [String: Set<String>]
    private let unitPhraseTokens: [String: [Set<String>]]

    init(units: [String: RAGUnit]) {
        self.units = units
        var unitTok: [String: Set<String>] = [:]
        var titleTok: [String: Set<String>] = [:]
        var phraseTok: [String: [Set<String>]] = [:]
        for (pid, unit) in units {
            let titleSet = Set(BM25Tokenizer.tokenize(unit.title))
            var bag = titleSet
            var phrases: [Set<String>] = []
            if !titleSet.isEmpty { phrases.append(titleSet) }
            for alias in unit.aliases {
                let aliasSet = Set(BM25Tokenizer.tokenize(alias))
                if aliasSet.isEmpty { continue }
                phrases.append(aliasSet)
                for t in aliasSet { bag.insert(t) }
            }
            unitTok[pid] = bag
            titleTok[pid] = titleSet
            phraseTok[pid] = phrases
        }
        self.unitTokens = unitTok
        self.unitTitleTokens = titleTok
        self.unitPhraseTokens = phraseTok
    }

    func bestPhrasePrecision(queryTokens qToks: Set<String>) -> [String: Double] {
        if qToks.isEmpty { return [:] }
        var out: [String: Double] = [:]
        for (pid, phrases) in unitPhraseTokens {
            var best = 0.0
            for phrase in phrases {
                let overlap = qToks.intersection(phrase)
                if overlap.isEmpty { continue }
                best = max(best, Double(overlap.count) / Double(qToks.count))
            }
            if best > 0 { out[pid] = best }
        }
        return out
    }

    func rank(query: String, k: Int = 5) -> [BM25Hit] {
        let qToks = Set(BM25Tokenizer.tokenize(query))
        if qToks.isEmpty { return [] }
        var scored: [(Double, String)] = []
        for (pid, phrases) in unitPhraseTokens {
            var score = 0.0
            for phrase in phrases {
                let overlap = qToks.intersection(phrase)
                if overlap.isEmpty { continue }
                let recall = Double(overlap.count) / Double(phrase.count)
                let precision = Double(overlap.count) / Double(qToks.count)
                score = max(score, (recall + precision) / 2.0)
            }
            if score <= 0 { continue }
            if !qToks.intersection(unitTitleTokens[pid] ?? []).isEmpty {
                score += 0.05
            }
            scored.append((score, pid))
        }
        scored.sort { lhs, rhs in
            if lhs.0 != rhs.0 { return lhs.0 > rhs.0 }
            return lhs.1 < rhs.1
        }
        return scored.prefix(k).map { score, pid -> BM25Hit in
            let u = units[pid]!
            return BM25Hit(pageID: pid, score: score, linkID: u.linkID, actionAffordance: u.actionAffordance)
        }
    }
}

// MARK: - BM25 retriever (internal)

/// Mirrors Python `BM25Retriever`. Title × 3 + aliases × 2 + steps +
/// body field weighting (in token-replication form, same as Python).
final class BM25Retriever {
    private let units: [String: RAGUnit]
    private let k1: Double
    private let b: Double
    private let docTf: [String: [String: Int]]
    private let docLen: [String: Int]
    private let avgdl: Double
    private let idf: [String: Double]

    init(units: [String: RAGUnit], k1: Double = 1.5, b: Double = 0.75) {
        self.units = units
        self.k1 = k1
        self.b = b
        var docTf: [String: [String: Int]] = [:]
        var docLen: [String: Int] = [:]
        for (pid, unit) in units {
            var tokens: [String] = []
            let titleToks = BM25Tokenizer.tokenize(unit.title)
            for _ in 0..<3 { tokens.append(contentsOf: titleToks) }  // title × 3
            for alias in unit.aliases {
                let aToks = BM25Tokenizer.tokenize(alias)
                for _ in 0..<2 { tokens.append(contentsOf: aToks) }  // aliases × 2
            }
            for step in unit.steps {
                tokens.append(contentsOf: BM25Tokenizer.tokenize(step))
            }
            tokens.append(contentsOf: BM25Tokenizer.tokenize(unit.body))
            var tf: [String: Int] = [:]
            for t in tokens { tf[t, default: 0] += 1 }
            docTf[pid] = tf
            docLen[pid] = tokens.count
        }
        self.docTf = docTf
        self.docLen = docLen
        let nDocs = max(1, docLen.count)
        self.avgdl = Double(docLen.values.reduce(0, +)) / Double(nDocs)
        var df: [String: Int] = [:]
        for tf in docTf.values {
            for term in tf.keys { df[term, default: 0] += 1 }
        }
        var idf: [String: Double] = [:]
        // BM25 smoothed: log((N - df + 0.5) / (df + 0.5) + 1).
        // Matches `scripts/vz/retriever.py::BM25Retriever.__post_init__`.
        for (term, termDf) in df {
            let n = Double(nDocs)
            let d = Double(termDf)
            idf[term] = log((n - d + 0.5) / (d + 0.5) + 1.0)
        }
        self.idf = idf
    }

    func rank(query: String, k: Int = 5) -> [BM25Hit] {
        let qTerms = BM25Tokenizer.tokenize(query)
        if qTerms.isEmpty { return [] }
        var scored: [(Double, String)] = []
        for (pid, tf) in docTf {
            let len = max(1, docLen[pid] ?? 1)
            var score = 0.0
            for term in qTerms {
                guard let tfT = tf[term] else { continue }
                let idfV = idf[term] ?? 0.0
                let num = Double(tfT) * (k1 + 1.0)
                let denom = Double(tfT) + k1 * (1.0 - b + b * Double(len) / avgdl)
                score += idfV * num / denom
            }
            if score > 0 { scored.append((score, pid)) }
        }
        scored.sort { lhs, rhs in
            if lhs.0 != rhs.0 { return lhs.0 > rhs.0 }
            return lhs.1 < rhs.1
        }
        return scored.prefix(k).map { score, pid -> BM25Hit in
            let u = units[pid]!
            return BM25Hit(pageID: pid, score: score, linkID: u.linkID, actionAffordance: u.actionAffordance)
        }
    }
}

// MARK: - BM25 + hierarchy / priors

/// Same constants as Python `BM25HierarchyRetriever`.
private let bonusSectionHeadName = 0.6
private let bonusHistoryHint = 1.5
private let bonusHistoryHintParent = 0.6
private let bonusActionToolAlign = 1.5
private let bonusActionAssistAlign = 1.0
private let bonusNavigateViewAlign = 0.3
private let bonusQuestionViewAlign = 0.3
private let bonusAliasConfirmed = 6.0
private let bonusActiveTaskContext = 6.0
private let bonusActiveStepEvidence = 8.0
private let aliasConfirmedMinScore = 0.5
private let aliasContextOverrideMinScore = 0.75
private let activeStepEvidenceMinScore = 0.5
private let shortFollowupMaxTokens = 1

public final class BM25HierarchyRetriever: Sendable {
    private let corpus: RAGUnitCorpus
    private let units: [String: RAGUnit]
    private let bm25: BM25Retriever
    private let alias: TitleAliasRetriever
    private let sectionTokens: [String: Set<String>]
    private let sectionHeads: Set<String>
    private let linkIndexCached: [String: [String]]

    public init(corpus: RAGUnitCorpus) {
        self.corpus = corpus
        var dict: [String: RAGUnit] = [:]
        for u in corpus.allUnits { dict[u.pageID] = u }
        self.units = dict
        self.bm25 = BM25Retriever(units: dict)
        self.alias = TitleAliasRetriever(units: dict)
        var sec: [String: Set<String>] = [:]
        var heads = Set<String>()
        for (pid, unit) in dict {
            sec[pid] = Set(BM25Tokenizer.tokenize(unit.section))
            if pid.hasSuffix(".00") { heads.insert(pid) }
        }
        self.sectionTokens = sec
        self.sectionHeads = heads
        self.linkIndexCached = corpus.linkIndex
    }

    /// Mirrors Python `BM25HierarchyRetriever.rank`. Returns the top-k
    /// hits with priors applied. Deterministic, sub-millisecond.
    public func rank(
        query: String,
        history: [ConversationTurnSnippet] = [],
        k: Int = 5
    ) -> [BM25Hit] {
        rank(query: query, history: history, explicitPageHints: [], k: k)
    }

    /// State-conditioned retrieval for live chat. The current user turn
    /// is not an independent query: when the prior assistant answer cited
    /// a canonical RAG unit, that page is an explicit candidate in the
    /// retrieval distribution for the next turn. This is the structured
    /// version of multi-turn grounding: no phrase-specific continuation
    /// rules, just P(page | query, conversation_state).
    public func rank(
        query: String,
        context: RetrievalContext,
        k: Int = 5
    ) -> [BM25Hit] {
        let priorAssistant = context.priorAssistantText?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let hasPriorAssistant = priorAssistant.map { !$0.isEmpty } ?? false
        let history: [ConversationTurnSnippet] = hasPriorAssistant
            ? [ConversationTurnSnippet(role: "ASSISTANT", body: priorAssistant ?? "")]
            : []
        let explicitHints = hasPriorAssistant
            ? Set([context.priorPageID].compactMap { $0 })
            : []
        return rank(query: query, history: history, explicitPageHints: explicitHints, k: k)
    }

    private func rank(
        query: String,
        history: [ConversationTurnSnippet],
        explicitPageHints: Set<String>,
        k: Int
    ) -> [BM25Hit] {
        let qTerms = BM25Tokenizer.tokenize(query)
        let mood = inferQueryMood(query)
        let minedHistoryHints = extractHistoryPageHints(history: history, linkIndex: linkIndexCached)
        let historyHints = minedHistoryHints.union(explicitPageHints)

        // Pull deep candidate pool from BM25 (k=16 like Python).
        var base = bm25.rank(query: query, k: 16)

        // Short-followup fallback: BM25 returned nothing AND query is
        // very short → lean on history hints.
        if base.isEmpty && qTerms.count <= shortFollowupMaxTokens {
            let fallbackHints = explicitPageHints.isEmpty ? minedHistoryHints : explicitPageHints
            if !fallbackHints.isEmpty {
                let pid = fallbackHints.sorted().first!
                if let u = units[pid] {
                    return [BM25Hit(
                        pageID: pid,
                        score: bonusHistoryHint,
                        linkID: u.linkID,
                        actionAffordance: u.actionAffordance
                    )]
                }
            }
        }
        if base.isEmpty { return [] }

        // Precision-only alias confirmation against one coherent
        // title/alias phrase. This prevents false confidence from
        // stitching query tokens across unrelated aliases on the same page.
        let qToks = Set(qTerms)
        let aliasConfirmed = alias.bestPhrasePrecision(queryTokens: qToks).filter {
            $0.value >= aliasConfirmedMinScore
        }
        let activeStepEvidence = explicitPageHints.reduce(into: [String: Double]()) { out, pid in
            guard let unit = units[pid] else { return }
            let precision = bestStepMatchPrecision(queryTokens: qToks, steps: unit.steps)
            if precision >= activeStepEvidenceMinScore {
                out[pid] = precision
            }
        }
        let hasActiveStepEvidence = !activeStepEvidence.isEmpty
        let hasContextOverrideEvidence = !hasActiveStepEvidence && aliasConfirmed.contains { pid, precision in
            !explicitPageHints.contains(pid) && precision >= aliasContextOverrideMinScore
        }

        // Alias-only winners spliced in (in case BM25 missed them).
        let aliasHits = alias.rank(query: query, k: 5)
        var historyParents = Set<String>()
        for hint in historyHints {
            if let parent = units[hint]?.parentPageID {
                historyParents.insert(parent)
            }
        }
        let bm25PIDs = Set(base.map(\.pageID))
        var aliasOnly: [BM25Hit] = []
        for ah in aliasHits {
            if !bm25PIDs.contains(ah.pageID) && ah.score >= aliasConfirmedMinScore {
                aliasOnly.append(BM25Hit(
                    pageID: ah.pageID,
                    score: 0.0,
                    linkID: ah.linkID,
                    actionAffordance: ah.actionAffordance
                ))
            }
        }

        let baseAndAliasPIDs = bm25PIDs.union(aliasOnly.map(\.pageID))
        var historyOnly: [BM25Hit] = []
        for pid in historyHints.sorted() where !baseAndAliasPIDs.contains(pid) {
            if let u = units[pid] {
                historyOnly.append(BM25Hit(
                    pageID: pid,
                    score: 0.0,
                    linkID: u.linkID,
                    actionAffordance: u.actionAffordance
                ))
            }
        }

        var adjusted: [(Double, String)] = []
        for hit in base + aliasOnly + historyOnly {
            guard let unit = units[hit.pageID] else { continue }
            var score = hit.score
            if aliasConfirmed[hit.pageID] != nil { score += bonusAliasConfirmed }
            let secToks = sectionTokens[hit.pageID] ?? []
            if sectionHeads.contains(hit.pageID) && !qToks.intersection(secToks).isEmpty {
                score += bonusSectionHeadName
            }
            if explicitPageHints.contains(hit.pageID) {
                if activeStepEvidence[hit.pageID] != nil {
                    score += bonusActiveTaskContext + bonusActiveStepEvidence
                } else if !hasContextOverrideEvidence {
                    score += bonusActiveTaskContext
                }
            } else if minedHistoryHints.contains(hit.pageID) {
                score += bonusHistoryHint
            } else if historyParents.contains(hit.pageID) {
                score += bonusHistoryHintParent
            }
            let aff = unit.actionAffordance ?? ""
            switch mood {
            case .actionImperative:
                if aff == "tool_action" { score += bonusActionToolAlign }
                else if aff == "assist" { score += bonusActionAssistAlign }
            case .navigateImperative:
                if aff == "navigate" || aff == "view" { score += bonusNavigateViewAlign }
            case .question:
                if aff == "view" || aff == "navigate" || aff == "assist" { score += bonusQuestionViewAlign }
            case .statement:
                break
            }
            adjusted.append((score, hit.pageID))
        }
        adjusted.sort { lhs, rhs in
            if lhs.0 != rhs.0 { return lhs.0 > rhs.0 }
            return lhs.1 < rhs.1
        }
        return adjusted.prefix(k).compactMap { score, pid -> BM25Hit? in
            guard let u = units[pid] else { return nil }
            return BM25Hit(pageID: pid, score: score, linkID: u.linkID, actionAffordance: u.actionAffordance)
        }
    }

    private func bestStepMatchPrecision(queryTokens: Set<String>, steps: [String]) -> Double {
        guard !queryTokens.isEmpty, !steps.isEmpty else { return 0.0 }
        var best = 0.0
        for step in steps {
            let stepTokens = Set(BM25Tokenizer.tokenize(step))
            if stepTokens.isEmpty { continue }
            let overlap = queryTokens.intersection(stepTokens).count
            if overlap == 0 { continue }
            best = max(best, Double(overlap) / Double(queryTokens.count))
        }
        return best
    }
}

// `Foundation` already vends `log(_:)` via `Darwin`; no explicit
// shim needed.
