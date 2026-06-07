import Foundation

/// One canonical RAG unit, as produced by the Step 1–4a Python
/// canonicalisation pipeline (`scripts/vz/canonicalize.py`).
///
/// **Source of truth**: `data/finetune/vz-home-internet/rag_units.json`.
/// Bundled in the iOS app as `Resources/rag-units-v1.json`. The Swift
/// struct mirrors the JSON shape exactly so we can `JSONDecoder` it
/// without a transform layer.
///
/// The composer (Step 5) treats every field except `body` as part of
/// the response contract:
///
/// * `canonicalURL` — the only `vzhome://` URL the composer is allowed
///   to render for this unit.
/// * `citationLabel` — visible link label in the rendered text and the
///   citation chip.
/// * `steps` — exact step chain, joined by `" > "`. Body content is
///   never opened to mine new steps.
/// * `actionAffordance` — drives the route-classification side; see
///   `VerizonChatDispatcher.deriveRoute`. The `ToolRegistry` gate is
///   the FINAL arbiter for whether confirmation is shown (guardrail
///   #3 in the Step 6 plan).
public struct RAGUnit: Codable, Sendable, Equatable {
    public let pageID: String
    public let taskID: String?
    public let title: String
    public let section: String
    public let level: Int
    public let parentPageID: String?
    public let linkID: String
    public let canonicalURL: String
    public let aliases: [String]
    public let steps: [String]
    public let body: String
    public let sourceDoc: String
    public let citationLabel: String?
    public let actionAffordance: String?

    enum CodingKeys: String, CodingKey {
        case pageID = "page_id"
        case taskID = "task_id"
        case title
        case section
        case level
        case parentPageID = "parent_page_id"
        case linkID = "link_id"
        case canonicalURL = "canonical_url"
        case aliases
        case steps
        case body
        case sourceDoc = "source_doc"
        case citationLabel = "citation_label"
        case actionAffordance = "action_affordance"
    }

    /// The `vzhome://...` URL with the optional `?launchPoint=…` query
    /// stripped. Used by alias / equivalence checks.
    public var canonicalURLBare: String {
        if let q = canonicalURL.firstIndex(of: "?") {
            return String(canonicalURL[canonicalURL.startIndex..<q])
        }
        return canonicalURL
    }

    /// User-facing label used in citation chips and link rendering.
    /// Falls back to title if the upstream record forgot to set it.
    public var displayLabel: String {
        if let label = citationLabel, !label.isEmpty { return label }
        return title
    }

    /// Does the query refer to this unit's task objective, as opposed
    /// to one navigation step inside that task? This lets the router
    /// offer a tool only when the user is asking about the task itself
    /// ("how do I restart my router?"), not when they ask about a
    /// sub-step ("where is the Equipment tile?").
    public func queryTargetsTaskObjective(
        _ query: String,
        minPrecision: Double = 0.5
    ) -> Bool {
        let qTokens = Set(BM25Tokenizer.tokenize(query))
        guard !qTokens.isEmpty else { return false }
        let overlap = qTokens.intersection(taskObjectiveTokens)
        return Double(overlap.count) / Double(qTokens.count) >= minPrecision
    }

    private var taskObjectiveTokens: Set<String> {
        var tokens = Set(BM25Tokenizer.tokenize(title))
        if let taskID, !taskID.isEmpty {
            for token in BM25Tokenizer.tokenize(taskID) { tokens.insert(token) }
        }
        if let citationLabel, !citationLabel.isEmpty {
            for token in BM25Tokenizer.tokenize(citationLabel) { tokens.insert(token) }
        }
        for alias in aliases {
            for token in BM25Tokenizer.tokenize(alias) { tokens.insert(token) }
        }
        return tokens
    }

    /// Fraction of the query's content tokens that appear in this unit's
    /// "aboutness" fields (title, task id, citation label, aliases, steps,
    /// section). A length-normalised `[0, 1]` lexical-coverage measure.
    ///
    /// Used by `TelcoPolicyEngine` as the corroboration veto for the
    /// out-of-scope scope-risk signal (ADR-029 §3): a unit that lexically
    /// covers a majority of the query's content words is a real local answer
    /// and overrides the scope decline. Body prose is intentionally excluded —
    /// a term mentioned in passing in the body does not make the page *about*
    /// the query, and including it would let tangential pages spuriously clear
    /// the veto.
    ///
    /// This is **not** a universal grounding floor: measured holdout
    /// distributions show many genuine terse support turns with coverage well
    /// below 0.5, so the policy engine only consults it in conjunction with the
    /// out-of-scope lexicon, never as a blanket gate.
    public func groundingCoverage(forQuery query: String) -> Double {
        let qTokens = Set(BM25Tokenizer.tokenize(query))
        guard !qTokens.isEmpty else { return 0.0 }
        var bag = taskObjectiveTokens
        for step in steps {
            for token in BM25Tokenizer.tokenize(step) { bag.insert(token) }
        }
        for token in BM25Tokenizer.tokenize(section) { bag.insert(token) }
        return Double(qTokens.intersection(bag).count) / Double(qTokens.count)
    }
}
