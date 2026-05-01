import Foundation

/// Generative retrieval primitive. Given the query plus the complete
/// 32-entry knowledge base in-context, the extractor returns exactly
/// one verbatim passage cited to its source entry — or the `none`
/// sentinel when no entry is a good fit.
///
/// There is no lexical retriever, TF-IDF, or embedding step. The LFM
/// performs the retrieval in its forward pass, picking an `entry_id`
/// from a fixed 32-class enum and emitting a verbatim snippet of that
/// entry's `answer`. This matches the generative-retrieval pattern
/// proven in the Spotify demo (semantic-ID beam search over a closed
/// catalog).
///
/// Why a single protocol instead of "retrieve top-K + re-rank":
///   1. The KB is tiny (32 entries). The entire KB fits in the 8K
///      context window of LFM2.5-350M with room to spare.
///   2. Ranking and selection are the same decision — the model
///      doesn't waste tokens on re-ranking.
///   3. Verbatim-passage output makes UI citation trivial: the
///      returned string is rendered as-is, no post-processing.
public protocol KBExtractor: Sendable {
    func extract(query: String, kb: [KBEntry]) async -> KBCitation
}

/// Single extraction result. `entryId == KBCitation.noMatchID` is the
/// sentinel for "no KB entry fits" — callers should route the query
/// to `.outOfScope` or a soft decline in that case. `passage` is the
/// verbatim substring the LFM emitted from the selected entry's
/// `answer`; it is NOT a summary or paraphrase (preserves training
/// distribution and avoids hallucination risk).
public struct KBCitation: Sendable, Equatable {
    /// Sentinel entry id for "no KB entry fits the query." Matches
    /// the training-time convention used by the (future) KB-extract
    /// LoRA. Distinct from an empty string so the parse path can
    /// tell "model emitted `none`" from "model emitted nothing."
    public static let noMatchID = "none"

    public let entryId: String
    /// Reserved for future citation display: render the exact sentence
    /// the extractor chose as an inline quote chip above the assistant
    /// reply. Currently unread by `ChatViewModel` (grounded-QA uses the
    /// full `KBEntry` answer for LFM generation), but kept in the API
    /// so the UI and the eval harness (verbatim-passage verification)
    /// can land without an API break.
    public let passage: String
    public let confidence: Double
    public let runtimeMS: Int

    public init(
        entryId: String,
        passage: String,
        confidence: Double,
        runtimeMS: Int
    ) {
        self.entryId = entryId
        self.passage = passage
        self.confidence = confidence
        self.runtimeMS = runtimeMS
    }

    /// Convenience factory for "the model explicitly declined" or
    /// "no KB entry fits." Keeps the callsite readable.
    public static func noMatch(runtimeMS: Int) -> KBCitation {
        KBCitation(
            entryId: noMatchID,
            passage: "",
            confidence: 0.0,
            runtimeMS: runtimeMS
        )
    }

    /// True when the citation points at a real KB entry.
    public var isMatch: Bool { entryId != Self.noMatchID && !entryId.isEmpty }
}
