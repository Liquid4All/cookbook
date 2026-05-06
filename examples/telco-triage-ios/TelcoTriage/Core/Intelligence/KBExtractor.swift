import Foundation

/// Retrieval primitive for the local support KB.
///
/// Implementations can be lexical, embedding-backed, or generative; the
/// chat flow depends only on the returned citation. The current cookbook
/// app wires `KeywordKBExtractor` for the small curated KB, then lets the
/// resident LFM synthesize the answer from the selected article. Larger
/// customer corpora can swap in `EmbeddingKBExtractor` or `LFMKBExtractor`
/// without changing the chat pipeline.
public protocol KBExtractor: Sendable {
    func extract(query: String, kb: [KBEntry]) async -> KBCitation
}

/// Single extraction result. `entryId == KBCitation.noMatchID` is the
    /// sentinel for "no KB entry fits" — callers should route the query
    /// to `.outOfScope` or a soft decline in that case. `passage` is an
    /// excerpt from the selected entry's `answer`; it is NOT a summary
    /// or paraphrase.
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
