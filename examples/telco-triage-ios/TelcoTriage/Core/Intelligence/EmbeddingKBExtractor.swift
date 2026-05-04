import Foundation

/// LFM-embedding RAG extractor — replaces the brittle 32-class
/// generative LoRA / classifier-head approach with cosine retrieval
/// over LFM-encoded KB vectors.
///
/// Why this exists: the previous extractor was a classifier whose
/// output space was fixed at training time. Adding a KB entry, or
/// retraining any other adapter, broke its alignment. Embedding RAG
/// is the textbook retrieval pattern — KB changes are a runtime
/// rebuild (no training), and "no match" is a threshold check (no
/// forced argmax onto a wrong article).
///
/// First-pass behaviour: if the index isn't yet ready (the build runs
/// in the background at app launch), the extractor returns
/// `KBCitation.noMatch` for that turn. The chat UI degrades to "no
/// matching article" gracefully — never freezing on a 5-second build.
struct EmbeddingKBExtractor: KBExtractor {
    /// Sentinel `entryId` returned while the embedding index is still
    /// building. The chat surface keys on this to render a "warming
    /// up" banner instead of the standard noMatch decline so users
    /// can distinguish "wait a moment" from "your question has no
    /// answer".
    static let loadingEntryID = "__index_loading__"

    private let index: LFMKBEmbeddingIndex
    /// Cosine threshold below which the answer is "no match". 0.50 is
    /// a conservative starting point for mean-pooled LFM2.5-350M
    /// embeddings on this KB; tune against the audit set.
    private let threshold: Double

    init(index: LFMKBEmbeddingIndex, threshold: Double = 0.50) {
        self.index = index
        self.threshold = threshold
    }

    func extract(query: String, kb: [KBEntry]) async -> KBCitation {
        let start = Date()

        let isReady = await index.isReady
        guard isReady else {
            AppLog.intelligence.info("KB index not ready yet — returning loading sentinel")
            return KBCitation(
                entryId: Self.loadingEntryID,
                passage: "",
                confidence: 0,
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        let matches: [LFMKBEmbeddingIndex.Match]
        do {
            matches = try await index.search(query: query, topK: 1, threshold: threshold)
        } catch {
            AppLog.intelligence.error("KB embedding search failed: \(error.localizedDescription, privacy: .public)")
            return .noMatch(runtimeMS: Int(Date().timeIntervalSince(start) * 1000))
        }

        guard let best = matches.first,
              let entry = kb.first(where: { $0.id == best.entryId })
        else {
            return .noMatch(runtimeMS: Int(Date().timeIntervalSince(start) * 1000))
        }

        // Passage = first paragraph of the answer. Keeps the citation
        // short for the trace and lets the generative path produce the
        // full grounded answer separately.
        let passage = Self.firstParagraph(of: entry.answer)
        return KBCitation(
            entryId: entry.id,
            passage: passage,
            confidence: best.similarity,
            runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }

    private static func firstParagraph(of text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let r = trimmed.range(of: "\n\n") {
            return String(trimmed[..<r.lowerBound])
        }
        return trimmed
    }
}
