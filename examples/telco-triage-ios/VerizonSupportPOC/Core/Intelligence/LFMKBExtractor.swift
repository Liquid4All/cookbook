import Foundation

/// Legacy `KBExtractor` backed by LFM2.5-350M + kb-extractor-v1 LoRA.
///
/// **Note**: this is no longer the primary path. The active KB
/// extractor is `KeywordKBExtractor` — deterministic alias matching
/// over the curated KB. This file is retained for the fallback case
/// where the keyword extractor is unwired or deliberately bypassed.
///
/// Phase C history: the LoRA was trained to memorize a 32-entry KB
/// using the prompt below. Production prompt is ~40 tokens (query
/// only, no inline KB). The training template lived in the upstream
/// generation pipeline; if you re-train this LoRA the prompt and the
/// training script's `build_training_prompt()` must match byte-for-byte.
///
/// The `buildPrompt(query:)` output MUST match the Python training template
/// in `scripts/generate_kb_extract.py::build_training_prompt()` byte-for-byte.
/// Gate 3.7 (prompt sync) — a regression test in
/// `scripts/tests/test_generate_kb_extract.py` pins this invariant.
///
/// Errors and parse failures surface as `KBCitation.noMatch` so the
/// chat flow gracefully degrades to a decline rather than crashing.
public struct LFMKBExtractor: KBExtractor {
    private let backend: AdapterInferenceBackend
    private let adapterPath: String
    private let maxTokens: Int

    public init(backend: AdapterInferenceBackend, adapterPath: String = "", maxTokens: Int = 200) {
        self.backend = backend
        self.adapterPath = adapterPath
        // 200 ≈ entry_id (short) + one full sentence of passage +
        // JSON scaffolding. Measured upper bound from training-eval
        // traces + margin.
        self.maxTokens = maxTokens
    }

    public func extract(query: String, kb: [KBEntry]) async -> KBCitation {
        let start = Date()
        let prompt = Self.buildPrompt(query: query)

        let raw: String
        do {
            raw = try await backend.generate(
                messages: [.user(prompt)],
                adapterPath: adapterPath,
                maxTokens: maxTokens
            )
        } catch {
            AppLog.intelligence.error("kb extractor backend failed: \(error.localizedDescription, privacy: .public)")
            return .noMatch(runtimeMS: Int(Date().timeIntervalSince(start) * 1000))
        }

        guard let parsed = Self.parseAssistantJSON(raw) else {
            let preview = raw.prefix(200).replacingOccurrences(of: "\n", with: "⏎")
            AppLog.intelligence.error("kb extractor produced unparseable output (len=\(raw.count, privacy: .public)) preview=\(preview, privacy: .private)")
            return .noMatch(runtimeMS: Int(Date().timeIntervalSince(start) * 1000))
        }

        // Validate the emitted entry_id against the real KB so a
        // hallucinated id never leaks into the UI as a citation.
        // The `none` sentinel always validates.
        if parsed.entryId == KBCitation.noMatchID {
            return KBCitation(
                entryId: KBCitation.noMatchID,
                passage: "",
                confidence: parsed.confidence,
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }
        guard kb.contains(where: { $0.id == parsed.entryId }) else {
            AppLog.intelligence.warning("kb extractor emitted unknown entry_id: \(parsed.entryId, privacy: .public)")
            return .noMatch(runtimeMS: Int(Date().timeIntervalSince(start) * 1000))
        }

        return KBCitation(
            entryId: parsed.entryId,
            passage: parsed.passage,
            confidence: parsed.confidence,
            runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }

    // MARK: - Prompt

    /// Phase C prompt: ~40 tokens, no inline KB. The LoRA memorizes
    /// the 32-entry KB; the prompt just passes the query.
    ///
    /// BYTE-IDENTITY CONTRACT: this output must match
    /// `scripts/generate_kb_extract.py::build_training_prompt(query)`
    /// exactly. Update both in the same commit or accuracy drops 30-60 pp.
    static func buildPrompt(query: String) -> String {
        """
        You are the telco home internet knowledge base extractor. For the user query below,
        select the most relevant KB entry and return its entry_id + first-sentence
        passage.

        If no KB entry matches the query, return entry_id "none".

        Query: "\(query)"

        JSON:
        """
    }

    // MARK: - Parsing

    static func parseAssistantJSON(_ raw: String) -> (
        entryId: String,
        passage: String,
        confidence: Double
    )? {
        let trimmed = JSONExtract.stripFences(raw).trimmingCharacters(in: .whitespacesAndNewlines)
        guard let jsonSlice = JSONExtract.firstJSONObject(in: trimmed),
              let data = jsonSlice.data(using: .utf8),
              let any = try? JSONSerialization.jsonObject(with: data),
              let dict = any as? [String: Any],
              let entryId = dict["entry_id"] as? String
        else { return nil }

        let passage = (dict["passage"] as? String) ?? ""
        let confidence = (dict["confidence"] as? Double)
            ?? (dict["confidence"] as? NSNumber)?.doubleValue
            ?? 0.0

        return (entryId, passage, confidence)
    }
}
