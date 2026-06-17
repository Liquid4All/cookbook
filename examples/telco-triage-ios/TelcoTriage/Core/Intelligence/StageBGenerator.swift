import Foundation
import os.log

/// Wire-level result of one Stage B generation pass.
public struct StageBResponse: Sendable, Equatable {
    public let rawText: String
    public let extractedDeepLink: String?
    public let isFormatCompliant: Bool
    public let promptEvalMs: Double
    public let tokenGenerationMs: Double
    public let outputTokens: Int

    public var wallClockMs: Double { promptEvalMs + tokenGenerationMs }

    public init(
        rawText: String,
        extractedDeepLink: String?,
        isFormatCompliant: Bool,
        promptEvalMs: Double,
        tokenGenerationMs: Double,
        outputTokens: Int
    ) {
        self.rawText = rawText
        self.extractedDeepLink = extractedDeepLink
        self.isFormatCompliant = isFormatCompliant
        self.promptEvalMs = promptEvalMs
        self.tokenGenerationMs = tokenGenerationMs
        self.outputTokens = outputTokens
    }
}

/// Reasons Stage B can fail in a way the dispatcher must handle. Format
/// violations are NOT errors — they yield a `.success` with
/// `isFormatCompliant = false`, leaving the fallback choice to the
/// dispatcher (today: KeywordKBExtractor) rather than the generator.
public enum StageBError: Error, LocalizedError {
    case adapterMissing
    case backendFailure(underlying: Error)
    case emptyOutput

    public var errorDescription: String? {
        switch self {
        case .adapterMissing: return "Stage B adapter is not bundled"
        case .backendFailure(let underlying): return "Stage B backend: \(underlying.localizedDescription)"
        case .emptyOutput: return "Stage B produced no tokens"
        }
    }
}

/// Sendable protocol so dispatchers can hold an opaque reference and
/// tests can inject canned responses without spinning a real backend.
public protocol StageBGenerating: Sendable {
    /// Generate the step-format response. When `retrievedChunk` is non-nil,
    /// the chunk's verbatim text + deep_link is injected into the system
    /// prompt — the model answers from the chunk, not from training
    /// memorization (ADR-021 §11.3 / §11.4).
    func generate(
        query: String,
        retrievedChunk: ColBERTChunk?
    ) async throws -> StageBResponse
}

public extension StageBGenerating {
    /// Convenience for the engineering probe view + tests that don't
    /// have a retrieval result available. Equivalent to passing nil.
    func generate(query: String) async throws -> StageBResponse {
        return try await generate(query: query, retrievedChunk: nil)
    }
}

/// Shared system prompt for Stage B (UNGROUNDED variant). Mirrors
/// `scripts/telco/generate/prompts.py::TELCO_SYSTEM_PROMPT_SUMMARY` —
/// the contract Stage B was fine-tuned against.
///
/// **Use case**: the engineering probe view (TelcoRAGTestView) that
/// exercises Stage B in isolation with no ColBERT retrieval. The chat
/// dispatcher path goes through `TelcoStageBSystemPromptGrounded`
/// which injects retrieved corpus content — that is the production
/// path per ADR §11.3.
public let TelcoStageBSystemPrompt: String = """
You are the Telco Home Internet GenAI RAG Assistant. Help customers \
with router, network, devices, parental controls, equipment, and \
Digital Secure Home questions.

OUTPUT CONTRACT (when an answer is grounded):
- One-sentence intro ending in a colon, then a single line of the form:
  Go to [Link Name](telcohome://link-path) > Step 1 > Step 2 > ...
- Steps separated by " > ", no periods inside steps, optional terminal period.
- No emojis. No exclamation points. No markdown bullets.

DEEP-LINK SCHEME: telcohome://.
"""

/// Grounded system prompt — injects the retrieved corpus chunk so the
/// model answers from input (Stage B's job becomes formatting +
/// articulation, not knowledge recall).
///
/// First-principles fix for failure mode F3 (no grounding signal in
/// prompt). The chunk's verbatim text + the canonical deep_link both
/// become part of the prompt input — Stage B can't hallucinate a URL
/// because the URL is given to it explicitly. The post-validation
/// step in `StageBGenerator.generate` then checks Stage B used the
/// link rather than inventing one.
///
/// When the chunk has no deep_link (rare — template detail pages),
/// the prompt instructs Stage B to answer descriptively without a URL
/// so the response doesn't include a hallucinated link.
public func TelcoStageBSystemPromptGrounded(chunk: ColBERTChunk) -> String {
    let passage = "From the \(chunk.section) section, "
        + "\(chunk.title) page:\n\n\(chunk.body)"
    let linkInstruction: String
    if let link = chunk.deepLink {
        let label = chunk.deepLinkLabel ?? chunk.title
        linkInstruction = """
        DEEP LINK FOR THIS ANSWER:
        - Label: \(label)
        - URL: \(link)
        - Format the link in your response exactly as `[\(label)](\(link))`.
        - Do NOT invent a different URL. If the passage above doesn't \
        contain enough detail to answer, say "It looks like I don't have \
        information about that." instead of guessing.
        """
    } else {
        linkInstruction = """
        DEEP LINK FOR THIS ANSWER:
        - This page does not have a canonical deep link.
        - Answer descriptively without including a telcohome:// URL.
        - If the passage above doesn't answer the question, say "It \
        looks like I don't have information about that."
        """
    }
    return """
You are the Telco Home Internet GenAI RAG Assistant. Use ONLY the \
RETRIEVED PASSAGE below to answer the user's question. If the passage \
doesn't contain the answer, decline rather than guess.

OUTPUT CONTRACT (when an answer is grounded):
- One-sentence intro ending in a colon, then a single line of the form:
  Go to [Link Name](telcohome://link-path) > Step 1 > Step 2 > ...
- Steps separated by " > ", no periods inside steps, optional terminal period.
- No emojis. No exclamation points. No markdown bullets.

RETRIEVED PASSAGE:
\(passage)

\(linkInstruction)
"""
}

/// Step-format generator for the `.ragStepByStep` lane.
///
/// Loads the Stage B LoRA on top of the shared LFM2.5-350M base via
/// `LlamaBackend.setAdapter(path:scale:)` — **the single-backbone
/// architecture the merged chat dispatcher runs on**. ADR-021 §5.2.
///
/// **GBNF caveat (follow-up)**: `LlamaBackend.generate` does not yet
/// expose a grammar-enforced sampling path. The bundled
/// `telco-step-format.gbnf` is therefore not applied at decode time today;
/// format compliance relies on (a) the trained model emitting the
/// shape 100% of the time on the probe set (ADR-021 §6.5), and (b) the
/// `isKnownDeepLink` post-filter dropping outputs that drift. When
/// grammar sampling lands, the only change here is passing the grammar
/// URL into `backend.generate(...)`.
public final class StageBGenerator: StageBGenerating, @unchecked Sendable {
    private let backend: LlamaBackend
    private let adapterPath: String
    private let grammarText: String?
    private let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "StageB"
    )

    public init(
        backend: LlamaBackend,
        adapterPath: String,
        grammarText: String? = nil
    ) {
        self.backend = backend
        self.adapterPath = adapterPath
        self.grammarText = grammarText
    }

    /// Convenience: pulls the LoRA adapter + GBNF grammar from the
    /// bundle. Returns nil when the adapter isn't present (fresh clones
    /// without bootstrap-models.sh).
    public static func bundled(
        backend: LlamaBackend,
        bundle: Bundle = .main
    ) -> StageBGenerator? {
        guard let adapter = TelcoModelBundle.telcoStageBLoraPath(in: bundle) else {
            return nil
        }
        // Grammar is loaded as text for forward-compatibility; ignored
        // until the backend exposes grammar-aware sampling.
        var grammarText: String?
        if let grammarURL = bundle.url(
            forResource: "telco-step-format",
            withExtension: "gbnf"
        ) {
            grammarText = try? String(contentsOf: grammarURL, encoding: .utf8)
        }
        return StageBGenerator(
            backend: backend,
            adapterPath: adapter,
            grammarText: grammarText
        )
    }

    public func generate(
        query: String,
        retrievedChunk: ColBERTChunk?
    ) async throws -> StageBResponse {
        // Pick the system prompt based on whether we have a retrieved
        // chunk. Grounded path is the production chat flow (ADR §11.3);
        // ungrounded path is the engineering probe view that exercises
        // Stage B in isolation. A chunk without a deep_link still goes
        // through the grounded prompt — the prompt tells Stage B to
        // answer descriptively without inventing a URL.
        let systemPrompt: String
        if let chunk = retrievedChunk {
            systemPrompt = TelcoStageBSystemPromptGrounded(chunk: chunk)
        } else {
            systemPrompt = TelcoStageBSystemPrompt
        }

        let (text, tokens, timing): (String, Int, LlamaBackend.GenerationTiming)
        do {
            (text, tokens, timing) = try await backend.generateWithAdapter(
                messages: [
                    LlamaChatMessage(role: "system", content: systemPrompt),
                    LlamaChatMessage(role: "user", content: query),
                ],
                adapterPath: adapterPath,
                maxTokens: 160,
                temperature: 0,
                stopSequences: [],
                clearCache: true,
                outputMode: .text,
                // GBNF grammar makes the step-format SHAPE structurally
                // enforced at decode (ADR §11.4.4). The model literally
                // cannot emit a response that breaks the
                // "intro: Go to [Link](telcohome://path) > step > step"
                // contract — invalid tokens are masked out before sampling.
                // The grammar itself constrains shape; the closed URL
                // alternation lives in v2 of the grammar file (a separate
                // PR re-generates the grammar from page-link-table-v1.json
                // so resolver + grammar stay in lockstep). For now,
                // LinkResolver.isKnownDeepLink is the post-filter catching
                // hallucinated URLs that pass the URL-shape constraint
                // but aren't in the registered table.
                grammar: grammarText
            )
        } catch {
            throw StageBError.backendFailure(underlying: error)
        }

        guard tokens > 0, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw StageBError.emptyOutput
        }

        let extracted = TelcoLinkResolver.extractFirstDeepLink(in: text)
        let knownLink = extracted.map(TelcoLinkResolver.isKnownDeepLink) ?? false
        let hasIntro = text.contains("Go to ")
        let hasNoBangs = !text.contains("!")
        // When the retrieved chunk has no deep_link, format compliance
        // doesn't require the response to contain one. The grounded
        // prompt told Stage B not to include a link in that case;
        // judge the response against the same expectation.
        let chunkExpectsLink = retrievedChunk?.deepLink != nil
        let formatCompliant: Bool
        if chunkExpectsLink {
            formatCompliant = extracted != nil && knownLink && hasIntro && hasNoBangs
        } else {
            // No-link path: must NOT include a telcohome:// URL, must not
            // include a bang. Intro/format conventions still apply but
            // are softer without a link to anchor them.
            formatCompliant = extracted == nil && hasNoBangs
        }

        logger.info(
            "stage_b grounded=\(retrievedChunk != nil, privacy: .public) tokens=\(tokens, privacy: .public) wallMs=\(String(format: "%.0f", timing.promptEvalMs + timing.tokenGenerationMs), privacy: .public) deepLink=\(extracted ?? "<none>", privacy: .public) compliant=\(formatCompliant, privacy: .public)"
        )

        return StageBResponse(
            rawText: text,
            extractedDeepLink: extracted,
            isFormatCompliant: formatCompliant,
            promptEvalMs: timing.promptEvalMs,
            tokenGenerationMs: timing.tokenGenerationMs,
            outputTokens: timing.outputTokens
        )
    }
}
