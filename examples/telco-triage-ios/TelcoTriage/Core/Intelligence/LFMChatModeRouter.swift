import Foundation

/// Production `ChatModeRouter` backed by LFM2.5-350M plus the
/// `chat-mode-router-v2` LoRA adapter.
///
/// This adapter is trained on the user-intent boundary that caused the
/// original telco demo failures: how/why support questions should go to
/// local RAG, while imperative requests should go to the tool selector.
/// Examples from the parent eval set include "why is my wifi slow" and
/// "how do I restart my router" as `kb_question`, while "restart my
/// router" and "run diagnostics on my home network" stay `tool_action`.
///
/// Inference dispatches through `AdapterInferenceBackend` so the
/// router works identically against on-device llama.cpp, a sidecar
/// backend, or any future transport. `adapterPath: ""` is reserved for
/// degraded tests or missing-artifact mode; production bundles pass the
/// trained adapter path.
///
/// Errors are caught and surfaced as `.outOfScope` with confidence
/// 0.0 — a fail-safe default that keeps the chat flow alive even when
/// inference fails. Never crashes the chat flow.
public struct LFMChatModeRouter: ChatModeRouter {
    private let backend: AdapterInferenceBackend
    private let adapterPath: String
    private let maxTokens: Int

    /// - Parameter adapterPath: path to the ChatModeRouter LoRA GGUF
    ///   (e.g. `chat-mode-router-v2.gguf`). Pass the empty string to
    ///   run against the base LFM2.5-350M only — accuracy drops to
    ///   ~23 % without the LoRA per the Phase B validation run.
    public init(
        backend: AdapterInferenceBackend,
        adapterPath: String,
        maxTokens: Int = 64
    ) {
        self.backend = backend
        self.adapterPath = adapterPath
        // 64 is sized for the JSON schema below: 4-mode enum + short
        // reasoning + confidence. 40 tokens is the measured upper
        // bound for well-formed output plus margin.
        self.maxTokens = maxTokens
    }

    public func classify(query: String) async -> ChatModePrediction {
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
            AppLog.intelligence.error("chat mode router backend failed: \(error.localizedDescription, privacy: .public)")
            return ChatModePrediction(
                mode: .outOfScope,
                confidence: 0.0,
                reasoning: "inference error — defaulting to out-of-scope",
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        guard let parsed = Self.parseAssistantJSON(raw) else {
            let preview = raw.prefix(200).replacingOccurrences(of: "\n", with: "⏎")
            AppLog.intelligence.error("chat mode router produced unparseable output (len=\(raw.count, privacy: .public)) preview=\(preview, privacy: .private)")
            return ChatModePrediction(
                mode: .outOfScope,
                confidence: 0.0,
                reasoning: "unparseable model output",
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        return ChatModePrediction(
            mode: parsed.mode,
            confidence: parsed.confidence,
            reasoning: parsed.reasoning,
            runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }

    // MARK: - Prompt

    /// Short, schema-locked prompt. The 4-mode taxonomy is explicit so
    /// the model doesn't invent a 5th class; the JSON example pins the
    /// output format. Examples cover the interesting modality-sensitive
    /// pairs ("how do I restart my router" vs "restart my router") —
    /// the exact distinction a 17-intent classifier can't make.
    static func buildPrompt(query: String) -> String {
        // Critical: the JSON example below uses a neutral mode value
        // (`<MODE>`) and a generic reasoning placeholder. If the example
        // uses a specific mode value like "kb_question", LFM2.5-350M's
        // few-shot pattern copies the example verbatim for every query
        // (measured: 26.7% accuracy with `kb_question` as the example).
        // Schema-only examples force the model to actually classify.
        return """
        Classify this telco home internet customer support message into exactly one mode.

        Modes:
        - kb_question: the user is asking for information or how-to. Answered from the knowledge base.
          Examples: "how do I restart my router", "what are my WPS options", "where is the reset button"
        - tool_action: the user wants the assistant to do something right now. Answered by running a tool.
          Examples: "restart my router", "run a speed test", "pause internet for my son"
        - personal_summary: the user wants a summary of their own account, network, plan, or usage.
          Examples: "summarize my home network", "tell me about my plan", "how is my wifi doing"
        - out_of_scope: the query is not about telco home internet support.
          Examples: "what's the weather", "tell me a joke", "who won the game"

        Query: "\(query)"

        Respond with a JSON object containing:
          mode: one of kb_question, tool_action, personal_summary, out_of_scope
          confidence: a number between 0.0 and 1.0
          reasoning: one short sentence justifying the mode

        JSON:
        """
    }

    // MARK: - Parsing

    /// Extracts mode + confidence + reasoning. Reuses the shared JSON
    /// fence-stripper and balanced-brace walker so the same emission
    /// quirks (markdown fences, trailing prose) are tolerated in all
    /// LFM-backed classifiers.
    static func parseAssistantJSON(_ raw: String) -> (
        mode: ChatMode,
        confidence: Double,
        reasoning: String
    )? {
        let trimmed = JSONExtract.stripFences(raw).trimmingCharacters(in: .whitespacesAndNewlines)
        guard let jsonSlice = JSONExtract.firstJSONObject(in: trimmed),
              let data = jsonSlice.data(using: .utf8),
              let any = try? JSONSerialization.jsonObject(with: data),
              let dict = any as? [String: Any],
              let modeString = dict["mode"] as? String,
              let mode = ChatMode(rawValue: modeString)
        else { return nil }

        let confidence = (dict["confidence"] as? Double)
            ?? (dict["confidence"] as? NSNumber)?.doubleValue
            ?? 0.0
        let reasoning = (dict["reasoning"] as? String) ?? ""

        return (mode, confidence, reasoning)
    }
}
