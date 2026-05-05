import Foundation

/// Shared bridge that owns the classifier heads, their paired LoRA
/// adapters, and the LlamaBackend reference for embedding extraction.
/// Constructed once at app launch and injected into the classifier-backed
/// router/extractor/selector.
///
/// Architecture: swap classification LoRA (~1ms cached) + backbone
/// forward pass (~30-50ms) + classifier head matmul (<1ms) = ~35-55ms.
/// The LoRA adapter specializes the backbone's hidden states for
/// classification — without it, accuracy drops 30-70pp (Phase 7 eval).
public final class ClassifierBackedBridge: @unchecked Sendable {
    private let backend: LlamaBackend

    /// Classifier heads keyed by task name.
    public let chatModeHead: ClassifierHead
    public let kbEntryHead: ClassifierHead
    public let toolHead: ClassifierHead

    /// Classification LoRA adapter paths — each is paired with its
    /// classifier head and must be applied before embedding extraction.
    public let chatModeClfAdapterPath: String
    public let kbExtractClfAdapterPath: String
    public let toolSelectorClfAdapterPath: String

    public init(
        backend: LlamaBackend,
        chatModeHead: ClassifierHead,
        kbEntryHead: ClassifierHead,
        toolHead: ClassifierHead,
        chatModeClfAdapterPath: String,
        kbExtractClfAdapterPath: String,
        toolSelectorClfAdapterPath: String
    ) {
        self.backend = backend
        self.chatModeHead = chatModeHead
        self.kbEntryHead = kbEntryHead
        self.toolHead = toolHead
        self.chatModeClfAdapterPath = chatModeClfAdapterPath
        self.kbExtractClfAdapterPath = kbExtractClfAdapterPath
        self.toolSelectorClfAdapterPath = toolSelectorClfAdapterPath
    }

    /// Swap the classification LoRA adapter and extract the last-token
    /// hidden state. The adapter specializes the backbone's representations
    /// for the classification task — the classifier head was trained on
    /// these LoRA-modified hidden states.
    ///
    /// CRITICAL: The classifier was trained on RAW query text (no chat
    /// template, no instruction prompt). The training tokenizer call is
    /// `tokenizer(examples["text"])` — plain text with BOS, nothing else.
    /// Passing chat-template-wrapped text produces a constant last-token
    /// hidden state (dominated by template tokens), causing every input
    /// to classify identically.
    func embedQuery(_ query: String, adapterPath: String) async throws -> [Float] {
        try await backend.setAdapter(path: adapterPath, scale: 1.0)
        return try await backend.embeddings(prompt: query, clearCache: true)
    }
}

// MARK: - ChatModeRouter (classifier-backed)

/// Classifier-backed chat mode router. Replaces the generative
/// `LFMChatModeRouter` with a single backbone forward pass + a
/// 4-way linear classifier head.
public struct ClassifierChatModeRouter: ChatModeRouter {
    private let bridge: ClassifierBackedBridge

    public init(bridge: ClassifierBackedBridge) {
        self.bridge = bridge
    }

    public func classify(query: String) async -> ChatModePrediction {
        let start = Date()

        do {
            let hidden = try await bridge.embedQuery(query, adapterPath: bridge.chatModeClfAdapterPath)
            let prediction = bridge.chatModeHead.classify(hidden)

            guard let mode = ChatMode(rawValue: prediction.label) else {
                AppLog.intelligence.warning("classifier head emitted unknown mode: \(prediction.label, privacy: .public)")
                return ChatModePrediction(
                    mode: .outOfScope,
                    confidence: 0.0,
                    reasoning: "classifier: unknown label",
                    runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
                )
            }

            return ChatModePrediction(
                mode: mode,
                confidence: Double(prediction.confidence),
                reasoning: "classifier head (softmax \(String(format: "%.1f", prediction.confidence * 100))%)",
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        } catch {
            AppLog.intelligence.error("classifier chat mode router failed: \(error.localizedDescription, privacy: .public)")
            return ChatModePrediction(
                mode: .outOfScope,
                confidence: 0.0,
                reasoning: "classifier inference error",
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }
    }
}

// MARK: - KBExtractor (classifier-backed)

/// Classifier-backed KB entry selector. The head outputs one of 34
/// labels (33 KB entries + "none"). The passage field is populated
/// from the KB entry's answer (first sentence) since the classifier
/// head can't generate text.
public struct ClassifierKBExtractor: KBExtractor {
    private let bridge: ClassifierBackedBridge

    public init(bridge: ClassifierBackedBridge) {
        self.bridge = bridge
    }

    public func extract(query: String, kb: [KBEntry]) async -> KBCitation {
        let start = Date()

        do {
            let hidden = try await bridge.embedQuery(query, adapterPath: bridge.kbExtractClfAdapterPath)
            let prediction = bridge.kbEntryHead.classify(hidden)

            let runtimeMS = Int(Date().timeIntervalSince(start) * 1000)

            if prediction.label == "none" {
                return KBCitation(
                    entryId: KBCitation.noMatchID,
                    passage: "",
                    confidence: Double(prediction.confidence),
                    runtimeMS: runtimeMS
                )
            }

            // Validate against the live KB
            guard let entry = kb.first(where: { $0.id == prediction.label }) else {
                AppLog.intelligence.warning("classifier KB head emitted unknown entry_id: \(prediction.label, privacy: .public)")
                return .noMatch(runtimeMS: runtimeMS)
            }

            // Extract first sentence from the entry's answer as the passage
            let passage = Self.firstSentence(from: entry.answer)

            return KBCitation(
                entryId: prediction.label,
                passage: passage,
                confidence: Double(prediction.confidence),
                runtimeMS: runtimeMS
            )
        } catch {
            AppLog.intelligence.error("classifier KB extractor failed: \(error.localizedDescription, privacy: .public)")
            return .noMatch(runtimeMS: Int(Date().timeIntervalSince(start) * 1000))
        }
    }

    /// Extract the first sentence from a KB entry answer. Splits on
    /// period/exclamation/question mark followed by a space or newline.
    private static func firstSentence(from answer: String) -> String {
        let stripped = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        // Find first sentence-ending punctuation followed by whitespace
        if let range = stripped.range(of: #"[.!?][\s\n]"#, options: .regularExpression) {
            return String(stripped[stripped.startIndex...range.lowerBound])
        }
        // No sentence break found — return first 200 chars
        return String(stripped.prefix(200))
    }
}

// MARK: - ToolSelector (classifier-backed)

/// Classifier-backed tool selector. The head outputs one of 9 labels
/// (8 tools + "none"). Free-form slots come from the existing query
/// extraction stage after the model has chosen the tool.
public struct ClassifierToolSelector: ToolSelector {
    private let bridge: ClassifierBackedBridge

    public init(bridge: ClassifierBackedBridge) {
        self.bridge = bridge
    }

    public func select(
        query: String,
        extraction: ExtractionResult,
        availableTools: [Tool]
    ) async -> ToolSelection {
        let start = Date()

        do {
            let hidden = try await bridge.embedQuery(query, adapterPath: bridge.toolSelectorClfAdapterPath)
            let prediction = bridge.toolHead.classify(hidden)

            let runtimeMS = Int(Date().timeIntervalSince(start) * 1000)

            if prediction.label == "none" {
                return ToolSelection(
                    intent: nil,
                    confidence: Double(prediction.confidence),
                    reasoning: "classifier: no tool match",
                    runtimeMS: runtimeMS
                )
            }

            guard let intent = ToolIntent(toolID: prediction.label) else {
                AppLog.intelligence.warning("classifier tool head emitted unknown tool_id: \(prediction.label, privacy: .public)")
                return ToolSelection(
                    intent: nil,
                    confidence: 0,
                    reasoning: "classifier: unknown tool",
                    runtimeMS: runtimeMS
                )
            }

            return ToolSelection(
                intent: intent,
                confidence: Double(prediction.confidence),
                arguments: TelcoDecisionRouter.arguments(for: intent, extraction: extraction),
                reasoning: "classifier head (softmax \(String(format: "%.1f", prediction.confidence * 100))%)",
                runtimeMS: runtimeMS
            )
        } catch {
            AppLog.intelligence.error("classifier tool selector failed: \(error.localizedDescription, privacy: .public)")
            return ToolSelection(
                intent: nil,
                confidence: 0,
                reasoning: "classifier inference error",
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }
    }
}
