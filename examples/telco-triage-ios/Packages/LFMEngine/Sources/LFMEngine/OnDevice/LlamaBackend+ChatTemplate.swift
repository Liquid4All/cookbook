import Foundation
import os.log

#if LLAMA_CPP_AVAILABLE
import llama
#endif

/// A single role-tagged message for chat-template application.
/// Mirrors `llama_chat_message` on the C side — two string fields, by value.
public struct LlamaChatMessage: Sendable, Hashable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }

    /// Convenience factory for a user-turn message.
    public static func user(_ content: String) -> LlamaChatMessage {
        LlamaChatMessage(role: "user", content: content)
    }

    /// Convenience factory for a system-turn message.
    public static func system(_ content: String) -> LlamaChatMessage {
        LlamaChatMessage(role: "system", content: content)
    }

    /// Convenience factory for an assistant-turn message (used for few-shot
    /// prompting or when replaying a prior assistant reply in a multi-turn
    /// conversation).
    public static func assistant(_ content: String) -> LlamaChatMessage {
        LlamaChatMessage(role: "assistant", content: content)
    }
}

// MARK: - Chat-template application

extension LlamaBackend {

    /// Extension-local logger. The main `LlamaBackend` logger is private,
    /// so we declare our own with the same subsystem/category for log-stream
    /// continuity.
    fileprivate static let templateLogger = Logger(
        subsystem: "ai.liquid.banking",
        category: "LlamaBackend"
    )

    /// Apply the loaded model's baked-in chat template (stored in the GGUF's
    /// `tokenizer.chat_template` metadata) to a list of messages, returning
    /// the formatted prompt string ready for tokenization.
    ///
    /// For LFM2.5-350M-DPO that template is ChatML-flavored:
    ///   ```
    ///   <|startoftext|><|im_start|>system
    ///   <system content><|im_end|>
    ///   <|im_start|>user
    ///   <user content><|im_end|>
    ///   <|im_start|>assistant
    ///   ```
    /// (Final newline + open `assistant` turn is controlled by
    /// `addAssistantMarker`.)
    ///
    /// Why this matters: `leap-finetune` wraps every training example in the
    /// GGUF's chat template before tokenizing. If the iOS runtime tokenizes
    /// the raw user-message string (no special tokens), the LoRA adapter sees
    /// an out-of-distribution input, its weight deltas misalign, and inference
    /// collapses to the trivial output the adapter learned for garbage input
    /// (observed in the field as `intent: "unknown", confidence: 0.0`).
    ///
    /// - Parameters:
    ///   - messages: The role-tagged turns to format.
    ///   - addAssistantMarker: When `true`, the template appends the tokens
    ///     that open a new assistant turn (e.g. `<|im_start|>assistant\n`),
    ///     so the caller's very next generation step produces the assistant's
    ///     reply. Pass `false` only for rare cases like scoring an existing
    ///     assistant message.
    /// - Returns: The formatted prompt string. Never empty for non-empty input.
    /// - Throws: `LFMEngineError.modelNotLoaded` if `loadModel()` hasn't run,
    ///   or `LFMEngineError.inferenceFailed` on any C-side template failure
    ///   (unknown template, buffer overflow that a retry didn't fix, etc.).
    public func applyChatTemplate(
        _ messages: [LlamaChatMessage],
        addAssistantMarker: Bool = true
    ) throws -> String {
        #if LLAMA_CPP_AVAILABLE
        guard let model else { throw LFMEngineError.modelNotLoaded }
        guard !messages.isEmpty else { return "" }

        // Resolve template: prefer the default baked into the GGUF. LFM2.5
        // ships a full Jinja template via `tokenizer.chat_template`; recent
        // llama.cpp supports arbitrary Jinja via minja.
        // Passing NULL asks llama.cpp to use the model's default.
        let tmplPtr = llama_model_chat_template(model, nil)
        guard tmplPtr != nil else {
            throw LFMEngineError.inferenceFailed(
                "Model has no chat template baked into the GGUF"
            )
        }

        // Flatten Swift strings to C strings whose lifetime we control for
        // the duration of the `llama_chat_apply_template` call. We build
        // two parallel arrays — role cstrs and content cstrs — plus a
        // matching `llama_chat_message` struct per message.
        let roleCStrings: [UnsafeMutablePointer<CChar>] = messages.map {
            strdup($0.role)
        }
        let contentCStrings: [UnsafeMutablePointer<CChar>] = messages.map {
            strdup($0.content)
        }
        defer {
            roleCStrings.forEach { free($0) }
            contentCStrings.forEach { free($0) }
        }

        var cMessages: [llama_chat_message] = []
        cMessages.reserveCapacity(messages.count)
        for i in 0..<messages.count {
            cMessages.append(llama_chat_message(
                role: UnsafePointer(roleCStrings[i]),
                content: UnsafePointer(contentCStrings[i])
            ))
        }

        // First pass: size the output buffer. llama.cpp recommends
        // 2× total character count as a safe initial allocation, then
        // re-allocate if the returned size exceeds it.
        let totalChars = messages.reduce(0) { $0 + $1.role.count + $1.content.count }
        var bufCapacity = max(512, totalChars * 2)

        for attempt in 0..<3 {
            var buffer = [CChar](repeating: 0, count: bufCapacity)
            let written = buffer.withUnsafeMutableBufferPointer { bufPtr in
                cMessages.withUnsafeBufferPointer { msgPtr in
                    llama_chat_apply_template(
                        tmplPtr,
                        msgPtr.baseAddress,
                        messages.count,
                        addAssistantMarker,
                        bufPtr.baseAddress,
                        Int32(bufCapacity)
                    )
                }
            }

            if written < 0 {
                throw LFMEngineError.inferenceFailed(
                    "llama_chat_apply_template returned \(written) (template unknown or malformed)"
                )
            }

            if Int(written) <= bufCapacity {
                // Null-terminate defensively and decode as UTF-8.
                buffer[Int(written)] = 0
                let s = buffer.withUnsafeBufferPointer { ptr -> String in
                    String(cString: ptr.baseAddress!)
                }
                return s
            }

            // Under-allocated — grow and retry. At most twice.
            bufCapacity = Int(written) + 16
            Self.templateLogger.info(
                "chat-template buffer undersized on attempt \(attempt, privacy: .public); resizing to \(bufCapacity, privacy: .public)"
            )
        }

        throw LFMEngineError.inferenceFailed(
            "llama_chat_apply_template failed to fit output after 3 resize attempts"
        )
        #else
        throw LFMEngineError.inferenceFailed("llama.cpp not available on this platform")
        #endif
    }

    /// Run inference against a chat-template-formatted conversation.
    ///
    /// Convenience that pairs `applyChatTemplate(_:)` with the existing
    /// `generate(prompt:...)` path — the ONLY correct way to call a LoRA
    /// adapter that was trained via leap-finetune, which applies this exact
    /// template at training time.
    ///
    /// Propagates the same parameters as `generate(prompt:...)` (see that
    /// method for the detailed contract).
    public func generate(
        messages: [LlamaChatMessage],
        maxTokens: Int,
        temperature: Float = 0.0,
        stopSequences: [String] = [],
        clearCache: Bool = true,
        outputMode: GenerationParams.OutputMode = .text
    ) throws -> (text: String, tokenCount: Int, timing: GenerationTiming) {
        let prompt = try applyChatTemplate(messages, addAssistantMarker: true)
        return try generate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            stopSequences: stopSequences,
            clearCache: clearCache,
            outputMode: outputMode
        )
    }
}
