import Foundation

/// A single role-tagged turn, mirroring `LlamaChatMessage` in
/// LFMEngine. Duplicated here (rather than re-exported) so test
/// targets can instantiate these without depending on llama.cpp
/// symbols directly.
public struct AdapterChatMessage: Sendable, Hashable {
    public enum Role: String, Sendable, Hashable {
        case system
        case user
        case assistant
    }

    public let role: Role
    public let content: String

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }

    public static func system(_ content: String) -> AdapterChatMessage {
        AdapterChatMessage(role: .system, content: content)
    }

    public static func user(_ content: String) -> AdapterChatMessage {
        AdapterChatMessage(role: .user, content: content)
    }

    public static func assistant(_ content: String) -> AdapterChatMessage {
        AdapterChatMessage(role: .assistant, content: content)
    }
}

/// Minimal indirection the LFM classifiers + chat provider need from an
/// on-device inference engine. `AppState` constructs a concrete backend
/// (wrapping `LFMEngine.LlamaBackend`) and injects it here, keeping this
/// POC target free of the llama.cpp XCFramework at build time.
///
/// ## Two inference entrypoints — pick by training regime
///
///  - `generate(messages:...)` — **REQUIRED** path for any LoRA adapter
///    trained via `leap-finetune`. The concrete backend applies the
///    model's baked-in chat template (for LFM2.5:
///    `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`)
///    before tokenization. `leap-finetune` applies this exact template
///    at training time, so feeding a LoRA a raw, un-templated prompt
///    is an out-of-distribution input that causes the adapter to
///    collapse to a trivial output (session-049 field bug: every query
///    returned `intent: "unknown", confidence: 0.0`).
///
///  - `generate(prompt:...)` — base-model-only path, no chat-template
///    wrapping. Intended for continuation-style calls that feed a
///    prompt ending in `"Short answer:"` to the base model. Never
///    combine with a non-empty `adapterPath`.
///
/// ## `adapterPath` contract
///
/// - **Non-empty absolute path**: load the LoRA adapter GGUF onto the
///   currently-loaded base model and apply it before generating.
/// - **Empty string (`""`)**: run the base model with NO LoRA adapter.
///   The concrete backend is responsible for detaching any previously
///   applied adapter.
///
/// ## `stopSequences` contract
///
/// Callers that don't care about early termination (the JSON-emitting
/// classifier + tool-selector paths) pass `[]` — the default for the
/// convenience overloads. The chat provider passes curated stops so
/// the 350M base model halts at the first fake-turn boundary instead
/// of emitting "Reply:\n\nReply:\n\nReply:…" trailing garbage.
public protocol AdapterInferenceBackend: Sendable {
    /// Chat-template-wrapped inference. MUST be used by any caller that
    /// applies a LoRA adapter.
    func generate(
        messages: [AdapterChatMessage],
        adapterPath: String,
        maxTokens: Int,
        stopSequences: [String]
    ) async throws -> String

    /// Raw-prompt inference. No chat template applied. Intended for
    /// completion-style base-model calls only.
    func generate(
        prompt: String,
        adapterPath: String,
        maxTokens: Int,
        stopSequences: [String]
    ) async throws -> String
}

public extension AdapterInferenceBackend {
    /// Convenience overload for the classifier + tool-selector paths,
    /// which terminate on JSON parsing rather than explicit stops.
    func generate(
        messages: [AdapterChatMessage],
        adapterPath: String,
        maxTokens: Int
    ) async throws -> String {
        try await generate(
            messages: messages,
            adapterPath: adapterPath,
            maxTokens: maxTokens,
            stopSequences: []
        )
    }

    /// Convenience overload for the few legacy call sites that still
    /// run the base model in continuation mode without stops.
    func generate(
        prompt: String,
        adapterPath: String,
        maxTokens: Int
    ) async throws -> String {
        try await generate(
            prompt: prompt,
            adapterPath: adapterPath,
            maxTokens: maxTokens,
            stopSequences: []
        )
    }
}
