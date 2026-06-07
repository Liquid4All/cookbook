import Foundation
import os.log

#if LLAMA_CPP_AVAILABLE
import llama
#endif

/// Pooling of per-token hidden states into a single sequence vector.
///
/// LFM2.5's final layer is a depthwise conv (kernel-3 receptive field), so
/// last-token pooling reads only ~3 tokens and silently drops classification
/// quality on terse inputs. Mean pooling over all token positions is the
/// correct choice for LFM2.5 classifier heads (see liquid-models-architecture
/// §3/§16) and is the single source of truth for both the telco shared-clf
/// heads and the telco-turn-relation head — never reimplement it inline.
///
/// Defined here (rather than its own file) so it always compiles in the same
/// translation unit as its only caller, `meanPooledEmbedding`.
public enum EmbeddingPooling {

    /// Mean-pool a row-major `[numTokens × hiddenDim]` hidden-state matrix into
    /// one `hiddenDim` vector. Validates the buffer length against
    /// `numTokens × hiddenDim` before indexing, so a malformed backend result
    /// throws a typed error instead of trapping on an out-of-bounds access.
    public static func mean(
        _ raw: (embeddings: [Float], numTokens: Int, hiddenDim: Int)
    ) throws -> [Float] {
        guard raw.numTokens > 0, raw.hiddenDim > 0 else {
            throw LFMEngineError.inferenceFailed(
                "meanPool: empty embedding (numTokens=\(raw.numTokens), hiddenDim=\(raw.hiddenDim))"
            )
        }
        let expected = raw.numTokens * raw.hiddenDim
        guard raw.embeddings.count == expected else {
            throw LFMEngineError.inferenceFailed(
                "meanPool: buffer length \(raw.embeddings.count) != numTokens×hiddenDim (\(expected))"
            )
        }

        var pooled = [Float](repeating: 0, count: raw.hiddenDim)
        // Bounds are guaranteed by the guard above, so the unchecked buffer
        // access is safe and avoids per-element bounds checks in the hot loop.
        raw.embeddings.withUnsafeBufferPointer { buf in
            for token in 0..<raw.numTokens {
                let offset = token * raw.hiddenDim
                for dim in 0..<raw.hiddenDim {
                    pooled[dim] += buf[offset + dim]
                }
            }
        }
        let inverseCount = 1 / Float(raw.numTokens)
        for dim in 0..<raw.hiddenDim {
            pooled[dim] *= inverseCount
        }
        return pooled
    }
}

extension LlamaBackend {

    /// Extract the last-token hidden state from the backbone after a
    /// prompt-only forward pass (no autoregressive generation).
    ///
    /// This is the embedding path for classifier heads: tokenize the
    /// prompt, run a single `llama_decode`, then read the hidden state
    /// at the last token position via `llama_get_embeddings_ith`.
    ///
    /// The returned array has `llama_model_n_embd` elements (1024 for
    /// LFM2.5-350M). The caller feeds this into `ClassifierHead.classify`
    /// for sub-millisecond label prediction.
    ///
    /// - Parameters:
    ///   - prompt: The full prompt string to encode (already chat-template
    ///     wrapped if needed).
    ///   - clearCache: Whether to clear the KV cache before encoding.
    /// - Returns: The hidden state as a float32 array.
    public func embeddings(
        prompt: String,
        clearCache: Bool = true
    ) throws -> [Float] {
        #if LLAMA_CPP_AVAILABLE
        guard let context, let model else { throw LFMEngineError.modelNotLoaded }

        // Enable embedding output mode so llama_get_embeddings_ith works.
        llama_set_embeddings(context, true)
        defer { llama_set_embeddings(context, false) }

        let tokens = tokenize(prompt, addBos: true)
        guard !tokens.isEmpty else {
            throw LFMEngineError.invalidPromptFormat("Tokenization produced empty result")
        }

        if clearCache {
            let memory = llama_get_memory(context)
            llama_memory_clear(memory, true)
        }

        // Build batch: mark the last token for output (logits[i] = 1)
        var batch = llama_batch_init(Int32(tokens.count), 0, 1)
        defer { llama_batch_free(batch) }

        for (i, token) in tokens.enumerated() {
            let pos = Int32(i)
            batch.token[Int(pos)] = token
            batch.pos[Int(pos)] = llama_pos(pos)
            batch.n_seq_id[Int(pos)] = 1
            batch.seq_id[Int(pos)]![0] = 0
            // Only need embeddings for the last token
            batch.logits[Int(pos)] = (i == tokens.count - 1) ? 1 : 0
        }
        batch.n_tokens = Int32(tokens.count)

        let status = llama_decode(context, batch)
        guard status == 0 else {
            throw LFMEngineError.inferenceFailed("Embedding decode failed with code \(status)")
        }

        // Get the last token's hidden state
        guard let embeddingsPtr = llama_get_embeddings_ith(context, -1) else {
            throw LFMEngineError.inferenceFailed(
                "llama_get_embeddings_ith returned nil — embeddings may not be supported for this model"
            )
        }

        let nEmbd = Int(llama_model_n_embd(model))
        let result = Array(UnsafeBufferPointer(start: embeddingsPtr, count: nEmbd))

        return result
        #else
        throw LFMEngineError.inferenceFailed("llama.cpp not available on this platform")
        #endif
    }

    /// Convenience: extract embeddings from chat messages by first
    /// applying the model's chat template, then running the embedding
    /// forward pass. Use this path when the classifier was trained on
    /// chat-template-wrapped inputs (all leap-finetune classifiers).
    public func embeddings(
        messages: [LlamaChatMessage],
        clearCache: Bool = true
    ) throws -> [Float] {
        let prompt = try applyChatTemplate(messages, addAssistantMarker: true)
        return try embeddings(prompt: prompt, clearCache: clearCache)
    }

    // MARK: - Per-Token Embeddings (Phase 3: Token Classification)

    /// Extract hidden states for ALL token positions after a forward pass.
    ///
    /// Used for token-level classification (BIO tagging for PII detection).
    /// Unlike `embeddings(prompt:)` which extracts only the last token,
    /// this marks every token for output and returns a 2D matrix
    /// [numTokens, hiddenDim].
    ///
    /// Memory: 1024 floats x numTokens. Banking inputs are short (~50-100
    /// tokens), so this is ~400 KB — well within mobile memory.
    ///
    /// - Parameters:
    ///   - prompt: The text to tokenize and encode.
    ///   - clearCache: Whether to clear the KV cache before encoding.
    /// - Returns: A tuple of (embeddings: [numTokens * hiddenDim], numTokens: Int, hiddenDim: Int).
    public func allTokenEmbeddings(
        prompt: String,
        clearCache: Bool = true
    ) throws -> (embeddings: [Float], numTokens: Int, hiddenDim: Int) {
        #if LLAMA_CPP_AVAILABLE
        guard let context, let model else { throw LFMEngineError.modelNotLoaded }

        llama_set_embeddings(context, true)
        defer { llama_set_embeddings(context, false) }

        let tokens = tokenize(prompt, addBos: true)
        guard !tokens.isEmpty else {
            throw LFMEngineError.invalidPromptFormat("Tokenization produced empty result")
        }

        if clearCache {
            let memory = llama_get_memory(context)
            llama_memory_clear(memory, true)
        }

        // Build batch: mark ALL tokens for output
        var batch = llama_batch_init(Int32(tokens.count), 0, 1)
        defer { llama_batch_free(batch) }

        for (i, token) in tokens.enumerated() {
            let pos = Int32(i)
            batch.token[Int(pos)] = token
            batch.pos[Int(pos)] = llama_pos(pos)
            batch.n_seq_id[Int(pos)] = 1
            batch.seq_id[Int(pos)]![0] = 0
            batch.logits[Int(pos)] = 1  // Request embeddings for ALL tokens
        }
        batch.n_tokens = Int32(tokens.count)

        let status = llama_decode(context, batch)
        guard status == 0 else {
            throw LFMEngineError.inferenceFailed("Token embedding decode failed with code \(status)")
        }

        let nEmbd = Int(llama_model_n_embd(model))
        let numTokens = tokens.count
        var allEmbeddings = [Float](repeating: 0, count: numTokens * nEmbd)

        for i in 0..<numTokens {
            guard let ptr = llama_get_embeddings_ith(context, Int32(i)) else {
                throw LFMEngineError.inferenceFailed(
                    "llama_get_embeddings_ith returned nil at position \(i)"
                )
            }
            let buffer = UnsafeBufferPointer(start: ptr, count: nEmbd)
            allEmbeddings.replaceSubrange(i * nEmbd..<(i + 1) * nEmbd, with: buffer)
        }

        return (embeddings: allEmbeddings, numTokens: numTokens, hiddenDim: nEmbd)
        #else
        throw LFMEngineError.inferenceFailed("llama.cpp not available on this platform")
        #endif
    }

    /// Convenience: run a prompt-only forward pass and **mean-pool** all token
    /// hidden states into one vector — the correct pooling for LFM2.5 classifier
    /// heads (the final layer is conv, so last-token reads only ~3 tokens). Both
    /// the telco shared-clf and telco-turn-relation heads use this so their
    /// runtime pooling matches the mean pooling used at training time. The buffer
    /// length is validated by `EmbeddingPooling.mean`, which throws (rather than
    /// trapping) on a malformed result.
    public func meanPooledEmbedding(
        prompt: String,
        clearCache: Bool = true
    ) throws -> [Float] {
        let raw = try allTokenEmbeddings(prompt: prompt, clearCache: clearCache)
        return try EmbeddingPooling.mean(raw)
    }
}
