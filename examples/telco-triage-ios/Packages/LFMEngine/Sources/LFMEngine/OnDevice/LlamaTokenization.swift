import Foundation

#if LLAMA_CPP_AVAILABLE
import llama
#endif

/// Tokenization, detokenization, and sampler utilities for LlamaBackend.
extension LlamaBackend {

    // MARK: - Sampler

    /// Create or replace the sampler. Greedy for temperature <= 0, chain (temp + dist) otherwise.
    /// Caches the sampler — skips recreation when temperature hasn't changed.
    func applySampler(temperature: Float) {
        #if LLAMA_CPP_AVAILABLE
        if sampler != nil && samplerTemperature == temperature { return }

        if let s = sampler { llama_sampler_free(s) }

        if temperature <= 0 {
            self.sampler = llama_sampler_init_greedy()
        } else {
            let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())!
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(UInt32.random(in: 0...UInt32.max)))
            self.sampler = chain
        }
        samplerTemperature = temperature
        #endif
    }

    // MARK: - Tokenize

    /// Convert a string to llama tokens.
    func tokenize(_ text: String, addBos: Bool) -> [llama_token] {
        #if LLAMA_CPP_AVAILABLE
        guard let model else { return [] }

        let vocab = llama_model_get_vocab(model)
        let utf8 = Array(text.utf8)
        let maxTokens = utf8.count + (addBos ? 1 : 0) + 1

        var tokens = [llama_token](repeating: 0, count: maxTokens)
        let count = llama_tokenize(vocab, utf8.map { Int8(bitPattern: $0) }, Int32(utf8.count),
                                   &tokens, Int32(maxTokens), addBos, true)

        guard count >= 0 else { return [] }
        return Array(tokens.prefix(Int(count)))
        #else
        return []
        #endif
    }

    // MARK: - Detokenize

    /// Convert llama tokens back to a string.
    func detokenize(_ tokens: [llama_token]) -> String {
        #if LLAMA_CPP_AVAILABLE
        guard let model else { return "" }

        let vocab = llama_model_get_vocab(model)
        var result = ""
        var buf = [CChar](repeating: 0, count: 256)

        for token in tokens {
            let nChars = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, true)
            if nChars > 0 && nChars < Int32(buf.count) {
                buf[Int(nChars)] = 0
                result += String(cString: buf)
            }
        }
        return result
        #else
        return ""
        #endif
    }

    /// Detokenize a single token. Used for incremental stop-sequence checking
    /// to avoid O(N²) re-detokenization of the entire generated sequence.
    func detokenizeSingle(_ token: llama_token) -> String {
        #if LLAMA_CPP_AVAILABLE
        guard let model else { return "" }

        let vocab = llama_model_get_vocab(model)
        var buf = [CChar](repeating: 0, count: 256)
        let nChars = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, true)
        if nChars > 0 && nChars < Int32(buf.count) {
            buf[Int(nChars)] = 0
            return String(cString: buf)
        }
        return ""
        #else
        return ""
        #endif
    }
}
