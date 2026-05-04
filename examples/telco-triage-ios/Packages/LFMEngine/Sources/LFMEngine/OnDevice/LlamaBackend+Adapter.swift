import Foundation
import os.log

#if LLAMA_CPP_AVAILABLE
import llama
#endif

/// LoRA adapter management for LlamaBackend.
///
/// Extracted from LlamaBackend.swift to keep the main file under 300 lines.
extension LlamaBackend {

    // MARK: - LoRA Adapter

    /// Load and apply a LoRA adapter from a GGUF file.
    ///
    /// Caches loaded adapters in memory so subsequent swaps skip disk I/O.
    /// First load: ~100-300ms (flash read + tensor init). Cached swap: ~1ms.
    public func setAdapter(path: String, scale: Float = 1.0) throws {
        #if LLAMA_CPP_AVAILABLE
        guard let model, let context else { throw LFMEngineError.modelNotLoaded }

        // Skip if this adapter is already active
        if activeAdapterPath == path { return }

        // Check cache first — avoids re-reading .gguf from flash
        let adapter: OpaquePointer
        if let cached = adapterCache[path] {
            adapter = cached
            let logger = Logger(subsystem: "ai.liquid.banking", category: "LlamaBackend")
            logger.info("Adapter cache hit: \((path as NSString).lastPathComponent, privacy: .public)")
        } else {
            guard let loaded = llama_adapter_lora_init(model, path) else {
                throw LFMEngineError.adapterLoadFailed(
                    adapter: (path as NSString).lastPathComponent,
                    reason: "llama_adapter_lora_init returned nil"
                )
            }
            adapter = loaded
            adapterCache[path] = loaded
            let logger = Logger(subsystem: "ai.liquid.banking", category: "LlamaBackend")
            logger.info("Adapter loaded + cached: \((path as NSString).lastPathComponent, privacy: .public)")
        }

        // Apply the adapter to the context
        var adapterPtr: OpaquePointer? = adapter
        var adapterScale: Float = scale
        let result = llama_set_adapters_lora(
            context,
            &adapterPtr,
            1,
            &adapterScale
        )
        guard result == 0 else {
            throw LFMEngineError.adapterLoadFailed(
                adapter: (path as NSString).lastPathComponent,
                reason: "llama_set_adapters_lora failed with code \(result)"
            )
        }

        self.loraAdapter = adapter
        self.activeAdapterPath = path
        #else
        throw LFMEngineError.modelNotLoaded
        #endif
    }

    /// Remove the current LoRA adapter, reverting to base model.
    /// The adapter stays in cache for fast re-activation.
    public func removeAdapter() {
        #if LLAMA_CPP_AVAILABLE
        if loraAdapter != nil, let context {
            llama_set_adapters_lora(context, nil, 0, nil)
            self.loraAdapter = nil
            self.activeAdapterPath = nil
            let logger = Logger(subsystem: "ai.liquid.banking", category: "LlamaBackend")
            logger.info("Adapter deactivated (still cached)")
        }
        #endif
    }
}
