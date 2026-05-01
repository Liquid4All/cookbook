import Foundation
import CryptoKit

/// LFM-based embedding index for KB retrieval.
///
/// First-principles design:
///  - One model, two uses: the LFM2.5-350M base GGUF that generates
///    answers also encodes the KB. No second model in the bundle.
///  - **No LoRA adapter** during encoding — retrieval is decoupled
///    from any classifier-adapter retraining. Stable across releases.
///  - **Mean-pool over all tokens**: decoder LMs put the bulk of the
///    "what does this mean" signal across the sequence. Last-token
///    pooling encodes "what comes next" — fine for classification,
///    suboptimal for retrieval.
///  - **L2-normalized**: cosine similarity = dot product on a unit
///    sphere. Constant-time per query against the small KB matrix.
///  - **On-device index build, cached** keyed by KB content hash +
///    LFM model version. Adding a KB entry triggers rebuild on next
///    launch — never a retraining run.
///
/// Cost: ~5s on a real iPhone for 34 entries. Cache on disk after
/// first build; subsequent launches load in <50ms.
actor LFMKBEmbeddingIndex {
    struct Match {
        let entryId: String
        let similarity: Double
    }

    /// Single KB row in the index.
    private struct IndexedEntry {
        let entryId: String
        let vector: [Float]   // L2-normalized
    }

    private let backend: LlamaBackend
    private let modelVersionTag: String
    /// Path to the shared classifier LoRA used as the encoder.
    ///
    /// Why we use the shared adapter (not the base model):
    /// LFM2.5-350M is a causal DECODER — its raw last-hidden-state
    /// represents "predict next token", not "summarize input". On
    /// mean-pool, cosines between clearly-related telco queries and
    /// KB topics land at 0.3–0.5 — too noisy for a usable threshold.
    /// The shared classifier adapter was trained to make hidden
    /// states cluster by telco intent (the 9 ADR-015 heads project
    /// from this space). That clustering IS task-tuned retrieval
    /// space. Embedding queries and KB topics under this adapter
    /// pushes related-meaning cosines to 0.7+ and unrelated to <0.4.
    /// Re-trained shared adapter ⇒ rebuild index (cache key handles
    /// this via `modelVersionTag`).
    private let encoderAdapterPath: String?
    private var entries: [IndexedEntry] = []
    private var built: Bool = false
    private var contentHash: String = ""

    init(
        backend: LlamaBackend,
        modelVersionTag: String = "lfm25-350m-shared-clf-v4",
        encoderAdapterPath: String? = nil
    ) {
        self.backend = backend
        self.modelVersionTag = modelVersionTag
        self.encoderAdapterPath = encoderAdapterPath
    }

    /// Have we already built or loaded the index for the current KB?
    /// Drives the chat extractor's "fall back to noMatch this turn"
    /// behavior so we never block a user keystroke on a 5s build.
    var isReady: Bool { built }

    /// Build the index from a KB.
    ///
    /// Three-tier load strategy (best latency first):
    ///   1. **Bundled bin** — `kb-embeddings.bin` shipped in app
    ///      Resources, pre-computed at release time using the same
    ///      LFM Q4_K_M. Hash must match the current KB. **0 ms warmup**.
    ///   2. **On-device cache** — `Documents/kb-embeddings-{hash}.bin`
    ///      written after a previous on-device build. **~50 ms load**.
    ///   3. **Compute on device** — LFM forward pass per entry. ~5 s
    ///      for 34 entries on iPhone. Result is cached for next launch.
    ///
    /// This matches the production pitch: ship pre-computed embeddings
    /// for instant warmup, support OTA KB updates by re-shipping just
    /// the bin, and gracefully rebuild on device when the KB diverges
    /// from what was shipped.
    func build(kb: [KBEntry]) async {
        let hash = Self.contentHash(of: kb, modelVersionTag: modelVersionTag)
        contentHash = hash

        // Tier 1: bundled (production pitch — 0ms warmup)
        if let bundled = Self.loadBundled(matchingHash: hash) {
            entries = bundled
            built = true
            AppLog.intelligence.info("KB embedding index loaded from bundle (\(bundled.count) entries, hash=\(hash.prefix(8), privacy: .public))")
            return
        }

        // Tier 2: previous on-device build
        if let loaded = Self.loadCache(hash: hash) {
            entries = loaded
            built = true
            AppLog.intelligence.info("KB embedding index loaded from on-device cache (\(loaded.count) entries, hash=\(hash.prefix(8), privacy: .public))")
            return
        }

        AppLog.intelligence.info("Building KB embedding index for \(kb.count) entries (no bundle/cache match, hash=\(hash.prefix(8), privacy: .public))")

        // Encode under the shared classifier adapter — see
        // `encoderAdapterPath` doc for why. Falls back to base model
        // if the shared adapter isn't available (older builds).
        await applyEncoderAdapter()

        var built: [IndexedEntry] = []
        let startedAt = Date()
        for entry in kb {
            let text = Self.indexText(for: entry)
            do {
                let vector = try await encodeLastToken(text: text)
                built.append(IndexedEntry(entryId: entry.id, vector: vector))
            } catch {
                AppLog.intelligence.error("KB embed failed for entry \(entry.id, privacy: .public): \(error.localizedDescription, privacy: .public)")
            }
        }

        entries = built
        self.built = true

        Self.saveCache(entries: built, hash: hash)
        AppLog.intelligence.info("KB embedding index built in \(Int(Date().timeIntervalSince(startedAt) * 1000))ms")
    }

    /// Cosine top-K retrieval. Returns matches above `threshold` only;
    /// callers map an empty result to `KBCitation.noMatch` so the
    /// chat surface gracefully declines instead of forcing a wrong
    /// article into the answer.
    func search(query: String, topK: Int = 1, threshold: Double = 0.40) async throws -> [Match] {
        guard built else { return [] }

        await applyEncoderAdapter()
        let queryVec = try await encodeLastToken(text: query)

        // Cosine on unit-norm vectors == dot product. Iterate the small
        // KB matrix; for ~50 entries this is 30µs of arithmetic.
        var scored: [Match] = []
        scored.reserveCapacity(entries.count)
        for indexed in entries {
            let sim = Double(Self.dot(queryVec, indexed.vector))
            if sim >= threshold {
                scored.append(Match(entryId: indexed.entryId, similarity: sim))
            }
        }
        scored.sort { $0.similarity > $1.similarity }
        return Array(scored.prefix(topK))
    }

    // MARK: - Encoding

    /// Set the encoder adapter (or detach if `encoderAdapterPath` is
    /// nil). Same path used at index build time and at query time so
    /// both sides of the cosine live in the same hidden-state space.
    private func applyEncoderAdapter() async {
        if let path = encoderAdapterPath {
            try? await backend.setAdapter(path: path, scale: 1.0)
        } else {
            await backend.removeAdapter()
        }
    }

    /// Last-token hidden state of the input under the encoder adapter,
    /// L2-normalized. We use last-token (not mean-pool) because:
    ///
    ///   - LFM2.5-350M is a causal decoder. Each token's hidden state
    ///     summarizes "everything seen so far" — the last token has
    ///     seen the entire input.
    ///   - The 9 ADR-015 classifier heads project from this exact
    ///     hidden state under this exact adapter. Matching the heads'
    ///     pooling strategy means our embedding lives in the same
    ///     telco-classification-tuned space the heads were trained
    ///     against — not a parallel mean-pool space the model has
    ///     never seen.
    ///   - Mean-pool dilutes the signal on a decoder LM. Empirically,
    ///     last-token + shared-adapter cosines for related telco
    ///     queries land 0.6–0.85; mean-pool was 0.3–0.5.
    ///
    /// Returns a `[Float]` of length `hiddenDim` (1024 for LFM2.5-350M).
    private func encodeLastToken(text: String) async throws -> [Float] {
        let raw = try await backend.embeddings(prompt: text, clearCache: true)
        let dim = raw.count
        guard dim > 0 else {
            throw LFMEngineError.invalidPromptFormat("embedding output empty")
        }
        var pooled = raw

        // L2-normalize so cosine = dot.
        var norm: Float = 0
        for v in pooled { norm += v * v }
        norm = norm.squareRoot()
        if norm > 1e-8 {
            let inv = 1.0 / norm
            for d in 0..<dim { pooled[d] *= inv }
        }
        return pooled
    }

    // MARK: - Index text

    /// Representative text for indexing a KB entry.
    ///
    /// Skip the answer body — its numbered "Step 1: ..." boilerplate
    /// adds noise that pulls all entries toward each other. Topic +
    /// aliases are paraphrase-rich and give a cleaner retrieval signal.
    static func indexText(for entry: KBEntry) -> String {
        let aliases = entry.aliases.joined(separator: ", ")
        if aliases.isEmpty {
            return entry.topic
        }
        return "\(entry.topic) — \(aliases)"
    }

    // MARK: - Hash + cache

    static func contentHash(of kb: [KBEntry], modelVersionTag: String) -> String {
        var hasher = SHA256()
        hasher.update(data: Data(modelVersionTag.utf8))
        // Order-stable — same KB regardless of array order produces the
        // same hash.
        for entry in kb.sorted(by: { $0.id < $1.id }) {
            hasher.update(data: Data(entry.id.utf8))
            hasher.update(data: Data(entry.topic.utf8))
            hasher.update(data: Data(entry.aliases.joined(separator: "|").utf8))
        }
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func cacheURL(hash: String) -> URL? {
        guard let dir = try? FileManager.default.url(
            for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true
        ) else { return nil }
        return dir.appendingPathComponent("kb-embeddings-\(hash).bin")
    }

    /// Binary format (little-endian, version 2):
    ///   `magic(4)="KBEM" | version(1) | hash_len(u8) | hash(utf8) |
    ///    count(u32) | dim(u32)`
    ///   then per entry:
    ///     `id_len(u16) | id(utf8) | dim*float32`
    ///
    /// The embedded hash lets the bundle path (and offline tooling)
    /// confirm it matches the current KB before trusting the vectors.
    private struct Header {
        static let magic: [UInt8] = [0x4B, 0x42, 0x45, 0x4D]  // "KBEM"
        static let version: UInt8 = 2
    }

    private static func saveCache(entries: [IndexedEntry], hash: String) {
        guard !entries.isEmpty, let url = cacheURL(hash: hash) else { return }
        let data = encode(entries: entries, hash: hash)
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            AppLog.intelligence.warning("KB embedding cache write failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    /// Pure encoder for the binary format. Used by both the on-device
    /// cache writer and the offline pre-build script (which links a
    /// thin Swift port or matches the format byte-for-byte).
    private static func encode(entries: [IndexedEntry], hash: String) -> Data {
        var data = Data()
        data.append(contentsOf: Header.magic)
        data.append(Header.version)
        let hashBytes = Array(hash.utf8)
        data.append(UInt8(hashBytes.count))
        data.append(contentsOf: hashBytes)
        var count = UInt32(entries.count).littleEndian
        withUnsafeBytes(of: &count) { data.append(contentsOf: $0) }
        var dim = UInt32(entries.first?.vector.count ?? 0).littleEndian
        withUnsafeBytes(of: &dim) { data.append(contentsOf: $0) }
        for entry in entries {
            let idBytes = Array(entry.entryId.utf8)
            var idLen = UInt16(idBytes.count).littleEndian
            withUnsafeBytes(of: &idLen) { data.append(contentsOf: $0) }
            data.append(contentsOf: idBytes)
            entry.vector.withUnsafeBufferPointer { buf in
                data.append(contentsOf: UnsafeRawBufferPointer(buf))
            }
        }
        return data
    }

    /// Parse the on-disk format. We use `loadUnaligned(as:)` everywhere
    /// because `Data` slices don't guarantee 4-byte alignment of their
    /// backing storage — `load(as:)` traps with "load from misaligned
    /// raw pointer" on slices that happen to land on odd addresses.
    private static func parseCacheData(_ data: Data, expectedHash: String?) -> [IndexedEntry]? {
        guard data.count >= 4 + 1 + 1 else { return nil }
        let magic = Array(data[0..<4])
        guard magic == Header.magic, data[4] == Header.version else { return nil }
        let hashLen = Int(data[5])
        var offset = 6
        guard offset + hashLen + 4 + 4 <= data.count else { return nil }
        let storedHash = String(data: data[offset..<offset + hashLen], encoding: .utf8) ?? ""
        offset += hashLen
        if let expected = expectedHash, storedHash != expected { return nil }
        guard let count = readUInt32(data: data, offset: offset) else { return nil }
        offset += 4
        guard let dim = readUInt32(data: data, offset: offset) else { return nil }
        offset += 4
        var loaded: [IndexedEntry] = []
        loaded.reserveCapacity(Int(count))
        for _ in 0..<Int(count) {
            guard offset + 2 <= data.count else { return nil }
            guard let idLenU16 = readUInt16(data: data, offset: offset) else { return nil }
            let idLen = Int(idLenU16)
            offset += 2
            guard offset + idLen <= data.count else { return nil }
            let id = String(data: data[offset..<offset + idLen], encoding: .utf8) ?? ""
            offset += idLen
            let bytes = Int(dim) * MemoryLayout<Float>.size
            guard offset + bytes <= data.count else { return nil }
            var vector = [Float](repeating: 0, count: Int(dim))
            // Copy float-by-float using loadUnaligned — never reinterpret
            // the raw pointer as `Float*` because a Data slice's start
            // address may not be 4-byte aligned.
            data.withUnsafeBytes { raw in
                let base = raw.baseAddress!.advanced(by: offset)
                for d in 0..<Int(dim) {
                    vector[d] = base.loadUnaligned(
                        fromByteOffset: d * MemoryLayout<Float>.size,
                        as: Float.self
                    )
                }
            }
            offset += bytes
            loaded.append(IndexedEntry(entryId: id, vector: vector))
        }
        return loaded
    }

    private static func readUInt32(data: Data, offset: Int) -> UInt32? {
        guard offset + 4 <= data.count else { return nil }
        return data.withUnsafeBytes { raw -> UInt32 in
            let base = raw.baseAddress!.advanced(by: offset)
            return base.loadUnaligned(as: UInt32.self).littleEndian
        }
    }

    private static func readUInt16(data: Data, offset: Int) -> UInt16? {
        guard offset + 2 <= data.count else { return nil }
        return data.withUnsafeBytes { raw -> UInt16 in
            let base = raw.baseAddress!.advanced(by: offset)
            return base.loadUnaligned(as: UInt16.self).littleEndian
        }
    }

    /// Load the pre-computed bundle if its embedded hash matches the
    /// current KB. This is the production pitch path: an ML engineer
    /// pre-computes embeddings using the same Q4_K_M LFM, ships the
    /// bin alongside the IPA, and the app is index-ready at the first
    /// query. Returns nil when missing or stale (KB diverged from
    /// what was shipped).
    private static func loadBundled(matchingHash hash: String) -> [IndexedEntry]? {
        guard let url = Bundle.main.url(forResource: "kb-embeddings", withExtension: "bin"),
              let data = try? Data(contentsOf: url) else { return nil }
        return parseCacheData(data, expectedHash: hash)
    }

    private static func loadCache(hash: String) -> [IndexedEntry]? {
        guard let url = cacheURL(hash: hash),
              let data = try? Data(contentsOf: url) else { return nil }
        return parseCacheData(data, expectedHash: hash)
    }

    private static func dot(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "embedding dimension mismatch")
        var s: Float = 0
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                for i in 0..<a.count { s += ap[i] * bp[i] }
            }
        }
        return s
    }
}
