import Testing

@testable import LFMEngine

@Suite("EmbeddingPooling")
struct EmbeddingPoolingTests {

    @Test("mean averages a multi-token matrix column-wise")
    func meanOfTwoTokens() throws {
        // 2 tokens × hiddenDim 3: [1,2,3] and [3,4,5] → column mean [2,3,4]
        let raw = (embeddings: [Float]([1, 2, 3, 3, 4, 5]), numTokens: 2, hiddenDim: 3)
        let pooled = try EmbeddingPooling.mean(raw)
        #expect(pooled == [2, 3, 4])
    }

    @Test("mean of a single token returns that token")
    func meanOfSingleToken() throws {
        let raw = (embeddings: [Float]([7, 8, 9, 10]), numTokens: 1, hiddenDim: 4)
        #expect(try EmbeddingPooling.mean(raw) == [7, 8, 9, 10])
    }

    @Test("mean throws (does not trap) when the buffer is shorter than numTokens × hiddenDim")
    func throwsOnBufferTooSmall() {
        // Declares 2×2 = 4 floats but provides only 3 — the bug the validation
        // prevents: without the guard this would index out of bounds and crash.
        let raw = (embeddings: [Float]([1, 2, 3]), numTokens: 2, hiddenDim: 2)
        #expect(throws: LFMEngineError.self) {
            _ = try EmbeddingPooling.mean(raw)
        }
    }

    @Test("mean throws when the buffer is longer than numTokens × hiddenDim")
    func throwsOnBufferTooLarge() {
        let raw = (embeddings: [Float]([1, 2, 3, 4, 5]), numTokens: 2, hiddenDim: 2)
        #expect(throws: LFMEngineError.self) {
            _ = try EmbeddingPooling.mean(raw)
        }
    }

    @Test("mean throws on an empty / zero-dimension result")
    func throwsOnEmpty() {
        let raw = (embeddings: [Float]([]), numTokens: 0, hiddenDim: 0)
        #expect(throws: LFMEngineError.self) {
            _ = try EmbeddingPooling.mean(raw)
        }
    }
}
