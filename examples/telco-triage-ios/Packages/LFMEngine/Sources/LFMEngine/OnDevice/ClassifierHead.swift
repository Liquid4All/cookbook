import Accelerate
import Foundation

/// A lightweight linear classifier head that runs on top of a backbone's
/// last-token hidden state. Loaded from raw float32 binaries exported by
/// `export_heads.py` on the training server.
///
/// Architecture: `logits = W @ hidden_state + bias`, then softmax → argmax.
/// The matrix-vector multiply uses `cblas_sgemv` (Accelerate.framework) —
/// sub-millisecond on any Apple Silicon device.
///
/// Thread safety: the weight buffers are immutable after init, so the
/// `classify` method is safe to call from any isolation domain.
public struct ClassifierHead: Sendable {

    /// Number of output classes.
    public let numClasses: Int

    /// Dimensionality of the input hidden state (must match backbone n_embd).
    public let hiddenDim: Int

    /// Row-major weight matrix [numClasses, hiddenDim], float32.
    private let weights: [Float]

    /// Bias vector [numClasses], float32.
    private let bias: [Float]

    /// Maps class index → label string (e.g. 0 → "kb_question").
    public let id2label: [Int: String]

    /// Maps label string → class index.
    public let label2id: [String: Int]

    /// The task name from training metadata (e.g. "chat_mode_routing").
    public let task: String

    /// Classification result with label, confidence, and the full
    /// probability distribution for downstream gating decisions.
    public struct Prediction: Sendable {
        public let label: String
        public let labelIndex: Int
        public let confidence: Float
        public let probabilities: [Float]
    }

    // MARK: - Loading

    /// Load a classifier head from the three binary artifacts produced by
    /// the training export script.
    ///
    /// - Parameters:
    ///   - weightsURL: Path to `classifier_weights.bin` (float32, row-major [numClasses, hiddenDim]).
    ///   - biasURL: Path to `classifier_bias.bin` (float32, [numClasses]).
    ///   - metaURL: Path to `classifier_meta.json`.
    public init(weightsURL: URL, biasURL: URL, metaURL: URL) throws {
        let metaData = try Data(contentsOf: metaURL)
        guard let metaJSON = try JSONSerialization.jsonObject(with: metaData) as? [String: Any] else {
            throw ClassifierHeadError.invalidMeta("classifier_meta.json is not a JSON object")
        }

        guard let numClasses = metaJSON["num_classes"] as? Int,
              let hiddenDim = metaJSON["hidden_dim"] as? Int
        else {
            throw ClassifierHeadError.invalidMeta("missing num_classes or hidden_dim")
        }

        self.numClasses = numClasses
        self.hiddenDim = hiddenDim
        self.task = (metaJSON["task"] as? String) ?? "unknown"

        // Parse label maps. Keys in JSON are strings even for integer keys.
        var id2label: [Int: String] = [:]
        if let raw = metaJSON["id2label"] as? [String: String] {
            for (k, v) in raw {
                if let idx = Int(k) { id2label[idx] = v }
            }
        }
        self.id2label = id2label

        var label2id: [String: Int] = [:]
        if let raw = metaJSON["label2id"] as? [String: Any] {
            for (k, v) in raw {
                if let idx = v as? Int {
                    label2id[k] = idx
                } else if let idx = v as? NSNumber {
                    label2id[k] = idx.intValue
                }
            }
        }
        self.label2id = label2id

        // Load raw float32 buffers.
        let weightsData = try Data(contentsOf: weightsURL)
        let biasData = try Data(contentsOf: biasURL)

        let expectedWeightBytes = numClasses * hiddenDim * MemoryLayout<Float>.size
        let expectedBiasBytes = numClasses * MemoryLayout<Float>.size

        guard weightsData.count == expectedWeightBytes else {
            throw ClassifierHeadError.sizeMismatch(
                "weights: expected \(expectedWeightBytes) bytes, got \(weightsData.count)"
            )
        }
        guard biasData.count == expectedBiasBytes else {
            throw ClassifierHeadError.sizeMismatch(
                "bias: expected \(expectedBiasBytes) bytes, got \(biasData.count)"
            )
        }

        self.weights = weightsData.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }
        self.bias = biasData.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }
    }

    // MARK: - Inference

    /// Classify a hidden state vector into a label.
    ///
    /// - Parameter hiddenState: The backbone's last-token hidden state,
    ///   a float32 array of length `hiddenDim`.
    /// - Returns: The predicted label, its confidence (softmax probability),
    ///   and the full probability distribution.
    public func classify(_ hiddenState: [Float]) -> Prediction {
        precondition(
            hiddenState.count == hiddenDim,
            "ClassifierHead: hiddenState.count (\(hiddenState.count)) != hiddenDim (\(hiddenDim))"
        )

        // logits = W @ x + b
        // W is [numClasses, hiddenDim] row-major → cblas_sgemv with CblasRowMajor.
        var logits = [Float](repeating: 0, count: numClasses)

        // y = alpha * A * x + beta * y
        // A: [M, N] = [numClasses, hiddenDim], x: [N], y: [M]
        cblas_sgemv(
            CblasRowMajor,        // row-major layout
            CblasNoTrans,         // no transpose
            Int32(numClasses),    // M (rows of A = output dim)
            Int32(hiddenDim),     // N (cols of A = input dim)
            1.0,                  // alpha
            weights,              // A
            Int32(hiddenDim),     // lda (leading dimension = hiddenDim for row-major)
            hiddenState,          // x
            1,                    // incX
            0.0,                  // beta
            &logits,              // y
            1                     // incY
        )

        // Add bias
        vDSP_vadd(logits, 1, bias, 1, &logits, 1, vDSP_Length(numClasses))

        // Softmax: subtract max for numerical stability, then exp, then normalize.
        var maxLogit: Float = 0
        vDSP_maxv(logits, 1, &maxLogit, vDSP_Length(numClasses))

        var shifted = [Float](repeating: 0, count: numClasses)
        var negMax = -maxLogit
        vDSP_vsadd(logits, 1, &negMax, &shifted, 1, vDSP_Length(numClasses))

        var count = Int32(numClasses)
        var probs = [Float](repeating: 0, count: numClasses)
        vvexpf(&probs, shifted, &count)

        var sumExp: Float = 0
        vDSP_sve(probs, 1, &sumExp, vDSP_Length(numClasses))

        vDSP_vsdiv(probs, 1, &sumExp, &probs, 1, vDSP_Length(numClasses))

        // Argmax
        var maxProb: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(probs, 1, &maxProb, &maxIdx, vDSP_Length(numClasses))

        let labelIndex = Int(maxIdx)
        let label = id2label[labelIndex] ?? "unknown_\(labelIndex)"

        return Prediction(
            label: label,
            labelIndex: labelIndex,
            confidence: maxProb,
            probabilities: probs
        )
    }

    // MARK: - Regression

    /// Regression prediction result. The head has a single output neuron
    /// with sigmoid activation, producing a score in [0, 1].
    public struct RegressionPrediction: Sendable {
        /// The predicted score, clamped to [0, 1] via sigmoid.
        public let score: Float
        /// The raw logit before sigmoid (useful for calibration).
        public let rawLogit: Float
    }

    /// Regression inference: single-output linear head + sigmoid.
    ///
    /// Used for continuous targets like fraud risk_score (0.0 - 1.0).
    /// The head must have `numClasses == 1`.
    ///
    /// - Parameter hiddenState: The backbone's last-token hidden state.
    /// - Returns: A score in [0, 1] via sigmoid activation.
    public func classifyRegression(_ hiddenState: [Float]) -> RegressionPrediction {
        precondition(
            hiddenState.count == hiddenDim,
            "ClassifierHead: hiddenState.count (\(hiddenState.count)) != hiddenDim (\(hiddenDim))"
        )
        precondition(
            numClasses == 1,
            "ClassifierHead: regression requires numClasses == 1, got \(numClasses)"
        )

        // logit = w @ x + b (single output)
        var logit: Float = 0
        vDSP_dotpr(weights, 1, hiddenState, 1, &logit, vDSP_Length(hiddenDim))
        logit += bias[0]

        // Sigmoid: 1 / (1 + exp(-x))
        let score = 1.0 / (1.0 + exp(-logit))

        return RegressionPrediction(score: score, rawLogit: logit)
    }

    // MARK: - Multi-Label Classification

    /// Multi-label prediction result. Each class has an independent
    /// probability (sigmoid, not softmax). Multiple classes can be
    /// active simultaneously.
    public struct MultiLabelPrediction: Sendable {
        /// Labels whose probability exceeds the threshold.
        public let activeLabels: [String]
        /// Per-class probabilities (sigmoid, independent).
        public let probabilities: [Float]
        /// Binary vector: 1 where probability >= threshold, 0 otherwise.
        public let binaryVector: [Int]
    }

    /// Multi-label classification using per-class sigmoid activation.
    ///
    /// Unlike `classify()` which uses softmax (mutually exclusive classes),
    /// this applies sigmoid to each logit independently. Multiple classes
    /// can be active simultaneously (e.g., a compliance message can trigger
    /// both `insider_trading_risk` and `off_channel_communication`).
    ///
    /// - Parameters:
    ///   - hiddenState: The backbone's last-token hidden state.
    ///   - threshold: Per-class activation threshold (default 0.5).
    /// - Returns: Active labels, per-class probabilities, and binary vector.
    public func classifyMultiLabel(
        _ hiddenState: [Float],
        threshold: Float = 0.5
    ) -> MultiLabelPrediction {
        precondition(
            hiddenState.count == hiddenDim,
            "ClassifierHead: hiddenState.count (\(hiddenState.count)) != hiddenDim (\(hiddenDim))"
        )

        // logits = W @ x + b
        var logits = [Float](repeating: 0, count: numClasses)
        cblas_sgemv(
            CblasRowMajor, CblasNoTrans,
            Int32(numClasses), Int32(hiddenDim),
            1.0, weights, Int32(hiddenDim),
            hiddenState, 1,
            0.0, &logits, 1
        )
        vDSP_vadd(logits, 1, bias, 1, &logits, 1, vDSP_Length(numClasses))

        // Per-class sigmoid: p = 1 / (1 + exp(-x))
        // Negate logits, exp, add 1, reciprocal
        var negated = [Float](repeating: 0, count: numClasses)
        var minusOne: Float = -1.0
        vDSP_vsmul(logits, 1, &minusOne, &negated, 1, vDSP_Length(numClasses))

        var count = Int32(numClasses)
        var exps = [Float](repeating: 0, count: numClasses)
        vvexpf(&exps, negated, &count)

        var ones = [Float](repeating: 1, count: numClasses)
        var denom = [Float](repeating: 0, count: numClasses)
        vDSP_vadd(exps, 1, ones, 1, &denom, 1, vDSP_Length(numClasses))

        var probs = [Float](repeating: 0, count: numClasses)
        vvrecf(&probs, denom, &count)

        // Threshold to binary vector and collect active labels
        var activeLabels: [String] = []
        var binaryVector = [Int](repeating: 0, count: numClasses)

        for i in 0..<numClasses {
            if probs[i] >= threshold {
                binaryVector[i] = 1
                if let label = id2label[i] {
                    activeLabels.append(label)
                }
            }
        }

        return MultiLabelPrediction(
            activeLabels: activeLabels,
            probabilities: probs,
            binaryVector: binaryVector
        )
    }

    // MARK: - Token Classification (Phase 3: BIO Tagging)

    /// Token-level prediction result. Each token gets a label from the
    /// same label set. Used for BIO-tagged NER (PII detection).
    public struct TokenPrediction: Sendable {
        /// Per-token label indices (argmax of softmax).
        public let labelIndices: [Int]
        /// Per-token label strings.
        public let labels: [String]
        /// Per-token confidence (max softmax probability).
        public let confidences: [Float]
        /// Number of tokens classified.
        public let numTokens: Int
    }

    /// Classify every token position using batch matrix multiplication.
    ///
    /// Uses `cblas_sgemm` (matrix-matrix multiply) instead of `cblas_sgemv`
    /// (matrix-vector): `logits = W @ H^T + bias` where H is [numTokens, hiddenDim].
    ///
    /// - Parameters:
    ///   - allTokenEmbeddings: Flat array of [numTokens * hiddenDim] float32,
    ///     row-major (token-major).
    ///   - numTokens: Number of tokens in the input.
    /// - Returns: Per-token predictions with labels and confidences.
    public func classifyTokens(
        _ allTokenEmbeddings: [Float],
        numTokens: Int
    ) -> TokenPrediction {
        precondition(
            allTokenEmbeddings.count == numTokens * hiddenDim,
            "ClassifierHead: expected \(numTokens * hiddenDim) floats, got \(allTokenEmbeddings.count)"
        )

        // logits = H @ W^T + bias_broadcast
        // H: [numTokens, hiddenDim], W: [numClasses, hiddenDim]
        // Result: [numTokens, numClasses]
        var logits = [Float](repeating: 0, count: numTokens * numClasses)

        // C = alpha * A * B^T + beta * C
        // A = H [numTokens, hiddenDim]
        // B = W [numClasses, hiddenDim] -> B^T = [hiddenDim, numClasses]
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,         // A not transposed
            CblasTrans,           // B transposed
            Int32(numTokens),     // M (rows of C)
            Int32(numClasses),    // N (cols of C)
            Int32(hiddenDim),     // K (shared dimension)
            1.0,                  // alpha
            allTokenEmbeddings,   // A
            Int32(hiddenDim),     // lda
            weights,              // B
            Int32(hiddenDim),     // ldb (before transpose)
            0.0,                  // beta
            &logits,              // C
            Int32(numClasses)     // ldc
        )

        // Add bias to each row
        logits.withUnsafeMutableBufferPointer { logitsPtr in
            for t in 0..<numTokens {
                let offset = t * numClasses
                vDSP_vadd(
                    logitsPtr.baseAddress! + offset, 1,
                    bias, 1,
                    logitsPtr.baseAddress! + offset, 1,
                    vDSP_Length(numClasses)
                )
            }
        }

        // Per-token softmax + argmax
        var labelIndices = [Int](repeating: 0, count: numTokens)
        var labels = [String](repeating: "", count: numTokens)
        var confidences = [Float](repeating: 0, count: numTokens)

        for t in 0..<numTokens {
            let offset = t * numClasses
            var tokenLogits = Array(logits[offset..<offset + numClasses])

            // Softmax
            var maxLogit: Float = 0
            vDSP_maxv(tokenLogits, 1, &maxLogit, vDSP_Length(numClasses))

            var negMax = -maxLogit
            var shifted = [Float](repeating: 0, count: numClasses)
            vDSP_vsadd(tokenLogits, 1, &negMax, &shifted, 1, vDSP_Length(numClasses))

            var count = Int32(numClasses)
            var probs = [Float](repeating: 0, count: numClasses)
            vvexpf(&probs, shifted, &count)

            var sumExp: Float = 0
            vDSP_sve(probs, 1, &sumExp, vDSP_Length(numClasses))
            vDSP_vsdiv(probs, 1, &sumExp, &probs, 1, vDSP_Length(numClasses))

            // Argmax
            var maxProb: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(probs, 1, &maxProb, &maxIdx, vDSP_Length(numClasses))

            labelIndices[t] = Int(maxIdx)
            labels[t] = id2label[Int(maxIdx)] ?? "O"
            confidences[t] = maxProb
        }

        return TokenPrediction(
            labelIndices: labelIndices,
            labels: labels,
            confidences: confidences,
            numTokens: numTokens
        )
    }

    // MARK: - Convenience

    /// Load a classifier head from a directory containing the three
    /// standard artifact files.
    public init(directory: URL) throws {
        try self.init(
            weightsURL: directory.appendingPathComponent("classifier_weights.bin"),
            biasURL: directory.appendingPathComponent("classifier_bias.bin"),
            metaURL: directory.appendingPathComponent("classifier_meta.json")
        )
    }

    /// Load a classifier head from the app bundle using the standard
    /// naming convention: `{prefix}_classifier_{weights,bias,meta}.{bin,json}`.
    public static func fromBundle(prefix: String) -> ClassifierHead? {
        guard let weightsURL = Bundle.main.url(
            forResource: "\(prefix)_classifier_weights", withExtension: "bin"
        ),
        let biasURL = Bundle.main.url(
            forResource: "\(prefix)_classifier_bias", withExtension: "bin"
        ),
        let metaURL = Bundle.main.url(
            forResource: "\(prefix)_classifier_meta", withExtension: "json"
        ) else {
            return nil
        }

        return try? ClassifierHead(weightsURL: weightsURL, biasURL: biasURL, metaURL: metaURL)
    }
}

/// Errors specific to classifier head loading.
public enum ClassifierHeadError: Error, LocalizedError {
    case invalidMeta(String)
    case sizeMismatch(String)

    public var errorDescription: String? {
        switch self {
        case .invalidMeta(let msg): return "ClassifierHead meta error: \(msg)"
        case .sizeMismatch(let msg): return "ClassifierHead size mismatch: \(msg)"
        }
    }
}
