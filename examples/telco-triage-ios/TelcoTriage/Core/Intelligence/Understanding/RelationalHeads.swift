import Foundation

/// ADR-024 — pairwise relational signals over consecutive turns.
///
/// Where the single-input heads in ADR-022 (`chat_mode`, `topic_gate`,
/// `refusal_flags`, `emotional_state`, `slot_completeness`) classify
/// the CURRENT turn in isolation, the relational heads classify the
/// **relationship between the current turn and the prior turn**.
/// They consume pairs of hidden states (e.g. `(h_t, h_{t-1}^a)`) and
/// emit categorical signals the deterministic router fuses with the
/// single-input vector.
///
/// **Why a separate file**: the single-input layer is per-query; the
/// relational layer is per-conversation. Keeping the types in one
/// place keeps the multi-turn architecture audit-able from a single
/// header — the same way `SlotCompleteness.swift` makes the slot
/// schema a self-contained contract.

// MARK: - turn_relationship (5-way)

/// 5-way taxonomy of relationships between the current user turn and
/// the assistant's PRIOR reply. Closed-world: every observed multi-turn
/// pattern in the Verizon 50-conversation corpus maps to exactly one
/// of these. Adding a 6th case requires a corpus re-audit + retrain.
///
/// **Wire contract** (frozen — index → meaning maps directly into the
/// trained head's softmax outputs):
///
///  - 0 `independent`             — q_t is a fresh query unrelated to a_{t-1}.
///  - 1 `anaphoric`               — q_t references entities/links/concepts
///                                  the assistant mentioned ("Where is
///                                  Network?" after "Go to **Network** > …").
///  - 2 `clarificationAnswer`     — q_t answers a question the assistant
///                                  asked ("kitchen tablet" after "Which
///                                  device?").
///  - 3 `negativeContinuation`    — q_t reports the prior suggestion
///                                  failed ("didn't work", "still broken").
///  - 4 `affirmativeContinuation` — q_t acknowledges, thanks, or asks
///                                  to proceed ("yes go ahead", "great
///                                  what about X").
public enum TurnRelationship: Int, Sendable, Equatable, Codable, CaseIterable {
    case independent             = 0
    case anaphoric               = 1
    case clarificationAnswer     = 2
    case negativeContinuation    = 3
    case affirmativeContinuation = 4

    /// Stable wire string for logs + telemetry. Matches the training-
    /// data label vocabulary in `scripts/vz/relational/`.
    public var wireName: String {
        switch self {
        case .independent:             return "independent"
        case .anaphoric:               return "anaphoric"
        case .clarificationAnswer:     return "clarification_answer"
        case .negativeContinuation:    return "negative_continuation"
        case .affirmativeContinuation: return "affirmative_continuation"
        }
    }

    /// Bridge from the wire-label string. Returns `.independent` (the
    /// safe default — "don't change behavior") on unknown strings so a
    /// future training-data drift doesn't silently break dispatch.
    public static func from(wireName: String) -> TurnRelationship {
        switch wireName {
        case "independent":             return .independent
        case "anaphoric":               return .anaphoric
        case "clarification_answer":    return .clarificationAnswer
        case "negative_continuation":   return .negativeContinuation
        case "affirmative_continuation": return .affirmativeContinuation
        default:                        return .independent
        }
    }

    /// Whether this relationship indicates the user is continuing the
    /// prior topic (vs starting a fresh thread). Convenience for the
    /// router's "should I clear pendingClarification?" decision.
    public var isContinuation: Bool {
        switch self {
        case .independent:             return false
        case .anaphoric,
             .clarificationAnswer,
             .negativeContinuation,
             .affirmativeContinuation: return true
        }
    }
}

/// `turn_relationship` head output. Carries the typed top-1 label +
/// the head's softmax confidence. Per-class probabilities are kept
/// for the engineering trace (so a reviewer can see the runner-up
/// when the top-1 is borderline).
public struct TurnRelationshipOutcome: Sendable, Equatable {
    public let value: TurnRelationship
    public let confidence: Double
    /// Per-class softmax probabilities in case ordinal:
    /// `[independent, anaphoric, clarificationAnswer,
    /// negativeContinuation, affirmativeContinuation]`. Surfaced in
    /// the trace; never consulted by the router (per ADR-022 §4.3
    /// principle #1: heads inform, routers decide on top-1).
    public let probabilities: [Double]

    public init(
        value: TurnRelationship,
        confidence: Double,
        probabilities: [Double] = []
    ) {
        self.value = value
        self.confidence = confidence
        self.probabilities = probabilities
    }
}

// MARK: - slot_alignment (per-slot multi-label)

/// 4-flag multi-label output: does the current user turn fill each
/// of the four slot kinds? Same slot taxonomy as
/// `SlotCompleteness` (ADR-022 §4.3) — kept aligned by sharing the
/// `Slot` enum.
///
/// Used by the clarification-recovery path to replace the bare-noun
/// regex heuristic in `tryFulfillPendingClarification`. Where the
/// regex matched only known noun shapes ("kitchen tablet",
/// "upstairs"), the head can recognize free-form slot fills
/// ("the one in the corner room") that follow no fixed pattern.
///
/// **Wire contract** (frozen — index → slot in lockstep with
/// `SlotCompleteness.from(binaryVector:)`):
///   - 0 `device`
///   - 1 `location`
///   - 2 `time`
///   - 3 `accountRef`
public struct SlotAlignment: Sendable, Equatable, Hashable, Codable {
    public let fillsDevice: Bool
    public let fillsLocation: Bool
    public let fillsTime: Bool
    public let fillsAccountRef: Bool

    public init(
        fillsDevice: Bool,
        fillsLocation: Bool,
        fillsTime: Bool,
        fillsAccountRef: Bool
    ) {
        self.fillsDevice = fillsDevice
        self.fillsLocation = fillsLocation
        self.fillsTime = fillsTime
        self.fillsAccountRef = fillsAccountRef
    }

    public static let none = SlotAlignment(
        fillsDevice: false,
        fillsLocation: false,
        fillsTime: false,
        fillsAccountRef: false
    )

    /// Does the current turn fill the specified slot?
    public func fills(_ slot: Slot) -> Bool {
        switch slot {
        case .device:     return fillsDevice
        case .location:   return fillsLocation
        case .time:       return fillsTime
        case .accountRef: return fillsAccountRef
        }
    }

    /// The set of slots this turn fills.
    public var filledSlots: Set<Slot> {
        var s: Set<Slot> = []
        if fillsDevice     { s.insert(.device) }
        if fillsLocation   { s.insert(.location) }
        if fillsTime       { s.insert(.time) }
        if fillsAccountRef { s.insert(.accountRef) }
        return s
    }

    /// Decode from the head's per-class binary vector. Same defensive
    /// pattern as `SlotCompleteness.from(binaryVector:)`.
    public static func from(binaryVector: [Int]) -> SlotAlignment {
        func bit(_ i: Int) -> Bool { i < binaryVector.count && binaryVector[i] == 1 }
        return SlotAlignment(
            fillsDevice:     bit(0),
            fillsLocation:   bit(1),
            fillsTime:       bit(2),
            fillsAccountRef: bit(3)
        )
    }
}

public struct SlotAlignmentOutcome: Sendable, Equatable {
    public let value: SlotAlignment
    /// Per-slot sigmoid probabilities (4 floats). Surfaced in the trace.
    public let probabilities: [Double]

    public init(value: SlotAlignment, probabilities: [Double] = []) {
        self.value = value
        self.probabilities = probabilities
    }
}

// MARK: - stance_change (3-way)

/// 3-way taxonomy of stance changes between consecutive USER turns.
/// Captures the "actually, scratch that" / parameter-override patterns.
/// Smaller than `TurnRelationship` because stance shifts are rarer in
/// production data; the dedicated head keeps the categorical decision
/// clean.
///
/// **Wire contract**:
///   - 0 `continue` — q_t is consistent with q_{t-1}'s intent + params.
///   - 1 `revert`   — q_t explicitly cancels a prior decision ("scratch
///                    that", "never mind", "no wait").
///   - 2 `override` — q_t replaces a parameter while keeping the intent
///                    ("actually for my daughter, not my son").
public enum StanceChange: Int, Sendable, Equatable, Codable, CaseIterable {
    case `continue` = 0
    case revert     = 1
    case override   = 2

    public var wireName: String {
        switch self {
        case .continue: return "continue"
        case .revert:   return "revert"
        case .override: return "override"
        }
    }

    public static func from(wireName: String) -> StanceChange {
        switch wireName {
        case "continue": return .continue
        case "revert":   return .revert
        case "override": return .override
        default:         return .continue
        }
    }
}

public struct StanceChangeOutcome: Sendable, Equatable {
    public let value: StanceChange
    public let confidence: Double
    public let probabilities: [Double]

    public init(
        value: StanceChange,
        confidence: Double,
        probabilities: [Double] = []
    ) {
        self.value = value
        self.confidence = confidence
        self.probabilities = probabilities
    }
}

// MARK: - ADR-028 telco_turn_relation (12-way)

/// ADR-028 12-way dialogue-state label. Kept alongside the legacy
/// 5-way `TurnRelationshipOutcome` because the normal composer path
/// needs policy-grade labels such as `step_focus`, `repair_failed`,
/// and `topic_switch`, not just their compressed ADR-024 mapping.
public struct TelcoTurnRelationOutcome: Sendable, Equatable {
    public let value: TelcoTurnRelation
    public let confidence: Double
    /// Per-class softmax probabilities in `TelcoTurnRelation.allCases`
    /// order. Surfaced for rollout audits; policy uses only top-1 plus
    /// deterministic safety guards.
    public let probabilities: [Double]

    public init(
        value: TelcoTurnRelation,
        confidence: Double,
        probabilities: [Double] = []
    ) {
        self.value = value
        self.confidence = confidence
        self.probabilities = probabilities
    }
}

// MARK: - Strategy protocol

/// Pairwise relational classifier — the architectural twin of
/// `SharedBackboneStrategy` for ADR-024. Given the current turn's
/// hidden state plus the cached prior-turn hiddens, produces the three
/// relational outcomes.
///
/// **Why a separate strategy** (vs adding methods to the single-input
/// strategy): the relational adapter (`telco-relational-v1`) is a
/// different LoRA, trained on different data. Conflating it with the
/// single-input strategy would force the single-input call site to
/// pay relational adapter swaps even on the first turn (no prior
/// hidden state to compare against).
///
/// **Degraded build**: when the relational adapter or heads aren't
/// bundled, `UnavailableRelationalStrategy` returns nil for all three
/// outcomes. The router treats nil-relational as `INDEPENDENT`, which
/// is the single-turn baseline.
public protocol RelationalHeadsStrategy: Sendable {
    /// **SBERT v2 surface** — run the relational heads on cached prior-
    /// turn hidden states. The original ADR-024 §4 design. Used by
    /// future `SBERTRelationalStrategy` once an LFMEngine hidden-state
    /// hook ships (Phase δ.0/v2). Returns nil outcomes when prior
    /// hiddens are unavailable.
    func classify(
        currentUserQuery: String,
        priorUserHidden: [Float]?,
        priorAssistantHidden: [Float]?
    ) async throws -> RelationalOutcomes

    /// **Chat-template v1 surface** (ADR-024 Phase β plan §2 — the
    /// production path that ships first). Takes RAW prior-turn text
    /// rather than pre-computed hidden states. Implementations build
    /// a 2-turn chat-templated input, run the relational LoRA adapter,
    /// and mean-pool the final hidden state for the three classifier
    /// heads.
    ///
    /// **Inputs:**
    ///   - `currentUserQuery`: the just-arrived user message
    ///   - `priorAssistantText`: cached from `ConversationState`
    ///     (populated by `recordTurnSideEffects` after each assistant
    ///     reply renders, Phase α 2026-05-27)
    ///   - `priorUserText`: optional — the user message that drove the
    ///     prior assistant reply. Lets `stance_change` head compare
    ///     `(u_t, u_{t-1})` directly.
    ///
    /// **First-turn semantics:** when `priorAssistantText` is nil OR
    /// empty, returns `.none`. The strategy MUST NOT invent a fake
    /// prior — feeding empty strings to the model creates a degenerate
    /// input distribution that doesn't appear in training.
    ///
    /// **Default implementation** routes through the SBERT v2 method
    /// with nil hiddens so existing call sites continue to work — the
    /// new chat-template surface is opt-in for strategies that bundle
    /// the v1 adapter.
    func classifyFromText(
        currentUserQuery: String,
        priorAssistantText: String?,
        priorUserText: String?
    ) async throws -> RelationalOutcomes

    /// Text-surface classification with runtime state required by the
    /// ADR-028 12-way turn-relation head. Strategies that do not need
    /// these fields can use the default implementation.
    func classifyFromText(
        currentUserQuery: String,
        priorAssistantText: String?,
        priorUserText: String?,
        runtimeState: RelationalRuntimeState
    ) async throws -> RelationalOutcomes
}

extension RelationalHeadsStrategy {
    /// Default implementation — for strategies that haven't migrated to
    /// the v1 chat-template surface yet. Routes through the SBERT
    /// method with nil hiddens, which (per the v2 method's contract)
    /// returns `.none`.
    public func classifyFromText(
        currentUserQuery: String,
        priorAssistantText: String?,
        priorUserText: String?
    ) async throws -> RelationalOutcomes {
        _ = priorAssistantText
        _ = priorUserText
        return try await classify(
            currentUserQuery: currentUserQuery,
            priorUserHidden: nil,
            priorAssistantHidden: nil
        )
    }

    public func classifyFromText(
        currentUserQuery: String,
        priorAssistantText: String?,
        priorUserText: String?,
        runtimeState: RelationalRuntimeState
    ) async throws -> RelationalOutcomes {
        _ = runtimeState
        return try await classifyFromText(
            currentUserQuery: currentUserQuery,
            priorAssistantText: priorAssistantText,
            priorUserText: priorUserText
        )
    }
}

public struct RelationalRuntimeState: Sendable, Equatable {
    public let priorRoute: String?
    public let priorPageID: String?
    public let priorLinkID: String?
    public let pendingTool: String?
    public let pendingConfirmation: Bool
    public let pendingClarification: String?
    public let frustrationCount: Int

    public init(
        priorRoute: String? = nil,
        priorPageID: String? = nil,
        priorLinkID: String? = nil,
        pendingTool: String? = nil,
        pendingConfirmation: Bool = false,
        pendingClarification: String? = nil,
        frustrationCount: Int = 0
    ) {
        self.priorRoute = priorRoute
        self.priorPageID = priorPageID
        self.priorLinkID = priorLinkID
        self.pendingTool = pendingTool
        self.pendingConfirmation = pendingConfirmation
        self.pendingClarification = pendingClarification
        self.frustrationCount = frustrationCount
    }

    public static let empty = RelationalRuntimeState()
}

/// The combined output of one relational pass. All fields optional —
/// router handles nil per ADR-022 §4.3 design principle.
public struct RelationalOutcomes: Sendable, Equatable {
    public let telcoTurnRelation: TelcoTurnRelationOutcome?
    public let turnRelationship: TurnRelationshipOutcome?
    public let slotAlignment: SlotAlignmentOutcome?
    public let stanceChange: StanceChangeOutcome?
    /// Wall-clock for the relational forward pass + head projections.
    /// Surfaced in the engineering trace so the latency budget stays
    /// observable.
    public let runtimeMs: Double

    public init(
        telcoTurnRelation: TelcoTurnRelationOutcome? = nil,
        turnRelationship: TurnRelationshipOutcome? = nil,
        slotAlignment: SlotAlignmentOutcome? = nil,
        stanceChange: StanceChangeOutcome? = nil,
        runtimeMs: Double = 0
    ) {
        self.telcoTurnRelation = telcoTurnRelation
        self.turnRelationship = turnRelationship
        self.slotAlignment = slotAlignment
        self.stanceChange = stanceChange
        self.runtimeMs = runtimeMs
    }

    /// All-nil outcomes — used when the strategy is unavailable or
    /// when prior turn context is missing (first turn of a session).
    public static let none = RelationalOutcomes()
}

/// Degraded-build strategy. Returns all-nil outcomes. Used until the
/// `telco-relational-v1` adapter + heads ship in a future bundle
/// (per ADR-024 Phase γ training run).
public struct UnavailableRelationalStrategy: RelationalHeadsStrategy {
    public init() {}
    public func classify(
        currentUserQuery: String,
        priorUserHidden: [Float]?,
        priorAssistantHidden: [Float]?
    ) async throws -> RelationalOutcomes {
        _ = currentUserQuery
        _ = priorUserHidden
        _ = priorAssistantHidden
        return .none
    }

    public func classifyFromText(
        currentUserQuery: String,
        priorAssistantText: String?,
        priorUserText: String?
    ) async throws -> RelationalOutcomes {
        _ = currentUserQuery
        _ = priorAssistantText
        _ = priorUserText
        return .none
    }
}

/// **Chat-template v1 strategy** (ADR-024 Phase δ — shipped 2026-05-27).
///
/// Loads the bundled `telco-relational-v1.gguf` LoRA adapter and runs
/// a pure-generative classification pass for `turn_relationship`.
///
/// **Training recap** (γ-1):
///   - Generative SFT on LFM2.5-350M-Base via leap-finetune HIGH_R_LORA
///   - Training format: 2-turn chat-template with sentinel-separated user
///     message (`[USER_PRIOR]` / `[ASSISTANT_PRIOR]` / `[USER]`) and a
///     single class-label as the assistant reply
///   - turn_relationship: macro F1 = 0.917, all 5 classes ≥ 0.85 ✅
///   - slot_alignment / stance_change: deferred (sparse-positive trap
///     and class-collapse respectively — see F18-F21 in
///     docs/patterns/teacher-distillation-pipeline.md)
///
/// **Bundle contract** (v1 ships ONE artifact):
///   - `telco-relational-v1.gguf`  (~12 MB, F16 LoRA, r=16/α=32)
///   - No classifier-head binaries (generative SFT, not head-trained)
///
/// **Construction**:
///   - `bundled(backend:bundle:)` factory returns non-nil when the
///     adapter GGUF is present. Caller falls back to
///     `UnavailableRelationalStrategy` when nil (graceful degraded path
///     — router sees nil outcomes, falls through to single-turn baseline).
///
/// **Adapter swap contract** (shared-backend architecture):
///   The relational adapter is applied immediately before each call and
///   left loaded. The next Layer 1 call on `QueryUnderstandingClassifier`
///   will swap to its own adapter — the llama.cpp adapter cache handles
///   the transition. No teardown needed.
public final class ChatTemplateRelationalStrategy: RelationalHeadsStrategy, @unchecked Sendable {
    private let backend: LlamaBackend
    private let adapterPath: String

    // MARK: - System prompt (must match prepare_training_data.py EXACTLY)

    private static let systemPrompt =
        "You are a multi-turn understanding classifier for Verizon home internet support. " +
        "Classify the relationship between the current user turn and the prior assistant turn."

    public init(backend: LlamaBackend, adapterPath: String) {
        self.backend = backend
        self.adapterPath = adapterPath
    }

    /// Factory — returns a live strategy when `telco-relational-v1.gguf`
    /// is in the bundle, nil otherwise. No classifier-head binaries
    /// required (generative SFT path, not head-trained).
    public static func bundled(
        backend: LlamaBackend,
        bundle: Bundle = .main
    ) -> ChatTemplateRelationalStrategy? {
        guard let adapterPath = TelcoModelBundle.relationalV1AdapterPath(in: bundle) else {
            return nil
        }
        return ChatTemplateRelationalStrategy(backend: backend, adapterPath: adapterPath)
    }

    // MARK: - SBERT v2 surface (routes through text surface with nil priors)

    public func classify(
        currentUserQuery: String,
        priorUserHidden: [Float]?,
        priorAssistantHidden: [Float]?
    ) async throws -> RelationalOutcomes {
        _ = priorUserHidden
        _ = priorAssistantHidden
        return try await classifyFromText(
            currentUserQuery: currentUserQuery,
            priorAssistantText: nil,
            priorUserText: nil
        )
    }

    // MARK: - Chat-template v1 surface (the production path)

    public func classifyFromText(
        currentUserQuery: String,
        priorAssistantText: String?,
        priorUserText: String?
    ) async throws -> RelationalOutcomes {
        // First-turn guard: no prior assistant text → undefined relational
        // signal. Return .none so the router stays on the single-turn
        // baseline. Feeding the model an empty [ASSISTANT_PRIOR] creates
        // a degenerate distribution not present in training (F19).
        let trimmedPrior = priorAssistantText?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let prior = trimmedPrior, !prior.isEmpty else {
            return .none
        }

        let start = Date()

        // Build the pairwise input in EXACTLY the sentinel format that
        // prepare_training_data.py writes — including its conditional
        // emission of the `[USER_PRIOR]:` line. Deviating by even a single
        // character will push the input out of the training distribution
        // and trigger the "unparseable output" fallback path (F19).
        //
        // Python (prepare_training_data.py:build_user_content) skips the
        // `[USER_PRIOR]` line entirely when prior_user_query is empty:
        //   parts: list[str] = []
        //   if ex.prior_user_query.strip():
        //       parts.append(f"[USER_PRIOR]: {...}")
        //   if ex.prior_assistant_reply.strip():
        //       parts.append(f"[ASSISTANT_PRIOR]: {...}")
        //   parts.append(f"[USER]: {...}")
        // Match that conditional exactly. Note: by the time we get here
        // the first-turn guard above has confirmed `prior` (assistant) is
        // non-empty, so [ASSISTANT_PRIOR] always emits.
        //
        // PII note: `currentUserQuery` and `priorUserText` arrive as raw
        // user text. This matches every other on-device classifier path
        // (chat-mode router, tool selector, Stage A, KB extractor) —
        // local inference is the model boundary; PII redaction is for
        // surface badges and cloud-egress decisions only.
        let priorUserStr = priorUserText?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        var parts: [String] = []
        if !priorUserStr.isEmpty {
            parts.append("[USER_PRIOR]: \(priorUserStr)")
        }
        parts.append("[ASSISTANT_PRIOR]: \(prior)")
        parts.append("[USER]: \(currentUserQuery)")
        let userContent = parts.joined(separator: "\n")

        let messages = [
            LlamaChatMessage(role: "system", content: Self.systemPrompt),
            LlamaChatMessage(role: "user",   content: userContent),
        ]

        // Swap in the relational adapter, then race the generate against
        // a wall-clock timeout. The shared backend is on the critical
        // typing-response path for every Turn 2+; a pathological prompt
        // must NOT block the user's turn beyond `generateTimeoutMs`.
        //
        // `clearCache: true` flushes the KV cache so the prior context
        // from other calls doesn't leak into this classification.
        let generated: String
        do {
            try await backend.setAdapter(path: adapterPath, scale: 1.0)
            generated = try await Self.runWithTimeout(
                ms: Self.generateTimeoutMs
            ) {
                let (text, _, _) = try await self.backend.generate(
                    messages: messages,
                    maxTokens: 20,
                    temperature: 0,
                    stopSequences: ["\n", "<|im_end|>"],
                    clearCache: true,
                    outputMode: .text
                )
                return text
            }
        } catch {
            throw RelationalStrategyError.backendFailure(underlying: error)
        }

        let elapsedMs = Date().timeIntervalSince(start) * 1_000

        // Parse the generated text — check multi-word labels BEFORE
        // shorter prefixes to avoid partial-match false positives
        // ("negative" hitting inside "negative_continuation").
        let relationship = Self.parseTurnRelationship(from: generated)
        if relationship == nil {
            AppLog.intelligence.warning(
                "relational-head: unparseable output '\(generated.prefix(60), privacy: .public)' — defaulting to independent"
            )
        }

        // Confidence is a SYNTHETIC sentinel — the generative path has
        // no softmax to read. 1.0 means "parsed cleanly", 0.0 means
        // "fell back to .independent after a parse failure". Downstream
        // code must not threshold on this as if it were a calibrated
        // probability. When we ship the classifier-head variant, this
        // will become the actual argmax probability and `probabilities`
        // will carry the full vector.
        return RelationalOutcomes(
            turnRelationship: TurnRelationshipOutcome(
                value: relationship ?? .independent,
                confidence: relationship != nil ? 1.0 : 0.0,
                probabilities: []   // generative path; no softmax available
            ),
            slotAlignment: nil,     // deferred: sparse-positive trap (F20)
            stanceChange: nil,      // deferred: class collapse (F18)
            runtimeMs: elapsedMs
        )
    }

    /// Wall-clock timeout for `backend.generate`. 2000ms is ~10× the
    /// observed F16-on-device worst-case (~150-200ms for 20 tokens) and
    /// covers slow first-call cache warmup. Tunable if production turns
    /// up edge cases — but keep it small enough that the user's typing-
    /// response perception isn't degraded by a stuck classifier.
    private static let generateTimeoutMs: Int = 2_000

    /// Race a backend call against a wall-clock timeout. Throws
    /// `RelationalStrategyError.timeout` if the deadline fires first.
    /// Important: the underlying llama.cpp generate cannot actually be
    /// cancelled from Swift; on timeout we abandon the result but the
    /// generation completes in the background. This is acceptable
    /// because (a) the next caller's `clearCache: true` flushes its KV
    /// state and (b) the adapter swap is idempotent.
    private static func runWithTimeout<T: Sendable>(
        ms: Int,
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(ms) * 1_000_000)
                throw RelationalStrategyError.timeout(ms: ms)
            }
            guard let first = try await group.next() else {
                throw RelationalStrategyError.timeout(ms: ms)
            }
            group.cancelAll()
            return first
        }
    }

    // MARK: - Label parser

    /// Parse the model's text output back to a `TurnRelationship`.
    /// Returns nil on parse failure — caller logs + substitutes `.independent`.
    ///
    /// Order matters: check compound labels before single-token
    /// sub-strings, e.g. "negative_continuation" before "negative".
    static func parseTurnRelationship(from text: String) -> TurnRelationship? {
        let lower = text.lowercased()
        if lower.contains("affirmative_continuation") || lower.contains("affirmative continuation") {
            return .affirmativeContinuation
        }
        if lower.contains("negative_continuation") || lower.contains("negative continuation") {
            return .negativeContinuation
        }
        if lower.contains("clarification_answer") || lower.contains("clarification answer") {
            return .clarificationAnswer
        }
        if lower.contains("anaphoric") {
            return .anaphoric
        }
        if lower.contains("independent") {
            return .independent
        }
        return nil
    }
}

/// Errors raised by `ChatTemplateRelationalStrategy` during inference.
public enum RelationalStrategyError: Error, LocalizedError {
    /// The LlamaBackend `setAdapter` or `generate` call failed (propagated
    /// from the backend actor, e.g. model not loaded, context overflow,
    /// adapter file corrupted).
    case backendFailure(underlying: Error)

    /// `backend.generate` did not return within the wall-clock budget
    /// (`generateTimeoutMs`). The underlying llama.cpp call cannot be
    /// cancelled from Swift — it completes in the background — but the
    /// caller is unblocked.
    case timeout(ms: Int)

    public var errorDescription: String? {
        switch self {
        case .backendFailure(let e):
            return "RelationalStrategy backend failure: \(e.localizedDescription)"
        case .timeout(let ms):
            return "RelationalStrategy timed out after \(ms)ms"
        }
    }
}
