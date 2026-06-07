import Foundation
import Combine

/// ADR-023 Phase 2 — session-scoped conversation state that lives
/// alongside `ChatViewModel.messages`. The messages array is the
/// authoritative transcript; this object holds the **derived** signals
/// the intelligence layer reads on every turn:
///
///  - `pendingClarification` — the assistant's last reply asked a
///    clarification question (Verizon `.clarification` lane OR a
///    tool-action turn with missing required slots). The next user
///    message should be interpreted as the answer to that question
///    BEFORE running the full classifier.
///
///  - `pendingToolConfirmation` — the assistant's last reply proposed
///    a tool awaiting Confirm. A bare "yes" / "ok" from the user can
///    be routed straight to `confirmTool`. Cleared on confirm /
///    decline / any other message.
///
///  - `liveAgentRequestCount` — how many times the user has explicitly
///    asked for a human in this session. The `EscalateOnFrustrationNBA`
///    fires when this hits 2, even if the trained `emotional_state`
///    head is silent (calm-but-repeating power user).
///
///  - `didntWorkCount` — how many times the user has reported the
///    previous suggestion didn't work. Same NBA boost as above —
///    "fix it, didn't work, fix it again, didn't work" is the
///    canonical escalation signal regardless of affect.
///
/// **Scope**: one instance per chat session, held by `ChatViewModel`.
/// Reset on a new session (out of scope: persistence across app
/// launches — production privacy review required first).
///
/// **Concurrency**: `@MainActor` because the only writer is
/// `ChatViewModel` (also MainActor) and the only readers are SwiftUI +
/// MainActor-bound NBAs. No background actors touch it; that keeps
/// the @Published mutations free of cross-actor warnings.
///
/// **Observability**: `@Published` properties so the engineering trace
/// UI can render the live counters (planned: a footer row in
/// `UnderstandingTraceCard`).
@MainActor
public final class ConversationState: ObservableObject {

    // MARK: - Pending interactions

    /// Non-nil → the assistant's last reply asked a clarification
    /// question. The next user message should be tested as the answer
    /// before running the full classifier.
    @Published public private(set) var pendingClarification: PendingClarification?

    /// Non-nil → the assistant's last reply proposed a tool awaiting
    /// Confirm. The Confirm button itself drives `confirmTool`; this
    /// pointer lets a bare "yes" / "go ahead" route the same way
    /// without forcing the user to find the button.
    @Published public private(set) var pendingToolConfirmation: ToolDecision?

    // MARK: - Frustration accumulators (counter-based NBA boost)

    /// Count of live-agent escalation requests in this session.
    @Published public private(set) var liveAgentRequestCount: Int = 0

    /// Count of "didn't work" / "still broken" continuations.
    @Published public private(set) var didntWorkCount: Int = 0

    /// Total user turns observed (for trace + diagnostics).
    @Published public private(set) var userTurnCount: Int = 0

    // MARK: - ADR-024 — pairwise relational substrate

    /// Cached last-token hidden state for the prior USER query.
    /// Computed during the prior turn's understanding pass and stashed
    /// here at end-of-turn. Consumed by the `stance_change` head
    /// (`(h_t, h_{t-1}^u)`). Nil on the first turn of a session.
    @Published public private(set) var priorUserHidden: [Float]?

    /// Cached last-token hidden state for the prior ASSISTANT reply.
    /// Computed async after the assistant reply renders (off the
    /// user-visible latency path). Consumed by the `turn_relationship`
    /// and `slot_alignment` heads (`(h_t, h_{t-1}^a)`). Nil on the
    /// first turn or when the async compute hasn't completed.
    @Published public private(set) var priorAssistantHidden: [Float]?

    /// Cached TEXT of the prior assistant reply. Consumed by
    /// `ColBERTRetriever.retrieve(query:context:via:)` for anaphoric /
    /// negative-continuation follow-ups so the next turn's retrieval
    /// projects into the right corpus region. ADR-024 follow-up
    /// 2026-05-27 — the missing primitive that turned
    /// `.augmentRetrievalWithPriorAssistant` from a log statement into
    /// an actual augmentation. Distinct from `priorAssistantHidden`
    /// (which feeds the classifier heads); both are populated by
    /// `ChatViewModel.recordTurnSideEffects` at the end of each turn.
    @Published public private(set) var priorAssistantText: String?

    /// The tool intent the prior turn settled on (if any). Used by the
    /// `stance_change` head's REVERT path to know which intent's
    /// pending state to clear. Nil when the prior turn wasn't a tool
    /// action.
    @Published public private(set) var priorIntent: ToolIntent?

    /// The lane the prior turn resolved to. Used by the
    /// `NEGATIVE_CONTINUATION` path to keep the user in the same lane
    /// rather than re-routing to the same failing proposal.
    @Published public private(set) var priorLane: UnderstandingLane?

    /// `page_id` of the RAG unit cited on the most recent assistant
    /// turn (e.g. `"03.00"`). Mirrors the Python
    /// `ConversationState.prior_page_id` and feeds
    /// `RetrievalContext.priorPageID` so the dispatcher's short-followup
    /// override can reuse the prior page on bare wh-word / anaphoric-
    /// pronoun / slot-prefix queries. Step 5b Pre-flight Fix C
    /// iOS-integration follow-up.
    ///
    /// Populated by `ChatViewModel.recordTurnSideEffects` from the
    /// turn's `VerizonDispatchResult.citedRAGUnit?.pageID`. Nil when
    /// the prior turn produced no RAG citation (greeting, out-of-scope,
    /// live-agent escalation, clarify, ambiguous-yes-ignored).
    @Published public private(set) var priorPageID: String?

    /// Companion to `priorPageID` — `link_id` of the cited unit
    /// (e.g. `"network"`, `"restart-router"`, `"home"`). Carried for
    /// symmetry with the Python harness telemetry; the dispatcher's
    /// short-followup override gates on `priorPageID`, not this field,
    /// but downstream `link_id`-aware Step 7 work will consume it.
    @Published public private(set) var priorLinkID: String?

    /// ADR-023 Phase 3 finally lands (with ADR-024): per-intent slot
    /// accumulator. Filled across multiple turns when the user reveals
    /// slots piecemeal. Cleared on intent fire, on STANCE_REVERT, or
    /// after `slotStoreDecayTurns` turns of no activity for that intent.
    @Published public private(set) var slotStore: [ToolIntent: ToolArguments] = [:]

    /// Per-intent "last touched at turn" map. Used for decay so a long
    /// session doesn't accumulate stale slots forever.
    @Published public private(set) var slotStoreLastTouched: [ToolIntent: Int] = [:]

    /// Decay window: an intent's slots are cleared after this many
    /// turns of inactivity. Generous default — the user can always
    /// re-supply the slot if needed.
    ///
    /// **Worked example** (pinned to prevent off-by-one
    /// misinterpretation in code review):
    ///   - Slot touched at userTurnCount=3 → `lastTouched=3`
    ///   - Turn 4..8: cutoff is 4-5=-1, 5-5=0, 6-5=1, 7-5=2, 8-5=3.
    ///     `lastTouched (3) < cutoff` is false in every case → slot survives.
    ///   - Turn 9: cutoff is 9-5=4. `3 < 4` is true → slot decays.
    ///   - Net: slot survives 5 turns of inactivity (turns 4-8), decays
    ///     on the 6th turn after touch (turn 9). The constant matches
    ///     the natural reading "decay after 5 inactive turns".
    public static let slotStoreDecayTurns: Int = 5

    // MARK: - Init

    public init() {}

    // MARK: - Recording (called once per assistant turn)

    /// Apply the per-turn state delta. Called by `ChatViewModel` at the
    /// end of every fully-resolved turn (after the assistant reply has
    /// been appended). One call site → one source of truth for state
    /// transitions.
    ///
    /// - Parameters:
    ///   - userMessage: the raw user query that drove this turn (used
    ///     to update frustration accumulators).
    ///   - assistantLane: the lane the router landed on. Drives the
    ///     pendingClarification decision when the lane is `.clarification`.
    ///   - toolDecision: the tool decision (if any) attached to the
    ///     assistant reply. When non-nil AND `requiresConfirmation`,
    ///     sets `pendingToolConfirmation`. When the decision is the
    ///     compound-attachment variant (`isCompoundAttachment == true`)
    ///     we do NOT treat it as the primary expectation — bare "yes"
    ///     should follow the RAG instructions, not silently fire the
    ///     compound tool.
    ///   - missingSlots: for the `.toolAction` lane with required-but-
    ///     missing slots, the workflow records the in-flight intent +
    ///     missing slots so the next user message can fill them.
    ///   - originalQuery: the literal user query that produced the
    ///     clarification — passed through so a tool re-fire can keep
    ///     the original phrasing for trace.
    public func recordTurn(
        userMessage: String,
        assistantLane: UnderstandingLane,
        toolDecision: ToolDecision?,
        pendingToolConfirmation: ToolDecision? = nil,
        missingSlots: Set<Slot> = [],
        pendingIntent: ToolIntent? = nil,
        originalQuery: String? = nil
    ) {
        userTurnCount += 1

        // Frustration accumulators — pure-function detectors over the
        // user message text. Same detectors used by the NBA layer.
        if ConversationStateRecorder.isLiveAgentRequest(userMessage) {
            liveAgentRequestCount += 1
        }
        if ConversationStateRecorder.isDidntWorkContinuation(userMessage) {
            didntWorkCount += 1
        }

        // Pending tool confirmation — explicit state wins over visual
        // presentation. Composer answers can offer a real action via
        // inline copy ("Reply yes") without necessarily rendering a
        // ToolDecisionCard. Tool cards still set pending when they are
        // primary. Compound attachments remain secondary and never
        // become the target of a bare affirmative.
        let confirmationCandidate = pendingToolConfirmation
            ?? ((toolDecision?.isCompoundAttachment == false) ? toolDecision : nil)
        if let confirmationCandidate,
           confirmationCandidate.requiresConfirmation {
            self.pendingToolConfirmation = confirmationCandidate
        } else {
            self.pendingToolConfirmation = nil
        }

        // Pending clarification — set when the assistant's reply
        // explicitly opened a clarification slot. Two sources:
        //   1. Verizon `.clarification` lane (retrieval was ambiguous —
        //      we asked the user to pick).
        //   2. `.toolAction` lane with a required slot missing (we
        //      proposed a tool but need one more piece — the
        //      ClarifyMissingSlot NBA carries the question).
        if case .verizon(.clarification) = assistantLane {
            pendingClarification = PendingClarification(
                askedAt: Date(),
                source: .ragClarification,
                intent: pendingIntent,
                missingSlots: missingSlots,
                originalQuery: originalQuery ?? userMessage
            )
        } else if case .toolAction = assistantLane,
                  let intent = pendingIntent,
                  !missingSlots.isEmpty {
            pendingClarification = PendingClarification(
                askedAt: Date(),
                source: .missingSlot,
                intent: intent,
                missingSlots: missingSlots,
                originalQuery: originalQuery ?? userMessage
            )
        } else {
            // Any other terminal lane resolves the pending clarification
            // (the user changed topics or the assistant answered without
            // needing a follow-up). Clear so a stale pending doesn't
            // mis-route a future turn.
            pendingClarification = nil
        }

        // ADR-024 — slot store decay. Run after the counter increments
        // so the cutoff window is computed from the current turn index.
        applySlotStoreDecay()
    }

    // MARK: - ADR-024 — relational substrate mutators

    /// Cache the current turn's hidden states for the next turn's
    /// relational classification. Called twice per turn:
    ///   - User hidden cached during processTextQuery (before relational
    ///     pass) — synchronous, on the user-visible path.
    ///   - Assistant hidden cached AFTER the assistant reply renders
    ///     (async, off the critical path) — see ADR-024 §4.5.
    ///
    /// Both args optional so degraded builds (no relational adapter)
    /// can no-op on this path. When called, the previous cache values
    /// are overwritten — we keep only the most recent turn (per
    /// ADR-024 §12 Q3: N-back > 1 relational signal degrades sharply).
    public func cacheTurnHiddenStates(
        user: [Float]?,
        assistant: [Float]?
    ) {
        if let user { priorUserHidden = user }
        if let assistant { priorAssistantHidden = assistant }
    }

    /// Cache the literal TEXT of the just-rendered turn for retrieval
    /// augmentation. Separate from `cacheTurnHiddenStates` because:
    ///   - Text is available immediately (the assistant string).
    ///   - Hidden states require an extra backbone forward pass (Phase
    ///     8d) that may never land if the device is low on resources.
    /// Splitting them means retrieval augmentation works TODAY while the
    /// relational-head signal arrives later, independently.
    public func cacheTurnText(assistant: String?) {
        let trimmed = assistant?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        priorAssistantText = (trimmed?.isEmpty ?? true) ? nil : trimmed
    }

    /// Record the lane + intent the just-finished turn resolved to.
    /// Consumed by the next turn's relational fusion for stance/intent
    /// continuity decisions. Idempotent — called once per turn at the
    /// same place we call `recordTurn(...)`.
    public func recordPriorTurnContext(
        lane: UnderstandingLane?,
        intent: ToolIntent?
    ) {
        priorLane = lane
        priorIntent = intent
    }

    /// Record the RAG unit cited on the just-finished turn so the next
    /// turn's `RetrievalContext` can carry it forward. Source is
    /// `VerizonDispatchResult.citedRAGUnit` from the previous turn.
    /// Pass nil when the turn produced no RAG citation (greeting,
    /// out-of-scope, live-agent escalation, clarify, ambiguous-yes-
    /// ignored) — that's the explicit "clear prior page" signal.
    ///
    /// Idempotent. Called once per turn alongside
    /// `recordPriorTurnContext(...)` from
    /// `ChatViewModel.recordTurnSideEffects`.
    public func recordPriorPage(pageID: String?, linkID: String?) {
        priorPageID = pageID
        priorLinkID = linkID
    }

    /// Merge a slot value into the per-intent slot store. Called by
    /// the clarification-recovery and stance-OVERRIDE paths. Touches
    /// the per-intent "last touched" timestamp so decay accounts for
    /// the activity.
    public func accumulateSlot(
        intent: ToolIntent,
        key: String,
        value: String
    ) {
        var existing = slotStore[intent]?.values ?? [:]
        existing[key] = value
        slotStore[intent] = ToolArguments(existing)
        slotStoreLastTouched[intent] = userTurnCount
    }

    /// Read the accumulated slots for an intent. Returns `.empty` if
    /// the intent has no stored slots.
    public func slots(for intent: ToolIntent) -> ToolArguments {
        slotStore[intent] ?? .empty
    }

    /// Drop the slot store for a specific intent — called on intent
    /// fire (don't re-fire with stale slots) AND on STANCE_REVERT
    /// (the user changed their mind about THIS intent).
    public func clearSlotStore(for intent: ToolIntent) {
        slotStore.removeValue(forKey: intent)
        slotStoreLastTouched.removeValue(forKey: intent)
    }

    /// Apply the decay rule: clear any intent's slots that haven't
    /// been touched in `slotStoreDecayTurns` turns. Called once per
    /// turn, AFTER `userTurnCount` has incremented.
    private func applySlotStoreDecay() {
        let cutoff = userTurnCount - Self.slotStoreDecayTurns
        let staleIntents = slotStoreLastTouched.compactMap { (intent, lastTouched) -> ToolIntent? in
            lastTouched < cutoff ? intent : nil
        }
        for intent in staleIntents {
            clearSlotStore(for: intent)
        }
    }

    // MARK: - Manual mutators (called from confirm / decline paths)

    /// Called by `ChatViewModel.confirmTool` and `declineTool`. The
    /// tool either fired or was dismissed — the pending pointer is no
    /// longer valid.
    public func clearPendingToolConfirmation() {
        pendingToolConfirmation = nil
    }

    /// Called after `tryFulfillPendingClarification` successfully
    /// recovered an answer. The original question has been answered;
    /// the pending pointer is cleared so the next turn starts fresh.
    public func clearPendingClarification() {
        pendingClarification = nil
    }

    /// Reset everything — used when the chat session starts fresh
    /// (e.g., a future "New conversation" button).
    public func reset() {
        pendingClarification = nil
        pendingToolConfirmation = nil
        liveAgentRequestCount = 0
        didntWorkCount = 0
        userTurnCount = 0
        // ADR-024 — relational cache + slot store are session-scope;
        // reset clears them too.
        priorUserHidden = nil
        priorAssistantHidden = nil
        priorAssistantText = nil
        priorIntent = nil
        priorLane = nil
        priorPageID = nil
        priorLinkID = nil
        slotStore = [:]
        slotStoreLastTouched = [:]
    }

    /// Capture an immutable snapshot of the counters + pending-pointer
    /// presence for code that needs a Sendable value crossing actor
    /// boundaries (notably the NBA matchers, which are pure-function
    /// `Sendable` conformances). The snapshot is a point-in-time
    /// projection — mutating the state after taking it doesn't
    /// retroactively update the snapshot.
    public var snapshot: ConversationSnapshot {
        ConversationSnapshot(
            liveAgentRequestCount: liveAgentRequestCount,
            didntWorkCount: didntWorkCount,
            userTurnCount: userTurnCount,
            hasPendingClarification: pendingClarification != nil,
            hasPendingToolConfirmation: pendingToolConfirmation != nil,
            // ADR-024 — pass enough of the relational substrate that
            // the router can reason about prior-turn provenance without
            // crossing the MainActor boundary.
            priorIntent: priorIntent,
            priorLane: priorLane,
            priorAssistantText: priorAssistantText,
            hasPriorAssistantHidden: priorAssistantHidden != nil,
            hasPriorUserHidden: priorUserHidden != nil
        )
    }
}

/// Sendable point-in-time projection of `ConversationState`. The NBA
/// layer reads this — it gets only what it needs (read-only counters +
/// pending flags), no live reference to the mutable observable.
///
/// Why a separate type: `ConversationState` is `@MainActor` (ties to
/// SwiftUI and ChatViewModel). `NextBestAction` matchers are `Sendable`
/// pure functions that must cross actor boundaries cleanly. Passing the
/// live class would force every matcher to hop the main actor; passing
/// a snapshot keeps them pure.
public struct ConversationSnapshot: Sendable, Equatable {
    public let liveAgentRequestCount: Int
    public let didntWorkCount: Int
    public let userTurnCount: Int
    public let hasPendingClarification: Bool
    public let hasPendingToolConfirmation: Bool
    // ADR-024 — prior-turn provenance for router fusion.
    public let priorIntent: ToolIntent?
    public let priorLane: UnderstandingLane?
    /// Prior assistant reply (verbatim, trimmed). Consumed by
    /// `applyPostDecisionActions` to populate `RetrievalContext` when
    /// the router emits `.augmentRetrievalWithPriorAssistant`. Nil on
    /// the first turn or when the prior assistant message was empty.
    public let priorAssistantText: String?
    public let hasPriorAssistantHidden: Bool
    public let hasPriorUserHidden: Bool

    public init(
        liveAgentRequestCount: Int = 0,
        didntWorkCount: Int = 0,
        userTurnCount: Int = 0,
        hasPendingClarification: Bool = false,
        hasPendingToolConfirmation: Bool = false,
        priorIntent: ToolIntent? = nil,
        priorLane: UnderstandingLane? = nil,
        priorAssistantText: String? = nil,
        hasPriorAssistantHidden: Bool = false,
        hasPriorUserHidden: Bool = false
    ) {
        self.liveAgentRequestCount = liveAgentRequestCount
        self.didntWorkCount = didntWorkCount
        self.userTurnCount = userTurnCount
        self.hasPendingClarification = hasPendingClarification
        self.hasPendingToolConfirmation = hasPendingToolConfirmation
        self.priorIntent = priorIntent
        self.priorLane = priorLane
        self.priorAssistantText = priorAssistantText
        self.hasPriorAssistantHidden = hasPriorAssistantHidden
        self.hasPriorUserHidden = hasPriorUserHidden
    }

    /// Empty snapshot — used in tests and as the default when no
    /// session state is being tracked.
    public static let empty = ConversationSnapshot()
}

/// A pending clarification question awaiting the user's answer.
/// Carries enough context for `tryFulfillPendingClarification` to
/// re-fire the original intent with combined slots.
public struct PendingClarification: Sendable, Equatable {
    public enum Source: String, Sendable, Equatable {
        /// `.verizon(.clarification)` lane — Stage A flagged retrieval
        /// as ambiguous; we asked the user to disambiguate.
        case ragClarification
        /// `.toolAction` lane with a required slot missing — we
        /// proposed the tool but need one more piece.
        case missingSlot
    }

    public let askedAt: Date
    public let source: Source
    /// The tool we were trying to fire. Nil for `.ragClarification`
    /// when the lane decision didn't pin an intent.
    public let intent: ToolIntent?
    /// Which slots are still empty.
    public let missingSlots: Set<Slot>
    /// The user's original query that triggered the clarification —
    /// preserved so the re-fire trace shows the literal phrasing.
    public let originalQuery: String

    public init(
        askedAt: Date,
        source: Source,
        intent: ToolIntent?,
        missingSlots: Set<Slot>,
        originalQuery: String
    ) {
        self.askedAt = askedAt
        self.source = source
        self.intent = intent
        self.missingSlots = missingSlots
        self.originalQuery = originalQuery
    }
}

// MARK: - Pure-function recorder helpers

/// Pure functions that ConversationState uses internally AND that the
/// NBA layer can call directly without an actor hop. Each helper is
/// deterministic in its single argument — easy to test, easy to reason
/// about. No I/O, no LFM calls, no state.
public enum ConversationStateRecorder {

    /// True when the user message is an explicit live-agent / human
    /// handoff request. Conservative — false positives here would push
    /// real KB questions into the escalation chip, so the patterns are
    /// scoped tight to phrases that unambiguously ask for a person.
    ///
    /// Anchored to the 50-conversation Verizon corpus: covers "Agent",
    /// "Chat agent", "Speak to live agent", "Talk to a representative",
    /// "transfer me", "real person", "human".
    public static func isLiveAgentRequest(_ message: String) -> Bool {
        let normalized = normalize(message)
        // Whole-word match against the closed phrase set — substring
        // matching would false-positive on "agentic", "agency", etc.
        return liveAgentPatterns.firstMatch(
            in: normalized,
            range: NSRange(normalized.startIndex..., in: normalized)
        ) != nil
    }

    /// True when the user message is a "previous suggestion didn't
    /// work" continuation. These are the queries that, standalone,
    /// classify as out-of-scope today — the prior turn's context is
    /// what makes them coherent.
    public static func isDidntWorkContinuation(_ message: String) -> Bool {
        let normalized = normalize(message)
        return didntWorkPatterns.firstMatch(
            in: normalized,
            range: NSRange(normalized.startIndex..., in: normalized)
        ) != nil
    }

    /// True when the user message is a bare affirmative ("yes", "yep",
    /// "go ahead", "do it"). Used by the pending-clarification path to
    /// recognise "yes" as "confirm the previous proposal" rather than
    /// route it as a fresh out-of-scope query.
    public static func isBareAffirmative(_ message: String) -> Bool {
        let normalized = normalize(message)
        return bareAffirmativePatterns.firstMatch(
            in: normalized,
            range: NSRange(normalized.startIndex..., in: normalized)
        ) != nil
    }

    /// True when the user message is a bare negative/cancel response.
    /// Used only when a pending confirmation exists; otherwise it is an
    /// ambiguous short turn and must not trigger side effects.
    public static func isBareNegative(_ message: String) -> Bool {
        let normalized = normalize(message)
        return bareNegativePatterns.firstMatch(
            in: normalized,
            range: NSRange(normalized.startIndex..., in: normalized)
        ) != nil
    }

    // MARK: - Private

    private static func normalize(_ message: String) -> String {
        message
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    /// Whole-phrase live-agent triggers. Anchored to the 50-conversation
    /// Verizon corpus (CHAT_017/019/026/028/046/050 — every instance
    /// uses one of these phrases).
    // swiftlint:disable:next force_try
    private static let liveAgentPatterns: NSRegularExpression = try! NSRegularExpression(
        pattern: #"""
        (?xi)
        \b(
            (?:live\s+)?(?:agent|representative|rep)\b
          | (?:chat\s+(?:with\s+)?(?:a\s+)?(?:agent|rep|person|human))
          | (?:talk\s+(?:to\s+|with\s+)(?:a\s+)?(?:agent|rep|person|human|real\s+person))
          | (?:speak\s+(?:to\s+|with\s+)(?:a\s+)?(?:agent|rep|person|human|real\s+person))
          | (?:connect\s+(?:me|to\s+(?:a\s+)?(?:agent|rep|person|human|real\s+person)))
          | (?:need\s+(?:a\s+)?(?:real\s+)?(?:agent|rep|person|human|representative)(?:\s+(?:for|to)\s+help)?)
          | (?:live\s+person)
          | (?:(?:some\s+one|someone|somebody|anyone)\s+(?:can\s+)?help(?:\s+me)?)
          | (?:can\s+(?:some\s+one|someone|somebody|anyone)\s+help(?:\s+me)?)
          | (?:transfer\s+me)
          | (?:get\s+me\s+(?:a\s+)?(?:agent|rep|person|human))
          | (?:real\s+person)
          | (?:customer\s+service)
        )\b
        """#,
        options: [.allowCommentsAndWhitespace]
    )

    /// Whole-phrase "didn't work" patterns. Anchored to CHAT_013/033/
    /// 042/044 — common phrasings the production corpus surfaced.
    // swiftlint:disable:next force_try
    private static let didntWorkPatterns: NSRegularExpression = try! NSRegularExpression(
        pattern: #"""
        (?xi)
        \b(
            (?:didn[''’]?t|did\s+not)\s+work
          | (?:doesn[''’]?t|does\s+not)\s+work
          | (?:not\s+working)
          | (?:still\s+(?:broken|down|not\s+working|doesn[''’]?t|isn[''’]?t))
          | (?:tried\s+that)
          | (?:that\s+didn[''’]?t\s+(?:help|work))
          | (?:no\s+luck)
          | (?:same\s+(?:problem|issue))
        )\b
        """#,
        options: [.allowCommentsAndWhitespace]
    )

    /// Bare affirmatives. Tight on purpose — "yes please send it" is
    /// NOT a bare affirmative (it has a verb the LLM can route
    /// independently). Anchored on the production "Yes" / "ok" / "go
    /// ahead" continuations.
    // swiftlint:disable:next force_try
    private static let bareAffirmativePatterns: NSRegularExpression = try! NSRegularExpression(
        pattern: #"""
        ^\s*(?:
            yes|yeah|yep|yup|sure|ok|okay|k|fine
          | (?:go\s+ahead)|(?:do\s+it)|(?:please\s+do)
          | (?:sounds\s+good)|(?:that\s+works)
        )\s*[.!]?\s*$
        """#,
        options: [.allowCommentsAndWhitespace, .caseInsensitive]
    )

    // swiftlint:disable:next force_try
    private static let bareNegativePatterns: NSRegularExpression = try! NSRegularExpression(
        pattern: #"""
        ^\s*(?:
            no|nope|nah|cancel|stop
          | (?:do\s+not)|don[''’]?t|dont
          | (?:never\s+mind)|(?:no\s+thanks)|(?:no\s+thank\s+you)
        )\s*[.!]?\s*$
        """#,
        options: [.allowCommentsAndWhitespace, .caseInsensitive]
    )
}
