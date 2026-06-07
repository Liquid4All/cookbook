import Foundation

/// Carries optional cross-turn signal into a single `ColBERTRetriever`
/// call. The first-class primitive that replaces the
/// "augmentRetrievalWithPriorAssistant" log-only no-op (ADR-024 §4.6,
/// follow-up plumbing 2026-05-27).
///
/// Why a value type rather than a `String?` parameter:
///
///   - Sendable, `Equatable` — safe to capture in tests, to compare in
///     assertions, to thread through actor boundaries without copies.
///   - Extensible — future signals (e.g. weighted prior-user text,
///     turn-distance decay, intent-conditioned re-weighting) land here
///     without re-shaping every call site.
///   - The retrieval surface ALWAYS takes a `RetrievalContext`. Empty
///     means "single-turn behaviour" rather than "I forgot to wire it"
///     — the type system makes the contract obvious.
///
/// **Scope discipline.** Only signals that ColBERT can ACTUALLY consume
/// belong here. Lane, intent, slot-store snapshots do NOT — those route
/// through `ConversationSnapshot` to the router, not to the retriever.
public struct RetrievalContext: Sendable, Equatable {
    /// The literal text of the assistant's most recent reply. When
    /// non-nil, `ColBERTRetriever` prepends it to the user's query
    /// before encoding so anaphoric / negative-continuation follow-ups
    /// project into the right corpus region.
    ///
    /// Source: `ConversationState.priorAssistantText`, populated by
    /// `ChatViewModel.recordTurnSideEffects` after each assistant
    /// message renders.
    public let priorAssistantText: String?

    /// `page_id` of the RAG unit cited on the most recent assistant
    /// turn (e.g. `"03.00"`). When set, the dispatcher's short-followup
    /// override may force the new turn's evidence back to this page —
    /// see `VerizonChatDispatcher.runComposerPipeline` and
    /// `TelcoPolicyEngine` for the gating rule. Mirrors the Python harness's
    /// `ConversationState.prior_page_id` so the Step 5b acceptance gates
    /// hold byte-for-byte against the iOS dispatcher.
    ///
    /// Source: `ConversationState.priorPageID`, populated by
    /// `ChatViewModel.recordTurnSideEffects` from the prior turn's
    /// `VerizonDispatchResult.citedRAGUnit?.pageID`.
    public let priorPageID: String?

    /// Companion to `priorPageID` — the `link_id` of the cited unit
    /// (e.g. `"network"`, `"restart-router"`). Mirrors
    /// `ConversationState.priorLinkID`. Carried for symmetry with the
    /// Python `simulate_turn` telemetry; the dispatcher does not gate
    /// on it today but downstream phases (Step 7 tool-alias-by-link)
    /// may.
    public let priorLinkID: String?

    public init(
        priorAssistantText: String? = nil,
        priorPageID: String? = nil,
        priorLinkID: String? = nil
    ) {
        self.priorAssistantText = priorAssistantText
        self.priorPageID = priorPageID
        self.priorLinkID = priorLinkID
    }

    /// Sentinel for "no cross-turn signal." Default for every call site
    /// that hasn't opted in to augmentation, so back-compat is automatic.
    public static let empty = RetrievalContext(
        priorAssistantText: nil,
        priorPageID: nil,
        priorLinkID: nil
    )

    /// True iff at least one augmentation signal is set. Cheaper than
    /// pattern-matching at call sites — every consumer reads
    /// `context.isEmpty == false` then checks specific fields.
    public var isEmpty: Bool {
        priorAssistantText == nil && priorPageID == nil && priorLinkID == nil
    }
}
