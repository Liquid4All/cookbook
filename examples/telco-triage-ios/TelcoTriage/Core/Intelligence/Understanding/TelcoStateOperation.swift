import Foundation

/// ADR-029 §7 — the explicit dialogue **state-operation** layer.
///
/// # First principles (DST / SOM-DST)
///
/// Classifying *what the user is asking about* (the 9 shared understanding heads)
/// is not the same question as *what should happen to the dialogue state*. Modern
/// dialogue-state-tracking (SOM-DST) frames each turn as an operation over the
/// conversation's memory — carry the active goal over, reuse the evidence we
/// already surfaced, retrieve fresh, clear on a topic switch, repair, confirm, or
/// escalate. Conversational-search work (TREC CAsT) shows the dominant failure is
/// treating a context-dependent turn as if it were standalone (or, symmetrically,
/// reusing stale context when the turn moved on).
///
/// Before this layer existed those decisions were *implicit*: the policy engine
/// re-derived them inline from the relation label and a page-id heuristic, and
/// the harness inferred "reuse" from page-id continuity. That made the product
/// metric an artifact of page continuity rather than a measurement of the actual
/// state decision.
///
/// `TelcoStateOperation` makes the decision **first-class, explicit, serialized,
/// and scorable**. It is resolved once per turn from the turn relation, the
/// deterministic topic prior, and the dialogue blackboard's ground-truth state —
/// *before* retrieval and route policy — and then both retrieval and the policy
/// engine obey it.
///
/// # Separation of concerns (the contract)
///
/// * The **relation head** informs the operation but is not its only source: the
///   deterministic prior (explicit-human / account / unsupported lexicons) and a
///   structural short-turn signal also feed it (principle 3).
/// * The **9 shared heads** inform route / safety / tool / cloud inside the policy
///   engine; they do **not** own context reuse (principle 4).
/// * The **blackboard** stores prior state and the selected operation each turn
///   (principle 5).
/// * **Retrieval obeys the operation** via `TelcoRetrievalStrategy` (principle 6).
public enum TelcoStateOperation: String, CaseIterable, Sendable, Equatable {
    /// Continue the active task; the prior page/link remain the working set.
    case carryoverActiveTask = "carryover_active_task"
    /// Reuse the specific evidence we already surfaced (repair / ambiguous
    /// continuation) rather than re-retrieving.
    case reuseActiveEvidence = "reuse_active_evidence"
    /// Retrieve fresh, but bias the ranker toward the prior page/link (step focus
    /// inside the same task).
    case retrieveWithPriorBias = "retrieve_with_prior_bias"
    /// Retrieve fresh, ignoring any stale context (a continuation that has no
    /// active task to lean on).
    case retrieveFresh = "retrieve_fresh"
    /// A new, independent task supersedes whatever came before.
    case updateNewTask = "update_new_task"
    /// The user explicitly pivoted ("actually, instead…") — drop the active
    /// page/link/tool and start clean.
    case clearContextTopicSwitch = "clear_context_topic_switch"
    /// The user is answering a clarification the assistant asked — resolve the
    /// pending slot and continue the task.
    case clarificationAnswer = "clarification_answer"
    /// The turn is under-specified and there is no valid state to continue —
    /// ask for the missing detail instead of guessing.
    case askClarification = "ask_clarification"
    /// "I can't find it" — the named UI element is missing; reuse the active task.
    case repairCannotFind = "repair_cannot_find"
    /// "That didn't work" — the prior instruction failed; reuse the active task.
    case repairFailed = "repair_failed"
    /// Affirmative continuation of a pending action.
    case confirmationYes = "confirmation_yes"
    /// Negative continuation — the user declined the pending action.
    case confirmationNo = "confirmation_no"
    /// An unambiguous request for a human.
    case directEscalation = "direct_escalation"
    /// A destination the local corpus structurally cannot serve (account /
    /// billing / order / unsupported external action) — avoid local RAG.
    case unsupportedBackendOrOutOfScope = "unsupported_backend_or_out_of_scope"
}

/// How retrieval should behave under the resolved state operation. This is the
/// data-access projection of the operation (principle 6). The lexical retriever
/// is already state-conditioned (it consumes the prior page/link via
/// `RetrievalContext`, which the blackboard clears on topic-switch / new-task),
/// so `fresh` and `priorBias` are realized by the existing context plumbing;
/// `reusePrior` is realized by `TelcoPolicyResolution.reuseActiveEvidence`; and
/// `none` is realized by the terminal route ignoring the (unused) candidate.
public enum TelcoRetrievalStrategy: String, Sendable, Equatable {
    /// No retrieval is consulted (escalation / unsupported / clarification).
    case none
    /// Fresh retrieval, ignoring stale context.
    case fresh
    /// Fresh retrieval biased toward the prior page/link.
    case priorBias = "prior_bias"
    /// Reuse the active task's prior evidence; do not re-retrieve.
    case reusePrior = "reuse_prior"
}

/// The resolved dialogue-state decision for one turn: the operation, its derived
/// retrieval strategy, and a single audit code. Serialized into the blackboard
/// and the phone-flow harness report.
public struct TelcoStateResolution: Sendable, Equatable {
    public let operation: TelcoStateOperation
    public let retrieval: TelcoRetrievalStrategy
    /// Audit code for *why* this operation was chosen (e.g. `ambiguous_no_state`,
    /// `account_lexical`). Distinct from the policy engine's route reason.
    public let reason: String

    public init(operation: TelcoStateOperation, retrieval: TelcoRetrievalStrategy, reason: String) {
        self.operation = operation
        self.retrieval = retrieval
        self.reason = reason
    }
}

/// Resolves the dialogue-state operation for a turn, deterministically, from the
/// turn relation, the deterministic topic prior, and the blackboard state
/// snapshot. Pure and side-effect-free so it is unit-testable in isolation.
///
/// # Precedence (ordered by the cost of getting the state decision wrong)
///
/// 1. Explicit human request — escalate; nothing else can override it.
/// 2. Account/billing/order, or unsupported external action — the local corpus
///    cannot serve it; no local retrieval.
/// 3. Greeting — a phatic opener; treated as a fresh, contentless turn so the
///    policy's greeting rung can answer.
/// 4. Repair (didn't-work / can't-find) — reuse the active task's evidence when
///    the blackboard holds one to reuse; otherwise ask what failed.
/// 5. Negative confirmation — the user declined; ground normally. State-gated
///    (§7.2): valid only when a tool confirmation is actually pending — a "no"
///    with nothing to decline is not a confirmation act.
/// 6. Clarification answer — the short reply *answers* our question; continue the
///    task. State-gated (§7.2): valid only when the prior app route actually asked
///    a clarification (`priorRouteWasClarify`); a relation-head
///    `clarification_answer` with no pending clarification is an impossible output
///    (we never asked) and is dropped. Checked *before* the structural short-turn
///    gate so a 1–2 word answer to our own question is never mistaken for a new
///    ambiguous turn.
/// 7. Affirmative confirmation — ground the pending action. State-gated (§7.2):
///    valid only when a tool confirmation is actually pending — a bare "yes" with
///    no pending tool must never execute or confirm.
/// 8. Topic switch — clear context, retrieve fresh.
/// 9. Ambiguous short turn — by relation label *or* a structural ≤2 content-token
///    signal (the same length doctrine the situation overlay uses). With a valid
///    active task, carry it over (reuse); otherwise ask for the missing detail
///    rather than grounding a fresh page on two words.
/// 10. Continuation / step-focus — stay on the active task with prior bias, or
///     retrieve fresh if there is no active task.
/// 11. Independent new task (or no relation) — retrieve fresh.
public enum TelcoStateOperationResolver {
    /// Content-token count at or below which a turn is treated as structurally
    /// ambiguous. Mirrors `situation_eval.classify_situation`'s `ntok <= 2` rule
    /// — a structural length property, not a lexical pattern (ADR-029 §7).
    public static let shortTurnTokenCeiling = 2

    public static func resolve(
        query: String,
        relation: TelcoTurnRelation?,
        prior: TelcoDeterministicPrior,
        state: TelcoDialogueStateSnapshot
    ) -> TelcoStateResolution {
        // 1. Exact phatic greeting — the highest-precision opener. A bare
        //    "hello"/"hi" is never a real escalation or backend request, so it
        //    precedes the (noisier) relation-head escalation: an LFM mislabel of
        //    "Hello" as escalation_request must not hand the customer to a live
        //    agent. An exact greeting never matches the human-request lexicon, so
        //    ranking it first can never mask a genuine explicit handoff (§7).
        if prior.greeting {
            return .init(operation: .updateNewTask, retrieval: .fresh, reason: "greeting")
        }

        // 2. Explicit human request (relation head or its canonical lexicon).
        if relation == .escalationRequest || prior.explicitHumanRequest {
            return .init(operation: .directEscalation, retrieval: .none, reason: "explicit_handoff")
        }

        // 3. Backend / out-of-corpus destinations.
        if prior.accountOrBilling {
            return .init(
                operation: .unsupportedBackendOrOutOfScope, retrieval: .none, reason: "account_lexical"
            )
        }
        if prior.unsupportedExternal {
            return .init(
                operation: .unsupportedBackendOrOutOfScope,
                retrieval: .none,
                reason: "unsupported_external"
            )
        }

        // 4. Repair — reuse the active task's evidence (or ask if nothing to reuse).
        if relation == .repairFailed {
            return repairResolution(state: state, reason: "repair_failed", op: .repairFailed)
        }
        if relation == .repairCannotFind {
            return repairResolution(state: state, reason: "repair_cannot_find", op: .repairCannotFind)
        }

        // 5. Negative confirmation — declined; ground normally. STATE-GATE
        //    (ADR-029 §7.2): a `confirmation_no` is only a real dialogue act when a
        //    tool confirmation is actually pending. A relation-head "no" with
        //    nothing to decline is an impossible output; drop it and let the turn
        //    fall through to the short-turn / continuation gates rather than
        //    treating it as a declined action.
        if relation == .confirmationNo, state.pendingToolID != nil {
            return .init(operation: .confirmationNo, retrieval: .fresh, reason: "confirmation_no")
        }

        // 6. Clarification answer — a short reply ANSWERS the question; ground it.
        //    STATE-GATE (ADR-029 §7.2): valid ONLY when the prior app route actually
        //    asked a clarification (`priorRouteWasClarify`). A relation-head
        //    `clarification_answer` with no pending clarification is an impossible
        //    output — we never asked, so the turn cannot be answering us — and is
        //    dropped here so it falls through to the short-turn / continuation gates
        //    (which ask about a vague fragment instead of grounding it as an answer
        //    to a question we did not pose). Guarded BEFORE the short-turn gate so a
        //    genuine 1–2 word answer to our own clarify still grounds, never re-asks.
        if state.priorRouteWasClarify {
            return .init(
                operation: .clarificationAnswer,
                retrieval: state.hasActiveTask ? .priorBias : .fresh,
                reason: relation == .clarificationAnswer ? "clarification_answer" : "answering_prior_clarify"
            )
        }

        // 7. Affirmative confirmation — ground the pending action. STATE-GATE
        //    (ADR-029 §7.2): a `confirmation_yes` is only a real dialogue act when a
        //    tool confirmation is actually pending. A bare "yes" with no pending
        //    tool must NEVER execute or confirm; drop the impossible relation-head
        //    output and fall through to the short-turn gate (which carries an active
        //    task over, or asks). This mirrors the doctrine the deterministic
        //    blackboard fallback already applies to bare affirmatives.
        if relation == .confirmationYes, state.pendingToolID != nil {
            return .init(
                operation: .confirmationYes,
                retrieval: state.hasActiveTask ? .priorBias : .fresh,
                reason: "confirmation_yes"
            )
        }

        // 8. Topic switch — clear and retrieve fresh.
        if relation == .topicSwitch {
            return .init(
                operation: .clearContextTopicSwitch, retrieval: .fresh, reason: "topic_switch"
            )
        }

        // 9. Ambiguous short turn — relation label or structural ≤2-token signal.
        if relation == .ambiguousShortTurn || isStructurallyShort(query) {
            if state.hasActiveTask {
                return .init(
                    operation: .reuseActiveEvidence, retrieval: .reusePrior, reason: "ambiguous_carryover"
                )
            }
            return .init(operation: .askClarification, retrieval: .none, reason: "ambiguous_no_state")
        }

        // 10. Continuation / step-focus — stay on task with prior bias, else fresh.
        switch relation {
        case .continuationSameTask, .continuationSameSection:
            return state.hasActiveTask
                ? .init(
                    operation: .carryoverActiveTask, retrieval: .priorBias,
                    reason: relation!.rawValue
                )
                : .init(operation: .retrieveFresh, retrieval: .fresh, reason: "continuation_no_task")
        case .stepFocus:
            return state.hasActiveTask
                ? .init(
                    operation: .retrieveWithPriorBias, retrieval: .priorBias, reason: "step_focus"
                )
                : .init(operation: .retrieveFresh, retrieval: .fresh, reason: "step_focus_no_task")
        default:
            break
        }

        // 11. Independent new task, or no relation available.
        return .init(operation: .updateNewTask, retrieval: .fresh, reason: "independent_new_task")
    }

    private static func repairResolution(
        state: TelcoDialogueStateSnapshot,
        reason: String,
        op: TelcoStateOperation
    ) -> TelcoStateResolution {
        // With an active task there is evidence to re-guide; otherwise the policy
        // engine will ask what failed.
        state.hasActiveTask
            ? .init(operation: op, retrieval: .reusePrior, reason: reason)
            : .init(operation: op, retrieval: .none, reason: "\(reason)_no_task")
    }

    /// True when the turn carries ≤ `shortTurnTokenCeiling` content tokens. A
    /// structural length property (the dominant feature for these turns), not a
    /// keyword match — the same rule the situation overlay uses to label
    /// `ambiguous_short_turn`.
    static func isStructurallyShort(_ query: String) -> Bool {
        let tokens = query
            .split(whereSeparator: { $0.isWhitespace })
            .filter { !$0.isEmpty }
        return tokens.count <= shortTurnTokenCeiling && !tokens.isEmpty
    }
}
