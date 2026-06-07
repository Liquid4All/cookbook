import Foundation

/// ADR-029 §1 — the single authoritative route-policy owner.
///
/// # First principles
///
/// Each turn must produce exactly one discrete control action (the
/// `ComposerRoute`). The inputs are heterogeneous evidence sources with
/// *different epistemic status*, and the historical defect (ADR-029
/// "policy-owner bypass") was that the route was derived in two separate
/// places, each consuming a subset of the evidence, while the turn-relation
/// label and the dialogue blackboard's frustration/failed-attempt counters
/// were computed and then discarded.
///
/// This engine consolidates that decision. It consumes:
///
/// * **Turn relation** (`TelcoTurnRelation`) — the authoritative *dialogue
///   act* of this turn relative to prior state (escalation, repair,
///   confirmation, topic switch, …). Sourced from the ADR-028 head when
///   available, or the blackboard's sanctioned deterministic fallback
///   classifier when the head is absent/low-confidence.
/// * **Shared understanding** (`TelcoSharedUnderstanding`) — the *semantic
///   support signals* (intent, complexity, routing lane, escalation risk,
///   cloud needs, PII). Each is a confidence-bearing head outcome.
/// * **Dialogue state** (`TelcoDialogueStateSnapshot`) — accumulated,
///   deterministic dialogue *facts* (active task, prior page/link, pending
///   tool, remediation budget consumed). These are ground truth, not
///   predictions.
/// * **Evidence** — the retriever's selected `RAGUnit` (and candidates).
/// * **Capability** — `ToolRegistry` + `ToolAliasMap`: does a real
///   executable action exist for the selected unit.
/// * **Deterministic prior** (`TelcoDeterministicPrior`) — a *single-sourced*
///   typed fallback for topic signals (account/billing, unsupported external
///   action, greeting) that no relation label carries. Consulted only when a
///   confident head signal is unavailable. This is the same degraded-mode
///   pattern ADR-028 sanctions for the relation head — not an ad-hoc
///   keyword patch sprinkled into control flow.
///
/// # The lattice is ordered by the *cost of being wrong*
///
/// Routing is a risk-asymmetric decision. The engine resolves the highest
/// product-risk action first:
///
/// 1. Sensitive-data safety (never route payment/identity into RAG/tool).
/// 2. Human handoff requested or required (stalling an angry/blocked user is
///    the most expensive error in support).
/// 3. Affective escalation (complaint/churn/urgent).
/// 4. Human-required complexity.
/// 5. Exhausted repair (remediation budget spent → escalate).
/// 6. Account/backend deflection (answer needs live systems we don't hold).
/// 7. Unsupported external action.
/// 8. In-progress repair (reuse the active unit; don't re-retrieve).
/// 9. Clarification (cheap to ask; only when nothing groundable remains).
/// 10. Greeting.
/// 11. Grounded answer (evidence present; capability gate decides
///     rag / answer+action / tool).
/// 12. No local answer.
///
/// # Invariants
///
/// * The engine performs **no raw-query lexical routing**. Query text is
///   used only by `inferQueryMood` (a structural, parity-pinned classifier)
///   for the capability gate. All escalation/account/repair knowledge enters
///   as a typed relation label, a gated head outcome, or the single-sourced
///   `TelcoDeterministicPrior`.
/// * Shared-head outcomes are **gated evidence inside the grounded rung**,
///   never top-level route overrides (ADR-029 §2). A low-confidence or
///   conflicting head leaves the deterministic decision intact.
/// * Tool execution requires both a supported affordance *and* a registered
///   tool. Query mood alone never manufactures a confirmation handshake.
public enum TelcoPolicyEngine {

    /// Number of remediation attempts on the active task after which the
    /// engine stops re-guiding and hands the customer to a human. A repair
    /// turn (`repair_failed` / `repair_cannot_find`) consumes one unit of
    /// budget; once two attempts on the same task have failed, continuing to
    /// re-explain the same page is worse than escalating.
    public static let repairEscalationThreshold = 2

    /// Fraction of the query's content tokens the selected unit's "aboutness"
    /// fields must cover for that unit to *veto* the out-of-scope scope-risk
    /// signal (the corroboration veto in Tier C.5). Calibrated on the
    /// 50-conversation holdout: out-of-scope hardware/field-service turns cover
    /// ≤ 0.25 of their tangential page, so this floor (0.5 — a majority of the
    /// query's content words) clears them with margin while never firing on an
    /// in-scope turn (no in-scope holdout turn triggers the out-of-scope
    /// lexicon). It is deliberately **not** a universal grounding floor: many
    /// genuine terse support turns cover < 0.5, so a blanket floor would
    /// over-decline. Coverage is consulted only in conjunction with the
    /// out-of-scope lexicon.
    public static let groundingCoverageFloor = 0.5

    public static func decide(
        signals: TelcoPolicySignals,
        candidates: [TelcoRetrievalCandidate],
        selectedUnit: RAGUnit?,
        toolRegistry: ToolRegistry,
        aliasMap: ToolAliasMap?
    ) -> TelcoPolicyResolution {
        // The dialogue-state operation (ADR-029 §7) is authoritative when the
        // caller resolved it upstream (the composer dispatcher does, so it can
        // serialize it). When absent (legacy callers, unit tests) the engine
        // derives it from the same inputs, so the route is identical either way.
        let resolution = signals.stateResolution
            ?? TelcoStateOperationResolver.resolve(
                query: signals.query,
                relation: signals.relation,
                prior: signals.prior,
                state: signals.state
            )
        return decideRoute(
            signals: signals,
            resolution: resolution,
            candidates: candidates,
            selectedUnit: selectedUnit,
            toolRegistry: toolRegistry,
            aliasMap: aliasMap
        ).stamping(resolution)
    }

    /// Map a resolved dialogue-state operation + evidence + capability to one
    /// `ComposerRoute`. The state operation owns the context/terminal decision;
    /// this function is a thin, auditable mapping plus the grounding rungs.
    private static func decideRoute(
        signals: TelcoPolicySignals,
        resolution: TelcoStateResolution,
        candidates: [TelcoRetrievalCandidate],
        selectedUnit: RAGUnit?,
        toolRegistry: ToolRegistry,
        aliasMap: ToolAliasMap?
    ) -> TelcoPolicyResolution {
        let understanding = signals.understanding
        let state = signals.state
        let prior = signals.prior
        // A groundable local answer exists. This is the corroboration gate for
        // the *soft* (trained-head) deflection signals below: those heads are
        // not yet calibrated tightly enough on this traffic to override a real
        // local answer on their own, so they may preempt the grounded answer
        // only when nothing local grounds the turn. High-precision signals
        // (the resolved state operation, PII) are not gated this way.
        let hasGroundableAnswer = selectedUnit != nil

        // ── Tier A1 · Sensitive-data safety (pre-operation gate) ───────────
        //
        // Payment/identity data must never enter the local RAG/tool path. This
        // is the one decision that precedes the state operation: it is a hard
        // safety gate on the *content*, independent of what the dialogue is
        // doing. Gated at the elevated PII threshold.
        if understanding?.isBlocked == true {
            return .terminal(.outOfScope, handoff: nil, reason: "pii_block")
        }

        // ── The dialogue-state operation owns the context/terminal decision ──
        //
        // Resolved once, upstream, from the relation label, the deterministic
        // prior, and the blackboard state. The historical defect (ADR-029 §7)
        // was re-deriving these inline across A/B/C tiers, which made the
        // ambiguous-turn decision depend on whether the retriever happened to
        // return a tangential page. The operation removes that coupling.
        switch resolution.operation {
        case .directEscalation:
            // Explicit human request (relation head or canonical lexicon) —
            // unambiguous; escalate regardless of local evidence.
            return .terminal(.liveAgent, handoff: "live_agent", reason: "explicit_handoff")

        case .unsupportedBackendOrOutOfScope:
            // A destination outside the local corpus. Account/billing/order →
            // a navigable account destination; an unsupported external action →
            // a flat decline. Distinguished by the same single-sourced prior
            // that resolved the operation.
            if prior.accountOrBilling {
                return .terminal(.accountNav, handoff: "cloud", reason: "account_lexical")
            }
            return .terminal(.outOfScope, handoff: nil, reason: "unsupported_external")

        case .repairFailed, .repairCannotFind:
            // Exhausted → escalate (re-guiding the same page again is worse than
            // escalating; budget caps at `repairEscalationThreshold`). In-progress
            // → reuse the active task's evidence; do NOT re-retrieve. No active
            // task → ask what failed.
            if state.repairAttemptsOnActiveTask >= repairEscalationThreshold {
                return .terminal(.liveAgent, handoff: "live_agent", reason: "repair_exhausted")
            }
            if state.hasActiveTask {
                return reuseEvidenceResolution(reason: "repair_in_progress")
            }
            return .terminal(.clarify, handoff: nil, reason: "repair_without_task")

        case .askClarification:
            // An under-specified turn with no valid state to continue. Asking is
            // cheap and correct; never ground a fresh page on two words. This is
            // authoritative and NOT gated on grounding — the historical defect
            // ground ambiguous turns whenever the retriever returned any
            // tangential page (ADR-029 §7).
            return .terminal(.clarify, handoff: nil, reason: "ask_clarification")

        case .reuseActiveEvidence:
            // Ambiguous continuation of an active task → reuse its evidence
            // rather than re-retrieving on an under-specified turn.
            if state.hasActiveTask {
                return reuseEvidenceResolution(reason: "reuse_active_evidence")
            }
            // Defensive: the resolver only emits this with an active task; fall
            // through to grounding if state drifted.

        case .carryoverActiveTask, .retrieveWithPriorBias, .retrieveFresh,
             .updateNewTask, .clearContextTopicSwitch, .clarificationAnswer,
             .confirmationYes, .confirmationNo:
            // Grounding operations — the retrieval strategy already shaped the
            // evidence upstream; fall through to the greeting / soft-head /
            // scope / grounded rungs below.
            break
        }

        // ── Greeting — exact phatic opener with nothing groundable ─────────
        if prior.greeting && !hasGroundableAnswer {
            return .terminal(.greeting, handoff: nil, reason: "greeting")
        }

        // ── Soft head deflections (corroboration-gated) ────────────────────
        //
        // Trained-head deflection signals may preempt only when no local answer
        // grounds the turn. When a real page is available, the grounded answer
        // wins and the head outcome stays trace-only (ADR-029 §2). Promote any
        // of these back to a hard preempt once the corresponding head is
        // recalibrated (ADR-029 §6 / ADR-028 retrain).
        if !hasGroundableAnswer, let understanding {
            if understanding.supportIntent.isConfident(.agentHandoff)
                || understanding.routingLane.isConfident(.humanEscalation)
                || understanding.issueComplexity.isConfident(.humanRequired)
                || understanding.escalationRisk.isConfident(.complaint)
                || understanding.escalationRisk.isConfident(.churnRisk)
                || understanding.escalationRisk.isConfident(.urgent) {
                return .terminal(.liveAgent, handoff: "live_agent", reason: "head_escalation")
            }
            if understanding.requiresCloudAssist {
                return .terminal(.accountNav, handoff: "cloud", reason: "head_cloud")
            }
            if understanding.needsClarification {
                return .terminal(.clarify, handoff: nil, reason: "needs_clarification")
            }
        }

        // ── Out-of-local-scope decline (scope-risk × weak grounding) ───────
        //
        // A peripheral-hardware or field-service turn (printer, camera,
        // technician/install, fiber-line burial) names something the
        // home-internet corpus structurally cannot serve. The retriever still
        // returns its best page, but that page is tangential. This two-factor
        // gate declines only when BOTH the out-of-scope lexicon fires AND no
        // unit actually grounds the turn (coverage below the floor) — so a
        // future in-scope page that truly covers the term would still answer,
        // and an in-scope turn that merely has thin coverage is untouched
        // (coverage alone never declines, ADR-029 §3). This stays a grounding-
        // dependent decision (not a hard state operation) by design, so the
        // safety valve survives.
        if prior.outOfLocalScope && !isStronglyGrounded(query: signals.query, unit: selectedUnit) {
            return .terminal(.outOfScope, handoff: nil, reason: "oos_no_grounding")
        }

        // ── Off-domain decline (telco_topic_scope head × weak grounding) ───
        //
        // High-precision off-domain gate (ADR-032 frozen-probe pilot). The lexicon
        // gate above only covers peripheral-hardware terms; this catches GENERAL
        // off-domain ("what is the weather in new york", jokes, capitals) that no
        // lexicon enumerates. Deliberately near-certain (τ=0.97): the off-domain /
        // terse-in-domain boundary is content-inseparable (BM25, coverage AND cosine
        // all overlap — contract §3c), so the gate fires ONLY when the head is
        // near-certain off-domain AND no unit strongly grounds the turn. This catches
        // egregious off-domain with zero register over-decline (the only gate-reaching
        // over-decliners sit below 0.97), at the cost of missing ultra-terse off-domain
        // ("what's the weather", 0.979 — inseparable from real terse support). Stays a
        // grounding-dependent decision (the veto is the safety valve), not a HARD route.
        if understanding?.topicScope?.isConfident(
            .outOfScope, minimum: TelcoPolicyThreshold.topicScopeOutOfScope
        ) == true,
            !isStronglyGrounded(query: signals.query, unit: selectedUnit) {
            return .terminal(.outOfScope, handoff: nil, reason: "topic_scope_off_domain")
        }

        // ── Grounded answer ────────────────────────────────────────────────
        //
        // Evidence present — the capability gate (ToolRegistry + ToolAliasMap +
        // affordance + mood + gated head evidence) decides
        // rag_answer / answer_plus_action / tool_action.
        if let unit = selectedUnit {
            return resolveGrounded(
                query: signals.query,
                unit: unit,
                understanding: understanding,
                state: state,
                toolRegistry: toolRegistry,
                aliasMap: aliasMap
            )
        }

        // No local answer and nothing above fired.
        return .terminal(.noRagAnswer, handoff: nil, reason: "no_evidence")
    }

    /// A repair / ambiguous continuation that reuses the active task's prior unit
    /// rather than re-retrieving (sets `reuseActiveEvidence`, which the dispatcher
    /// honors when resolving composer evidence).
    private static func reuseEvidenceResolution(reason: String) -> TelcoPolicyResolution {
        TelcoPolicyResolution(
            route: .ragAnswer,
            requiresConfirmation: false,
            executableToolIntent: nil,
            handoff: nil,
            reuseActiveEvidence: true,
            reason: reason
        )
    }

    /// Whether the selected unit lexically covers enough of the query's content
    /// tokens to count as a real local answer. This is the corroboration veto
    /// for the out-of-scope scope-risk signal (Tier C.5) — it is consulted
    /// *only* alongside the out-of-scope lexicon, never as a universal grounding
    /// floor (measured holdout distributions show many genuine terse support
    /// turns below the floor, ADR-029 §3).
    private static func isStronglyGrounded(query: String, unit: RAGUnit?) -> Bool {
        guard let unit else { return false }
        return unit.groundingCoverage(forQuery: query) >= groundingCoverageFloor
    }

    // MARK: - Grounded rung (capability gate)

    /// Capability-gated grounded resolution. Mirrors the Step-5b doctrine that
    /// `ToolRegistry` + `ToolAliasMap` are the *final arbiter* for whether an
    /// answer carries an executable action; mood and affordance never
    /// manufacture a confirmation handshake on their own.
    ///
    /// Shared-head outcomes (`local_answer` / `local_tool`) participate here as
    /// *gated evidence* (ADR-029 §2): a confident `local_answer` suppresses an
    /// action offer; a confident `local_tool` upgrades to a tool action when a
    /// real tool exists. Low-confidence head outcomes leave the deterministic
    /// mood/affordance decision untouched.
    private static func resolveGrounded(
        query: String,
        unit: RAGUnit,
        understanding: TelcoSharedUnderstanding?,
        state: TelcoDialogueStateSnapshot,
        toolRegistry: ToolRegistry,
        aliasMap: ToolAliasMap?
    ) -> TelcoPolicyResolution {
        // View/navigate pages never carry a tool offer even if their link_id
        // happens to alias to a registered tool.
        if let affordance = unit.actionAffordance,
           affordance == "view" || affordance == "navigate" {
            return .grounded(.ragAnswer)
        }

        // Resolve the registered tool behind this unit (alias-map first, then
        // the legacy direct link_id == tool_id path for unmapped contexts).
        var tool: Tool?
        var imperativeOnly = false
        if let aliasMap, let alias = aliasMap.alias(forLinkID: unit.linkID) {
            if let intent = ToolIntent(toolID: alias.toolID) {
                tool = toolRegistry.tool(for: intent)
                imperativeOnly = alias.imperativeOnly
            }
        } else if let intent = ToolIntent(toolID: unit.linkID) {
            tool = toolRegistry.tool(for: intent)
        }
        guard let resolvedTool = tool,
              let resolvedIntent = ToolIntent(toolID: resolvedTool.id) else {
            // No executable capability → grounded answer only.
            return .grounded(.ragAnswer)
        }

        let mood = inferQueryMood(query)

        // Gated head evidence (ADR-029 §2). A confident `local_answer` suppresses
        // the action offer. A confident `local_tool` forces the tool action, but
        // only when corroborated — either the turn's structural mood expresses
        // action intent, or there is an active tool flow on this task (a pending
        // confirmation already in play). An action lane on a fresh passive
        // statement ("connection issues") is a head false positive and must not
        // manufacture a side-effecting tool offer.
        if understanding?.routingLane.isConfident(.localAnswer) == true {
            return .grounded(.ragAnswer)
        }
        if understanding?.routingLane.isConfident(.localTool) == true,
           mood == .actionImperative || state.pendingToolID != nil {
            return .tool(.toolAction, intent: resolvedIntent)
        }

        // Imperative-only carve-out for shared link_ids (e.g. parental-controls
        // pages sharing `home`): informational queries stay informational.
        if imperativeOnly && mood != .actionImperative {
            return .grounded(.ragAnswer)
        }

        switch mood {
        case .actionImperative:
            return .tool(.toolAction, intent: resolvedIntent)
        case .question:
            guard unit.queryTargetsTaskObjective(query) else {
                return .grounded(.ragAnswer)
            }
            return .tool(.answerPlusAction, intent: resolvedIntent)
        case .navigateImperative, .statement:
            return .grounded(.ragAnswer)
        }
    }
}

// MARK: - Engine inputs

/// Compact, deterministic view of the dialogue blackboard the policy engine
/// needs. Built by `ChatViewModel` from `TelcoDialogueBlackboard` *after* the
/// turn's relation has been reduced, so the remediation counters already
/// reflect the current turn.
public struct TelcoDialogueStateSnapshot: Sendable, Equatable {
    /// True when a prior answer left an active task/page the next turn can
    /// continue or repair.
    public let hasActiveTask: Bool
    public let priorPageID: String?
    public let priorLinkID: String?
    public let pendingToolID: String?
    /// Remediation attempts already recorded against the active task
    /// (`repair_failed` / `repair_cannot_find`). Drives the escalation budget.
    public let repairAttemptsOnActiveTask: Int
    public let frustrationCount: Int
    public let hasPriorAssistantTurn: Bool
    /// True when the *previous* turn's route was `clarify`. The state-operation
    /// resolver reads this so a short reply to our own clarification question is
    /// treated as a `clarification_answer` (grounds) rather than a fresh
    /// `ambiguous_short_turn` (re-asks). Sourced from the blackboard's last
    /// recorded policy decision (ADR-029 §7).
    public let priorRouteWasClarify: Bool

    public init(
        hasActiveTask: Bool,
        priorPageID: String?,
        priorLinkID: String?,
        pendingToolID: String?,
        repairAttemptsOnActiveTask: Int,
        frustrationCount: Int,
        hasPriorAssistantTurn: Bool,
        priorRouteWasClarify: Bool = false
    ) {
        self.hasActiveTask = hasActiveTask
        self.priorPageID = priorPageID
        self.priorLinkID = priorLinkID
        self.pendingToolID = pendingToolID
        self.repairAttemptsOnActiveTask = repairAttemptsOnActiveTask
        self.frustrationCount = frustrationCount
        self.hasPriorAssistantTurn = hasPriorAssistantTurn
        self.priorRouteWasClarify = priorRouteWasClarify
    }

    public static let empty = TelcoDialogueStateSnapshot(
        hasActiveTask: false,
        priorPageID: nil,
        priorLinkID: nil,
        pendingToolID: nil,
        repairAttemptsOnActiveTask: 0,
        frustrationCount: 0,
        hasPriorAssistantTurn: false,
        priorRouteWasClarify: false
    )
}

/// Single-sourced deterministic priors over the raw user turn, consumed by the
/// engine as high-precision fallback evidence.
///
/// Two kinds of signal live here:
///
/// * `explicitHumanRequest` — an unambiguous "talk to a person" phrase. This is
///   the *highest-precision* escalation signal in the product, so the engine
///   honors it regardless of whether an upstream relation head ran or how it
///   classified the turn. It is sourced from the single canonical lexicon
///   (`ConversationStateRecorder.isLiveAgentRequest`), the same function the
///   blackboard's relation fallback uses to emit `escalation_request`; the two
///   entry points therefore never diverge.
/// * `accountOrBilling` / `unsupportedExternal` / `greeting` — topic signals
///   that no `TelcoTurnRelation` label carries. These are consulted *only* when
///   no confident shared-understanding head produced the corresponding signal,
///   so a false positive can affect only a turn the model layer declined to
///   classify.
public struct TelcoDeterministicPrior: Sendable, Equatable {
    /// An unambiguous request for a human/agent/representative.
    public let explicitHumanRequest: Bool
    /// Account, billing, plan, order, or line-management topic — needs live
    /// systems. Includes the backend/order surface (order status, plan signup,
    /// add-a-line) the local corpus structurally cannot serve.
    public let accountOrBilling: Bool
    /// An external action the assistant cannot perform (e.g. "email me …").
    public let unsupportedExternal: Bool
    /// A phatic greeting with no support content.
    public let greeting: Bool
    /// A peripheral-hardware or field-service topic outside the home-internet
    /// corpus (printer, camera, technician/install, fiber-line burial). This is
    /// a *risk* signal, not a route by itself: the engine declines only when it
    /// also fails to ground the turn (scope-risk × weak coverage, ADR-029 §3).
    public let outOfLocalScope: Bool

    public init(
        explicitHumanRequest: Bool,
        accountOrBilling: Bool,
        unsupportedExternal: Bool,
        greeting: Bool,
        outOfLocalScope: Bool
    ) {
        self.explicitHumanRequest = explicitHumanRequest
        self.accountOrBilling = accountOrBilling
        self.unsupportedExternal = unsupportedExternal
        self.greeting = greeting
        self.outOfLocalScope = outOfLocalScope
    }

    public static let none = TelcoDeterministicPrior(
        explicitHumanRequest: false,
        accountOrBilling: false,
        unsupportedExternal: false,
        greeting: false,
        outOfLocalScope: false
    )

    /// Derive the deterministic topic prior from the raw user turn.
    ///
    /// This is the *single* home for the closed topic lexicons the engine may
    /// fall back on when no shared-understanding head ran (the headless floor,
    /// and the ADR-028 degraded-mode contract). Each lexicon is intentionally
    /// narrow and documented; a confident head signal always takes precedence
    /// over the prior inside the engine, so a false positive here can only
    /// affect a turn that the model layer declined to classify.
    ///
    /// IS account/billing: bill, payment, autopay, late fee, plan, data cap,
    /// add a line/user. IS NOT: any router/Wi-Fi/device/parental-controls
    /// support task (those are grounded locally and must not be deflected).
    public static func derive(query: String) -> TelcoDeterministicPrior {
        let normalized = query
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return TelcoDeterministicPrior(
            explicitHumanRequest: ConversationStateRecorder.isLiveAgentRequest(query),
            accountOrBilling: matches(accountBillingPattern, normalized),
            unsupportedExternal: matches(unsupportedExternalPattern, normalized),
            greeting: greetings.contains(normalized),
            outOfLocalScope: matches(outOfScopePattern, normalized)
        )
    }

    private static func matches(_ regex: NSRegularExpression, _ text: String) -> Bool {
        regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }

    private static let greetings: Set<String> = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    ]

    // swiftlint:disable:next force_try
    private static let accountBillingPattern = try! NSRegularExpression(
        pattern: #"""
        (?xi)
        \b(
            bill(?:ing)?
          | payment
          | pay\s+(?:my\s+)?bill
          | auto\s*pay
          | late\s+fee
          | monthly\s+cost
          | my\s+plan
          | what\s+plan\s+am\s+i\s+on
          | add\s+(?:a\s+)?(?:user|line|phone\s+line)
          | add\s+(?:a|an|another)\s+(?:\w+\s+){0,2}(?:user|line|phone)
          | sub\s+account
          | data\s+cap
          | (?:150|300|600)\s*(?:gb|gigs?)
          # ── backend / order surface (no local page can serve these) ──
          | order\s+(?:status|number|is\s+awaiting|awaiting)
          | awaiting\s+completion
          | place\s+an\s+order
          | order\s+one
          | set\s+up\s+a\s+plan
          | sign\s+up\s+for\s+a\s+plan
          | cost\s+to\s+add
          | (?:get|order|buy|want)\s+(?:a\s+)?new\s+set\s*top\s*box
        )\b
        """#,
        options: [.allowCommentsAndWhitespace]
    )

    // swiftlint:disable:next force_try
    private static let unsupportedExternalPattern = try! NSRegularExpression(
        pattern: #"""
        (?xi)
        \b(
            send\s+(?:for\s+)?email
          | email\s+me
          | send\s+me\s+(?:an\s+)?email
        )\b
        """#,
        options: [.allowCommentsAndWhitespace]
    )

    /// Peripheral-hardware / field-service topics the home-internet support
    /// corpus structurally cannot serve. Intentionally narrow: each term names
    /// a physical device or out-of-band service (a printer, a camera, a
    /// dispatched technician, the buried fiber drop) with no canonical app page.
    /// Consumed only as a *risk* signal by the engine's out-of-scope gate, which
    /// declines a turn only when this fires AND the retriever fails to ground it
    /// (ADR-029 §3) — so a future in-scope page that truly covers one of these
    /// terms would still answer, and the lexicon can never decline on its own.
    // swiftlint:disable:next force_try
    private static let outOfScopePattern = try! NSRegularExpression(
        pattern: #"""
        (?xi)
        \b(
            printer
          | cameras?
          | technician
          | optic\s+fiber
          | fiber\s+line
          | bury
          | seal\s+the
        )\b
        """#,
        options: [.allowCommentsAndWhitespace]
    )
}

/// Bundle of typed signals handed to the engine for one turn.
public struct TelcoPolicySignals: Sendable {
    public let query: String
    public let relation: TelcoTurnRelation?
    public let understanding: TelcoSharedUnderstanding?
    public let state: TelcoDialogueStateSnapshot
    public let prior: TelcoDeterministicPrior
    /// The explicit dialogue-state operation for this turn (ADR-029 §7). When the
    /// caller resolves it upstream (the composer dispatcher does, so it can
    /// serialize it), the engine treats it as authoritative. When omitted (legacy
    /// callers, unit tests), the engine derives it from the relation/prior/state
    /// via `TelcoStateOperationResolver` so behavior is identical either way.
    public let stateResolution: TelcoStateResolution?

    public init(
        query: String,
        relation: TelcoTurnRelation?,
        understanding: TelcoSharedUnderstanding?,
        state: TelcoDialogueStateSnapshot,
        prior: TelcoDeterministicPrior,
        stateResolution: TelcoStateResolution? = nil
    ) {
        self.query = query
        self.relation = relation
        self.understanding = understanding
        self.state = state
        self.prior = prior
        self.stateResolution = stateResolution
    }
}

// MARK: - Engine output

/// The engine's full decision. `ComposerRoute` is the rendered route;
/// `reuseActiveEvidence` tells the dispatcher to render from the active task's
/// prior unit rather than the fresh top hit (repair continuity); `reason` is
/// an audit code recorded on the blackboard.
public struct TelcoPolicyResolution: Sendable, Equatable {
    public let route: ComposerRoute
    public let requiresConfirmation: Bool
    public let executableToolIntent: ToolIntent?
    public let handoff: String?
    public let reuseActiveEvidence: Bool
    public let reason: String
    /// The dialogue-state operation this route served (ADR-029 §7), echoed from
    /// the resolved `TelcoStateResolution` so the dispatcher can serialize it onto
    /// the harness report. Defaults are placeholders overwritten by `stamping`.
    public let stateOperation: TelcoStateOperation
    public let stateOperationReason: String

    public init(
        route: ComposerRoute,
        requiresConfirmation: Bool,
        executableToolIntent: ToolIntent?,
        handoff: String?,
        reuseActiveEvidence: Bool,
        reason: String,
        stateOperation: TelcoStateOperation = .updateNewTask,
        stateOperationReason: String = "unset"
    ) {
        self.route = route
        self.requiresConfirmation = requiresConfirmation
        self.executableToolIntent = executableToolIntent
        self.handoff = handoff
        self.reuseActiveEvidence = reuseActiveEvidence
        self.reason = reason
        self.stateOperation = stateOperation
        self.stateOperationReason = stateOperationReason
    }

    /// Return a copy carrying the resolved state operation. The route decision is
    /// made by `decideRoute`; the operation is stamped on once at the boundary so
    /// every route path records which dialogue-state decision produced it.
    func stamping(_ resolution: TelcoStateResolution) -> TelcoPolicyResolution {
        TelcoPolicyResolution(
            route: route,
            requiresConfirmation: requiresConfirmation,
            executableToolIntent: executableToolIntent,
            handoff: handoff,
            reuseActiveEvidence: reuseActiveEvidence,
            reason: reason,
            stateOperation: resolution.operation,
            stateOperationReason: resolution.reason
        )
    }

    /// A non-grounded terminal route (no evidence, no tool).
    static func terminal(_ route: ComposerRoute, handoff: String?, reason: String) -> TelcoPolicyResolution {
        TelcoPolicyResolution(
            route: route,
            requiresConfirmation: false,
            executableToolIntent: nil,
            handoff: handoff,
            reuseActiveEvidence: false,
            reason: reason
        )
    }

    /// A grounded answer with no executable action.
    static func grounded(_ route: ComposerRoute) -> TelcoPolicyResolution {
        TelcoPolicyResolution(
            route: route,
            requiresConfirmation: false,
            executableToolIntent: nil,
            handoff: nil,
            reuseActiveEvidence: false,
            reason: "grounded_answer"
        )
    }

    /// A grounded answer that carries a registered, executable tool. The
    /// confirmation requirement is taken from the tool's own safety flag.
    static func tool(_ route: ComposerRoute, intent: ToolIntent) -> TelcoPolicyResolution {
        TelcoPolicyResolution(
            route: route,
            requiresConfirmation: intent.requiresConfirmation,
            executableToolIntent: intent,
            handoff: nil,
            reuseActiveEvidence: false,
            reason: route == .toolAction ? "tool_action" : "answer_plus_action"
        )
    }
}

// MARK: - Relation classification helpers

extension Optional where Wrapped == TelcoTurnRelation {
    /// True when the turn is a remediation report (the prior instruction
    /// failed, or the customer cannot locate the UI element we named).
    var isRepair: Bool {
        self == .repairFailed || self == .repairCannotFind
    }
}
