import Foundation

/// Five-lane routing decision per ADR-015 Phase 2.
///
/// The TelcoMultiHeadClassifier emits a TelcoDecisionVector with 9
/// sequence heads. `TelcoLaneRouter.route(_:)` is a pure function that
/// derives the lane from those signals + deterministic policy.
///
/// The model emits signals; the router owns the decision.
public enum TelcoLaneDecision: Sendable, Equatable {
    /// Answerable from on-device KB without backend state.
    case localAnswer(supportIntent: TelcoSupportIntent, reason: String)
    /// Run an on-device tool (restart gateway, speed test, diagnostics).
    case localTool(tool: TelcoTool, supportIntent: TelcoSupportIntent, reason: String)
    /// Hand off to cloud assist with a structured payload.
    case cloudAssist(
        supportIntent: TelcoSupportIntent,
        requirements: [TelcoCloudRequirement],
        missingSlots: [TelcoSlotKind],
        piiRisk: TelcoPIIRisk,
        reason: String
    )
    /// Connect the user to a human agent.
    case humanEscalation(supportIntent: TelcoSupportIntent, reason: String)
    /// Refuse locally — safety / policy / unsupported.
    case blocked(reason: TelcoBlockReason)
    /// Classifier wasn't fully loaded — degrade gracefully.
    case degraded(reason: String)
}

public enum TelcoSupportIntent: String, Sendable, Equatable, CaseIterable {
    case troubleshooting
    case outage
    case billing
    case appointment
    case deviceSetup = "device_setup"
    case planAccount = "plan_account"
    case equipmentReturn = "equipment_return"
    case agentHandoff = "agent_handoff"
    case unknown
}

public enum TelcoTool: String, Sendable, Equatable, CaseIterable {
    case restartGateway = "restart_gateway"
    case runDiagnostics = "run_diagnostics"
    case speedTest = "speed_test"
    case scheduleTechnician = "schedule_technician"
    case noTool = "no_tool"
    case cloudOnly = "cloud_only"
}

public enum TelcoCloudRequirement: String, Sendable, Equatable, CaseIterable {
    case liveNetworkStatus = "live_network_status"
    case accountState = "account_state"
    case billingRecord = "billing_record"
    case appointmentSystem = "appointment_system"
    case deviceInventory = "device_inventory"
    case planCatalog = "plan_catalog"
    case auth
}

public enum TelcoSlotKind: String, Sendable, Equatable, CaseIterable {
    case missingDevice = "missing_device"
    case missingSymptom = "missing_symptom"
    case missingDuration = "missing_duration"
    case missingLocation = "missing_location"
    case missingAccountAuth = "missing_account_auth"
    case missingContactPreference = "missing_contact_preference"
}

public enum TelcoPIIRisk: String, Sendable, Equatable, CaseIterable {
    case safe
    case containsAccountData = "contains_account_data"
    case containsContactData = "contains_contact_data"
    case containsPaymentIdentityData = "contains_payment_identity_data"
}

public enum TelcoBlockReason: Sendable, Equatable {
    /// Query asks for help with self-harm or threats.
    case safetySensitive
    /// Query attempts to bypass auth (e.g. someone else's account).
    case authBypass
    /// Out of scope (illegal, not telco support).
    case outOfScope
    /// Confidence below the floor across multiple safety heads.
    case lowConfidence
}

/// Pure function: TelcoDecisionVector + policy → TelcoLaneDecision.
public enum TelcoLaneRouter {
    /// Confidence below this on `routing_lane` => degrade to deterministic
    /// inference from `support_intent` + `cloud_requirements`.
    public static let laneConfidenceFloor: Double = 0.45

    /// `routing_lane=blocked` has ~60 synthetic examples. The head can
    /// learn the pattern but deterministic policy remains the primary gate:
    /// PII-risk + escalation-risk + intent pattern catches the cases the
    /// model misses.
    public static func route(_ vector: TelcoDecisionVector) -> TelcoLaneDecision {
        guard vector.hasADR015Heads else {
            return .degraded(reason: "ADR-015 heads not loaded")
        }

        let supportIntent = TelcoSupportIntent(rawValue: vector.supportIntent.label) ?? .unknown
        let piiRisk = TelcoPIIRisk(rawValue: vector.piiRisk.label) ?? .safe
        let escalation = vector.customerEscalationRisk.label

        // Policy block #1: explicit safety-sensitive content. This stays
        // deterministic because safety-sensitive phrases are a small, closed
        // class and the fallback should be conservative.
        if Self.violatesSafetyPolicy(vector: vector) {
            return .blocked(reason: .safetySensitive)
        }

        // Policy block #2: human escalation when the customer explicitly
        // demands a human OR escalation risk is "complaint" / "urgent" with
        // a non-trivial intent. The classifier head will pick this up most
        // of the time but we treat the support_intent + escalation_risk
        // signal as the source of truth.
        if supportIntent == .agentHandoff || escalation == "complaint" || escalation == "churn_risk" {
            return .humanEscalation(
                supportIntent: supportIntent,
                reason: "agent_handoff intent or complaint/churn-risk signal"
            )
        }

        // Policy boundary: if the shared heads say this needs live
        // carrier/backend state, keep it on the cloud-assist lane even
        // if another local chat-mode head later sees "my area" or
        // "my service" and mistakes it for a profile summary. The
        // model emits signals; this router owns the egress decision.
        if Self.requiresCloudAssist(vector: vector, supportIntent: supportIntent) {
            return cloudAssistDecision(
                vector: vector,
                supportIntent: supportIntent,
                piiRisk: piiRisk
            )
        }

        // Below confidence floor on the lane head: derive lane from
        // cloud_requirements + intent.
        let laneLabel: String
        if vector.routingLane.confidence >= laneConfidenceFloor {
            laneLabel = vector.routingLane.label
        } else {
            laneLabel = Self.deriveLaneFromSignals(vector: vector)
        }

        switch laneLabel {
        case "local_answer":
            return .localAnswer(
                supportIntent: supportIntent,
                reason: "answerable from on-device KB"
            )
        case "local_tool":
            let tool = TelcoTool(rawValue: vector.requiredTool.label) ?? .noTool
            return .localTool(
                tool: tool,
                supportIntent: supportIntent,
                reason: "on-device action: \(tool.rawValue)"
            )
        case "cloud_assist":
            return cloudAssistDecision(
                vector: vector,
                supportIntent: supportIntent,
                piiRisk: piiRisk
            )
        case "human_escalation":
            return .humanEscalation(
                supportIntent: supportIntent,
                reason: "lane head signaled human handoff"
            )
        case "blocked":
            return .blocked(reason: .lowConfidence)
        default:
            return .degraded(reason: "unknown lane label: \(laneLabel)")
        }
    }

    /// Derive a lane when the lane head's confidence is below the floor.
    /// Prefers cloud_assist when the cloud_requirements multi-label is
    /// non-empty — that's the strongest signal that backend systems are
    /// needed.
    private static func deriveLaneFromSignals(vector: TelcoDecisionVector) -> String {
        if !vector.cloudRequirements.activeLabels.isEmpty {
            return "cloud_assist"
        }
        let intent = TelcoSupportIntent(rawValue: vector.supportIntent.label)
        switch intent {
        case .troubleshooting:
            // Tool-bearing heads decide between local_tool and local_answer.
            let tool = TelcoTool(rawValue: vector.requiredTool.label) ?? .noTool
            return tool == .noTool || tool == .cloudOnly ? "local_answer" : "local_tool"
        case .deviceSetup, .equipmentReturn:
            return "local_answer"
        case .agentHandoff:
            return "human_escalation"
        case .billing, .outage, .appointment, .planAccount:
            return "cloud_assist"
        case .unknown, .none:
            return "local_answer"
        }
    }

    /// Lightweight deterministic safety guard. The training corpus includes
    /// ~60 blocked examples, enough for the model to learn the pattern,
    /// but the deterministic guard stays as a safety net. False positives
    /// are acceptable — we prefer erring toward a refusal message over
    /// silently routing harmful intent.
    private static func violatesSafetyPolicy(vector: TelcoDecisionVector) -> Bool {
        // Only fire on extreme-confidence head signals. We don't have raw
        // text in the vector (by design), so the policy is signal-based:
        // urgent + agent_handoff + low support intent confidence is a
        // crisis-call pattern. Real production replaces this with a
        // text-level classifier upstream.
        let urgent = vector.customerEscalationRisk.label == "urgent"
        let lowIntentConfidence = vector.supportIntent.confidence < 0.4
        let agentHandoff = vector.supportIntent.label == "agent_handoff"
        return urgent && agentHandoff && lowIntentConfidence
    }

    private static func requiresCloudAssist(
        vector: TelcoDecisionVector,
        supportIntent: TelcoSupportIntent
    ) -> Bool {
        if !vector.cloudRequirements.activeLabels.isEmpty {
            return true
        }
        let requiredTool = TelcoTool(rawValue: vector.requiredTool.label) ?? .noTool
        if requiredTool == .cloudOnly {
            return true
        }
        guard vector.issueComplexity.label == "backend_required" else {
            return false
        }
        switch supportIntent {
        case .outage, .billing, .appointment, .planAccount:
            return true
        case .troubleshooting, .deviceSetup, .equipmentReturn, .agentHandoff, .unknown:
            return false
        }
    }

    private static func cloudAssistDecision(
        vector: TelcoDecisionVector,
        supportIntent: TelcoSupportIntent,
        piiRisk: TelcoPIIRisk
    ) -> TelcoLaneDecision {
        let requirements = cloudRequirements(from: vector, supportIntent: supportIntent)
        let missingSlots = vector.slotCompleteness.activeLabels.compactMap {
            TelcoSlotKind(rawValue: $0)
        }
        return .cloudAssist(
            supportIntent: supportIntent,
            requirements: requirements,
            missingSlots: missingSlots,
            piiRisk: piiRisk,
            reason: cloudReason(intent: supportIntent, requirements: requirements)
        )
    }

    private static func cloudRequirements(
        from vector: TelcoDecisionVector,
        supportIntent: TelcoSupportIntent
    ) -> [TelcoCloudRequirement] {
        let explicit = vector.cloudRequirements.activeLabels.compactMap {
            TelcoCloudRequirement(rawValue: $0)
        }
        if !explicit.isEmpty {
            return explicit
        }
        switch supportIntent {
        case .outage:
            return [.liveNetworkStatus]
        case .billing:
            return [.billingRecord, .accountState]
        case .appointment:
            return [.appointmentSystem]
        case .planAccount:
            return [.accountState, .planCatalog]
        case .troubleshooting, .deviceSetup, .equipmentReturn, .agentHandoff, .unknown:
            return []
        }
    }

    private static func cloudReason(
        intent: TelcoSupportIntent,
        requirements: [TelcoCloudRequirement]
    ) -> String {
        if requirements.isEmpty {
            return "intent \(intent.rawValue) typically requires cloud orchestration"
        }
        let names = requirements.map { $0.rawValue }.joined(separator: ", ")
        return "needs: \(names)"
    }
}
