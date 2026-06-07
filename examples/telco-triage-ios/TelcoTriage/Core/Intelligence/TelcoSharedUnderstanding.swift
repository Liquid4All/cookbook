import Foundation

/// Conservative policy thresholds for turning classifier-head outputs
/// into user-visible control flow.
///
/// The heads are allowed to be uncertain; hard product decisions are
/// not. Low-confidence labels still render in engineering traces, but
/// they do not deflect a customer to cloud, agent, block, or clarify
/// unless the relevant head clears the policy floor.
public enum TelcoPolicyThreshold {
    public static let hardDecision: Double = 0.70
    public static let piiBlock: Double = 0.80
    /// Operating threshold for the `telco_topic_scope` off-domain gate. Set to a
    /// **high-precision 0.97** ("decline only on near-certain off-domain") from the
    /// deployment-register measurement: terse off-domain ("what's the weather", 0.979)
    /// overlaps genuine terse in-domain ("No parent control", 0.981) and can't be
    /// separated, but egregious off-domain ("weather in new york", 0.999; jokes;
    /// capitals) sits above every gate-reaching must-answer turn (the only over-
    /// decliners — gigs 0.955 / Fios-TV 0.916 — are below 0.97). Two-factor (also
    /// requires weak grounding); validated by a full-register zero-regression eval.
    /// Tradeoff: catches egregious off-domain, misses ultra-terse off-domain.
    public static let topicScopeOutOfScope: Double = 0.97
}

/// One shared LFM2.5-350M classifier pass for Telco Triage.
///
/// These labels are the runtime contract exported in
/// `telco_shared_clf_schema.json`. They are intentionally separate from
/// the older `QueryUnderstanding` / Stage A types because they answer a
/// different product question: "what support workflow should this turn
/// enter?" rather than "can the old RAG stack run?"
public struct TelcoSharedUnderstanding: Sendable, Equatable {
    public let supportIntent: TelcoHeadOutcome<TelcoSupportIntent>
    public let issueComplexity: TelcoHeadOutcome<TelcoIssueComplexity>
    public let routingLane: TelcoHeadOutcome<TelcoRoutingLane>
    public let cloudRequirements: TelcoMultiLabelOutcome<TelcoCloudRequirement>
    public let requiredTool: TelcoHeadOutcome<TelcoRequiredTool>
    public let escalationRisk: TelcoHeadOutcome<TelcoEscalationRisk>
    public let piiRisk: TelcoHeadOutcome<TelcoPIIRisk>
    public let transcriptQuality: TelcoHeadOutcome<TelcoTranscriptQuality>
    public let missingSlots: TelcoMultiLabelOutcome<TelcoMissingSlot>
    /// Off-domain scope signal (ADR-032 frozen-probe pilot). Optional: additive +
    /// rollback-safe (absent in headless / pre-topic-scope bundle → policy ignores it).
    public let topicScope: TelcoHeadOutcome<TelcoTopicScope>?
    public let forwardPassMs: Double
    public let headProjectionMs: Double

    public var totalMs: Double { forwardPassMs + headProjectionMs }

    public init(
        supportIntent: TelcoHeadOutcome<TelcoSupportIntent>,
        issueComplexity: TelcoHeadOutcome<TelcoIssueComplexity>,
        routingLane: TelcoHeadOutcome<TelcoRoutingLane>,
        cloudRequirements: TelcoMultiLabelOutcome<TelcoCloudRequirement>,
        requiredTool: TelcoHeadOutcome<TelcoRequiredTool>,
        escalationRisk: TelcoHeadOutcome<TelcoEscalationRisk>,
        piiRisk: TelcoHeadOutcome<TelcoPIIRisk>,
        transcriptQuality: TelcoHeadOutcome<TelcoTranscriptQuality>,
        missingSlots: TelcoMultiLabelOutcome<TelcoMissingSlot>,
        forwardPassMs: Double,
        headProjectionMs: Double,
        topicScope: TelcoHeadOutcome<TelcoTopicScope>? = nil
    ) {
        self.supportIntent = supportIntent
        self.issueComplexity = issueComplexity
        self.routingLane = routingLane
        self.cloudRequirements = cloudRequirements
        self.requiredTool = requiredTool
        self.escalationRisk = escalationRisk
        self.piiRisk = piiRisk
        self.transcriptQuality = transcriptQuality
        self.missingSlots = missingSlots
        self.forwardPassMs = forwardPassMs
        self.headProjectionMs = headProjectionMs
        self.topicScope = topicScope
    }

    /// Hard policy: the user either asked for a human, the issue is
    /// human-only, or the customer signal is high-risk enough that a
    /// demo assistant should offer handoff instead of pretending it can
    /// solve the case locally.
    public var requiresHumanHandoff: Bool {
        supportIntent.isConfident(.agentHandoff) ||
            issueComplexity.isConfident(.humanRequired) ||
            routingLane.isConfident(.humanEscalation) ||
            escalationRisk.isConfident(.urgent) ||
            escalationRisk.isConfident(.complaint) ||
            escalationRisk.isConfident(.churnRisk)
    }

    /// Hard policy: the requested answer depends on live systems or
    /// account state that is not present in the offline RAG corpus.
    public var requiresCloudAssist: Bool {
        routingLane.isConfident(.cloudAssist) ||
            issueComplexity.isConfident(.backendRequired) ||
            requiredTool.isConfident(.cloudOnly) ||
            cloudRequirements.hasConfidentActiveLabel()
    }

    /// Hard policy: the turn should not go through local RAG/composer.
    public var isBlocked: Bool {
        routingLane.isConfident(.blocked) ||
            piiRisk.isConfident(.containsPaymentIdentityData, minimum: TelcoPolicyThreshold.piiBlock)
    }

    /// Low-quality voice/partial transcript signal. This asks for
    /// clarification; it does not manufacture slot values.
    public var needsClarification: Bool {
        transcriptQuality.isConfident(.partial) ||
            transcriptQuality.isConfident(.asrUncertain)
    }
}

public struct TelcoHeadOutcome<Label: RawRepresentable & Sendable & Equatable>: Sendable, Equatable
where Label.RawValue == String {
    public let label: Label
    public let confidence: Double
    public let probabilities: [Double]
    public let labelIndex: Int

    public init(label: Label, confidence: Double, probabilities: [Double], labelIndex: Int) {
        self.label = label
        self.confidence = confidence
        self.probabilities = probabilities
        self.labelIndex = labelIndex
    }

    public func isConfident(
        _ expected: Label,
        minimum: Double = TelcoPolicyThreshold.hardDecision
    ) -> Bool {
        label == expected && confidence >= minimum
    }
}

public struct TelcoMultiLabelOutcome<Label: RawRepresentable & Sendable & Equatable>: Sendable, Equatable
where Label.RawValue == String {
    public let activeLabels: [Label]
    public let probabilities: [Double]

    public init(activeLabels: [Label], probabilities: [Double]) {
        self.activeLabels = activeLabels
        self.probabilities = probabilities
    }

    public var hasActiveLabels: Bool { !activeLabels.isEmpty }

    public func hasConfidentActiveLabel(
        minimum: Double = TelcoPolicyThreshold.hardDecision
    ) -> Bool {
        guard hasActiveLabels else { return false }
        return probabilities.contains { $0 >= minimum }
    }
}

public enum TelcoSupportIntent: String, Sendable, Equatable {
    case troubleshooting
    case outage
    case billing
    case appointment
    case deviceSetup = "device_setup"
    case planAccount = "plan_account"
    case equipmentReturn = "equipment_return"
    case agentHandoff = "agent_handoff"
}

public enum TelcoIssueComplexity: String, Sendable, Equatable {
    case simple
    case guided
    case multiStep = "multi_step"
    case backendRequired = "backend_required"
    case humanRequired = "human_required"
}

public enum TelcoRoutingLane: String, Sendable, Equatable {
    case localAnswer = "local_answer"
    case localTool = "local_tool"
    case cloudAssist = "cloud_assist"
    case humanEscalation = "human_escalation"
    case blocked
}

public enum TelcoCloudRequirement: String, CaseIterable, Sendable, Equatable {
    case liveNetworkStatus = "live_network_status"
    case accountState = "account_state"
    case billingRecord = "billing_record"
    case appointmentSystem = "appointment_system"
    case deviceInventory = "device_inventory"
    case planCatalog = "plan_catalog"
    case auth
}

public enum TelcoRequiredTool: String, Sendable, Equatable {
    case restartGateway = "restart_gateway"
    case runDiagnostics = "run_diagnostics"
    case speedTest = "speed_test"
    case scheduleTechnician = "schedule_technician"
    case noTool = "no_tool"
    case cloudOnly = "cloud_only"

    public var toolIntent: ToolIntent? {
        switch self {
        case .restartGateway: return .restartRouter
        case .runDiagnostics: return .runDiagnostics
        case .speedTest: return .runSpeedTest
        case .scheduleTechnician: return .scheduleTechnician
        case .noTool, .cloudOnly: return nil
        }
    }
}

public enum TelcoEscalationRisk: String, Sendable, Equatable {
    case low
    case frustrated
    case churnRisk = "churn_risk"
    case complaint
    case urgent
}

public enum TelcoPIIRisk: String, Sendable, Equatable {
    case safe
    case containsAccountData = "contains_account_data"
    case containsContactData = "contains_contact_data"
    case containsPaymentIdentityData = "contains_payment_identity_data"
}

public enum TelcoTranscriptQuality: String, Sendable, Equatable {
    case clean
    case noisy
    case partial
    case asrUncertain = "asr_uncertain"
}

public enum TelcoMissingSlot: String, CaseIterable, Sendable, Equatable {
    case missingDevice = "missing_device"
    case missingSymptom = "missing_symptom"
    case missingDuration = "missing_duration"
    case missingLocation = "missing_location"
    case missingAccountAuth = "missing_account_auth"
    case missingContactPreference = "missing_contact_preference"
}

/// Off-domain scope: is the turn about home-internet at all? (`telco_topic_scope`,
/// ADR-032 frozen-probe pilot.) `outOfScope` drives the two-factor decline; `greeting`
/// stays owned by the deterministic prior; `inScope` covers all home-internet asks.
public enum TelcoTopicScope: String, Sendable, Equatable {
    case inScope = "in_scope"
    case outOfScope = "out_of_scope"
    case greeting
}
