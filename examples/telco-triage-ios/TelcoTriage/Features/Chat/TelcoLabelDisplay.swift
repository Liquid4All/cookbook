import Foundation

/// Maps the classifier's machine labels (`device_setup`, `no_tool`,
/// `local_answer`, `contains_payment_identity_data`) to human-readable
/// strings for the engineering-mode pipeline card. Showing raw kebab/
/// snake labels in the UI bleeds the training schema into a surface
/// that demoers and reviewers read aloud.
///
/// The mapping is intentionally conservative — when the input doesn't
/// match a known token it falls back to capitalize-first-word-and-
/// replace-underscores so a new schema label still renders sensibly
/// without shipping a UI patch.
enum TelcoLabelDisplay {
    private static let overrides: [String: String] = [
        // support_intent
        "troubleshooting":   "Troubleshooting",
        "outage":            "Outage",
        "billing":           "Billing",
        "appointment":       "Appointment",
        "device_setup":      "Device setup",
        "plan_account":      "Plan & account",
        "equipment_return":  "Equipment return",
        "agent_handoff":     "Agent handoff",
        // routing_lane
        "local_answer":      "Local answer",
        "local_tool":        "Local tool",
        "cloud_assist":      "Cloud assist",
        "human_escalation":  "Human escalation",
        "blocked":           "Blocked",
        "degraded":          "Degraded",
        // required_tool
        "restart_gateway":   "Restart gateway",
        "run_diagnostics":   "Run diagnostics",
        "speed_test":        "Speed test",
        "schedule_technician": "Schedule technician",
        "no_tool":           "No tool",
        "cloud_only":        "Cloud-only action",
        // issue_complexity
        "simple":            "Simple",
        "guided":            "Guided",
        "multi_step":        "Multi-step",
        "backend_required":  "Backend required",
        "human_required":    "Human required",
        // pii_risk
        "safe":                              "Safe",
        "contains_account_data":             "Contains account data",
        "contains_contact_data":             "Contains contact data",
        "contains_payment_identity_data":    "Contains payment / ID data",
        // escalation_risk
        "low":          "Low",
        "frustrated":   "Frustrated",
        "complaint":    "Complaint",
        "urgent":       "Urgent",
        "churn_risk":   "Churn risk",
        // transcript_quality
        "clean":          "Clean",
        "noisy":          "Noisy",
        "partial":        "Partial",
        "asr_uncertain":  "ASR uncertain",
        // multi-label slot completeness flags
        "missing_device":              "Missing device",
        "missing_symptom":             "Missing symptom",
        "missing_duration":            "Missing duration",
        "missing_location":            "Missing location",
        "missing_account_auth":        "Missing account auth",
        "missing_contact_preference":  "Missing contact preference",
        // multi-label cloud requirements
        "live_network_status":  "Live network status",
        "account_state":        "Account state",
        "billing_record":       "Billing record",
        "appointment_system":   "Appointment system",
        "device_inventory":     "Device inventory",
        "plan_catalog":         "Plan catalog",
        "auth":                 "Auth",
        // sentinels
        "unavailable":          "Unavailable",
        "none":                 "None",
        "complete":             "Complete",
    ]

    /// Best-effort human label. Falls back to capitalizing the first
    /// word and replacing underscores when the token is unknown.
    static func text(_ raw: String) -> String {
        if let v = overrides[raw] { return v }
        let cleaned = raw.replacingOccurrences(of: "_", with: " ")
        guard let first = cleaned.first else { return raw }
        return first.uppercased() + cleaned.dropFirst()
    }

    /// Comma-joined display for multi-label outputs. Returns "None"
    /// for empty so the row never collapses to a blank value.
    static func list(_ raw: [String], emptyText: String = "None") -> String {
        if raw.isEmpty { return emptyText }
        return raw.map(text).joined(separator: ", ")
    }
}
