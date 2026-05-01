import Foundation

/// View-state payload for the PII-inspection sheet, driven from a
/// PII-badge tap on a past message. Read-only diff between the raw
/// user text and the redacted view — lets the customer see exactly
/// what the on-device PII scanner caught.
///
/// Cloud-escalation / approval flow is intentionally absent: this
/// demo keeps every query on-device, so there is nothing to approve
/// before "sending." The PII scanner is surfaced as a visible proof
/// of on-device privacy, not a gate to cloud egress.
struct PrivacyShieldState: Identifiable {
    let id = UUID()
    let original: String
    let sanitized: String
    let spans: [PIISpan]
}
