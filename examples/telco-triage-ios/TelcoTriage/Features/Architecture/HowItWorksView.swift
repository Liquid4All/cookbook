import SwiftUI

/// Architecture tab — CTO-grade view of the on-device / cloud boundary,
/// the processing pipeline, technical claims, and the roadmap.
///
/// Executive summary + boardroom script used to live here; they belong on
/// the ROI tab for the CFO audience. Here we keep only the technical
/// story: what's running locally, when cloud gets involved, what the
/// memory/latency/privacy characteristics are, and where routing
/// decisions are made.
struct HowItWorksView: View {
    @Environment(\.brand) private var brand
    @EnvironmentObject private var appState: AppState

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    intro
                    onDeviceBoundary
                    pipelineDiagram
                    routingDecision
                    claims
                    phaseRoadmap
                }
                .padding(16)
            }
            .background(brand.surfaceBackground.ignoresSafeArea())
            .navigationTitle("Architecture")
        }
    }

    // MARK: - Intro

    private var intro: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Liquid Foundation Models, end-to-end")
                .font(brand.titleFont)
                .foregroundStyle(brand.textPrimary)
            Text("**LFM2.5-350M** (bundled) handles simple home internet questions, triage, and safe tool routing on-device. **LFM2.5-Audio-1.5B** and **LFM2.5-VL-450M** are optional specialist packs for voice and vision. Complex requests hand off to the operator's existing cloud AI stack with a scrubbed summary, reducing the need to build a separate backend voice platform.")
                .font(.callout)
                .foregroundStyle(brand.textSecondary)
        }
    }

    // MARK: - On-device boundary (new — CTO audience)

    /// The trust surface. Makes the edge-vs-cloud boundary explicit so a
    /// telco CTO can answer "what actually leaves the device?" in one
    /// scroll. Paired with the Privacy Shield flow at runtime, this is
    /// the answer to every compliance question.
    private var onDeviceBoundary: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("On-device boundary")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .textCase(.uppercase)

            HStack(alignment: .top, spacing: 10) {
                BoundaryColumn(
                    title: "Stays local",
                    tint: brand.success,
                    icon: "iphone",
                    rows: [
                        "Raw user text (all of it)",
                        "PII detection + redaction",
                        "Intent classification",
                        "Multi-head support triage",
                        "Argument extraction",
                        "Tool selection + execution",
                        "Knowledge-base retrieval",
                        "Customer context (profile, equipment, usage)",
                        "All first-turn responses",
                    ]
                )
                BoundaryColumn(
                    title: "Goes to cloud (opt-in only)",
                    tint: brand.info,
                    icon: "cloud",
                    rows: [
                        "Scrubbed problem statement",
                        "Whitelisted context keys",
                        "Attempted action log",
                        "Handoff summary for cloud AI or agent",
                    ]
                )
            }

            Text("Cloud escalation is reserved for complex requests. The customer sees the exact packet, can untick any context field, and must tap approve. No background telemetry, no silent upload.")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.surfaceElevated.opacity(0.6), in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
    }

    // MARK: - Pipeline diagram

    private var pipelineDiagram: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Processing pipeline")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .textCase(.uppercase)

            VStack(spacing: 10) {
                PipelineStage(icon: "person.fill", label: "User question", latency: nil)
                PipelineArrow()
                PipelineStage(icon: "shield.lefthalf.filled", label: "PII scan (local)", latency: "~5 ms")
                PipelineArrow()
                PipelineStage(icon: "point.3.connected.trianglepath.dotted", label: "Multi-head triage", latency: "~50 ms")
                PipelineArrow()
                PipelineStage(icon: "cpu", label: "Local Q&A / tool routing", latency: "~80-180 ms")
                PipelineArrow()
                PipelineStage(icon: "bubble.left.and.bubble.right.fill", label: "Grounded response + deep link", latency: nil)
                HStack {
                    Spacer()
                    Text("↓ if deeper reasoning needed")
                        .font(.caption)
                        .foregroundStyle(brand.textSecondary)
                    Spacer()
                }
                PipelineStage(icon: "lock.shield", label: "Privacy Shield approval", latency: "customer-gated")
                PipelineArrow()
                PipelineStage(icon: "cloud", label: "Operator cloud AI handoff (PII stripped)", latency: "existing stack")
            }
        }
        .padding(16)
        .background(brand.surfaceElevated.opacity(0.6), in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
    }

    // MARK: - Routing

    /// The confidence-driven routing decision. Spells out when the app
    /// falls back from on-device to cloud so the behavior is legible to
    /// operators who'll own the thresholds.
    private var routingDecision: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Routing & fallback")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .textCase(.uppercase)

            VStack(alignment: .leading, spacing: 8) {
                RoutingRule(
                    threshold: "ChatMode → kb_question",
                    outcome: "Generative retrieval over all 32 KB entries, answer with verbatim citation",
                    tint: brand.success
                )
                RoutingRule(
                    threshold: "ChatMode → tool_action",
                    outcome: "Tool selector picks from 8 tools, propose with pre-filled arguments",
                    tint: brand.primary
                )
                RoutingRule(
                    threshold: "ChatMode → personal_summary",
                    outcome: "Summarize CustomerContext (plan, equipment, devices) directly",
                    tint: brand.info
                )
                RoutingRule(
                    threshold: "ChatMode → out_of_scope",
                    outcome: "Graceful \"out of scope\" response, nothing leaves the device",
                    tint: brand.textSecondary
                )
            }
            Text("Routing combines two local LFM decisions: ADR-015 heads produce telco/cloud/privacy signals, and chat-mode-router-v2 decides question vs. action. Simple questions stay on-device, actions become auditable tool proposals, and complex requests prepare cloud handoff only after privacy review.")
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.surfaceElevated.opacity(0.6), in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
    }

    // MARK: - Claims grid

    private var claims: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Technical claims")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .textCase(.uppercase)
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                ClaimTile(title: "Bundled", value: "~220 MB", note: "LFM2.5-350M Q4_K_M")
                ClaimTile(title: "Voice pack", value: "~1.5 GB", note: "LFM2.5-Audio-1.5B")
                ClaimTile(title: "Vision pack", value: "~287 MB", note: "LFM2.5-VL-450M Q4_0")
                ClaimTile(title: "Peak RAM", value: "~320 MB", note: "during inference")
                ClaimTile(title: "Battery", value: "< 5% / day", note: "typical session load")
                ClaimTile(title: "Latency", value: "80-180 ms", note: "on-device Q&A")
                ClaimTile(title: "Offline", value: "Full", note: "works in airplane mode")
                ClaimTile(title: "Cloud layer", value: "Liquid LFM", note: "no third-party frontier")
            }
        }
    }

    // MARK: - Roadmap

    private var phaseRoadmap: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Roadmap")
                .font(.subheadline).fontWeight(.semibold)
                .foregroundStyle(brand.textPrimary)
            PhaseRow(
                status: .shipped,
                phase: "Phase 1",
                detail: "Grounded natural-language Q&A over the local KB using LFM2.5-350M. Offline, private, sub-200ms. Source citations and app handoff chips on every answer."
            )
            PhaseRow(
                status: .shipped,
                phase: "Phase 2",
                detail: "Real tool calls on-device — Restart Router, Run Speed Test, Check Connection, WPS Pairing, Diagnostics, Extender Reboot, Parental Controls, Technician Scheduling, Set Downtime. Destructive actions confirm; read-only run immediately. Side effects mutate customer state live."
            )
            PhaseRow(
                status: .shipped,
                phase: "Phase 3",
                detail: "ADR-015 telco heads for operator-cloud routing signals, with a customer-visible Privacy Shield showing the exact escalation packet. Structured handoff (PII-scrubbed, whitelisted context) replaces raw-string forwarding. Vision understanding via LFM2.5-VL-450M pack. Voice via LFM2.5-Audio-1.5B. ARPU-driving Next-Best-Action engine."
            )
            PhaseRow(
                status: .next,
                phase: "Phase 4 (next)",
                detail: "Streaming inference · TTS output · real-time usage telemetry from the home internet session · predictive maintenance alerts · multi-device family-plan awareness."
            )
            PhaseRow(
                status: .future,
                phase: "Future",
                detail: "Customer-specific classifier heads · personalized LoRA adapters per segment · edge-reasoning benchmarks published alongside operator data."
            )
        }
    }
}

// MARK: - Pipeline components

private struct PipelineStage: View {
    let icon: String
    let label: String
    let latency: String?
    @Environment(\.brand) private var brand

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .frame(width: 28)
                .foregroundStyle(brand.textPrimary)
            Text(label)
                .font(.callout)
                .foregroundStyle(brand.textPrimary)
            Spacer()
            if let latency {
                Text(latency)
                    .font(brand.monoFont)
                    .foregroundStyle(brand.textSecondary)
            }
        }
        .padding(10)
        .background(brand.surfaceBackground, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(brand.border))
    }
}

private struct PipelineArrow: View {
    @Environment(\.brand) private var brand
    var body: some View {
        Image(systemName: "arrow.down")
            .font(.caption2)
            .foregroundStyle(brand.textSecondary)
    }
}

// MARK: - Boundary card

private struct BoundaryColumn: View {
    let title: String
    let tint: Color
    let icon: String
    let rows: [String]
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .foregroundStyle(tint)
                Text(title)
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundStyle(tint)
                    .textCase(.uppercase)
            }
            VStack(alignment: .leading, spacing: 4) {
                ForEach(rows, id: \.self) { row in
                    HStack(alignment: .top, spacing: 6) {
                        Circle()
                            .fill(tint.opacity(0.6))
                            .frame(width: 4, height: 4)
                            .padding(.top, 6)
                        Text(row)
                            .font(.caption)
                            .foregroundStyle(brand.textPrimary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .padding(10)
        .background(tint.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Routing rule row

private struct RoutingRule: View {
    let threshold: String
    let outcome: String
    let tint: Color
    @Environment(\.brand) private var brand

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Circle()
                .fill(tint)
                .frame(width: 6, height: 6)
                .padding(.top, 6)
            VStack(alignment: .leading, spacing: 2) {
                Text(threshold)
                    .font(brand.monoFont)
                    .foregroundStyle(brand.textPrimary)
                Text(outcome)
                    .font(.caption)
                    .foregroundStyle(brand.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
    }
}

// MARK: - Claim tile

private struct ClaimTile: View {
    let title: String
    let value: String
    let note: String
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .textCase(.uppercase)
            Text(value)
                .font(.system(size: 18, weight: .bold, design: .rounded))
                .foregroundStyle(brand.textPrimary)
            Text(note)
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
    }
}

// MARK: - Phase row

private struct PhaseRow: View {
    enum Status {
        case shipped, next, future

        var label: String {
            switch self {
            case .shipped: return "SHIPPED"
            case .next: return "NEXT"
            case .future: return "FUTURE"
            }
        }
    }

    let status: Status
    let phase: String
    let detail: String
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(status.label)
                    .font(.caption2).fontWeight(.bold)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(statusTint.opacity(0.15), in: Capsule())
                    .foregroundStyle(statusTint)
                Text(phase)
                    .font(.caption).fontWeight(.bold)
                    .foregroundStyle(brand.textPrimary)
                Spacer()
            }
            Text(detail)
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(statusTint.opacity(0.3), lineWidth: 1)
        )
    }

    private var statusTint: Color {
        switch status {
        case .shipped: return brand.success
        case .next: return brand.primary
        case .future: return brand.textSecondary
        }
    }
}
