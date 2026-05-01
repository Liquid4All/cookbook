import SwiftUI

/// Shared formatters reused across `PlanView` renders. `RelativeDateTime`
/// allocation isn't free and SwiftUI can call `body` dozens of times per
/// second during animations. Internal rather than private so the
/// network-section sub-views (extracted to PlanView+NetworkSections.swift
/// for 600-line-file budget) can share the same instance.
enum PlanFormatters {
    static let relative: RelativeDateTimeFormatter = {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .full
        return f
    }()
}

struct PlanView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.brand) private var brand

    var body: some View {
        let context = appState.customerContext
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Lead with what AI is doing here — the "why this
                    // tab exists" card. Without this, an exec scrolls
                    // through household data with no idea why AI matters.
                    AIContextHeader()

                    HeroCard(profile: context.profile)

                    NarrativeBlock(
                        eyebrow: "Your home right now",
                        summary: "The on-device model reads connection state, device health, and usage signals to pre-compute which tools and actions are relevant before the customer even asks."
                    ) {
                        HomeOperationsSection(context: context)
                    }

                    NarrativeBlock(
                        eyebrow: "AI-scored recommendations",
                        summary: "LFM2.5-350M scores upsell, retention, and proactive-support signals entirely on-device. No usage data leaves the phone. These recommendations update live as the customer interacts."
                    ) {
                        ForYouSection(engine: appState.nbaEngine)
                    }

                    NarrativeBlock(
                        eyebrow: "On-device action log",
                        summary: "Every action the AI assistant took during this session — parental controls, equipment restarts, technician dispatch — logged locally."
                    ) {
                        AssistantActivityBlock(context: context)
                    }

                    NarrativeBlock(
                        eyebrow: "Your fiber service",
                        summary: "Plan and usage data the AI assistant uses to score plan-fit recommendations and detect anomalies."
                    ) {
                        VStack(spacing: 12) {
                            PlanCard(plan: context.profile.plan)
                            UsageSummaryCard(usage: context.profile.usage, plan: context.profile.plan)
                        }
                    }

                    NarrativeBlock(
                        eyebrow: "Your home network",
                        summary: "Equipment the AI can diagnose and restart via natural language, plus family devices it can pause or resume through parental controls."
                    ) {
                        VStack(spacing: 12) {
                            EquipmentSection(equipment: context.profile.equipment)
                            ManagedDevicesSection(context: context)
                        }
                    }
                }
                .padding(16)
            }
            .background(brand.surfaceBackground.ignoresSafeArea())
            .navigationTitle("Household")
        }
    }
}

/// Top-of-tab card that explains what this tab is for. Answers the
/// exec question: "Why am I looking at a customer profile?"
///
/// The insight: this is the context the on-device model reads to make
/// support conversations zero-effort. "Restart my router" works without
/// follow-up questions because the model already knows which router,
/// its health status, and that it hasn't been rebooted in 14 days.
/// Buttons can't do this — they'd need a 5-screen wizard to collect
/// the same context that the model reads silently.
private struct AIContextHeader: View {
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Why this matters")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .textCase(.uppercase)
            Text("This is what the assistant already knows")
                .font(.title3)
                .fontWeight(.bold)
                .foregroundStyle(brand.textPrimary)
            Text("When a customer says \"restart my router,\" the on-device model reads this household context to fill in the blanks: which router, which network, what's the current health, who else is connected. No follow-up questions. No 5-screen wizard. One sentence in, one tap to confirm.")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: brand.cardCornerRadius)
                .stroke(brand.primary.opacity(0.15), lineWidth: 1)
        )
    }
}

/// Section wrapper that introduces each narrative beat with a short
/// eyebrow + body sentence above the content. Keeps the Plan tab
/// readable as a story ("here's now, here's what's next, here's what
/// happened") instead of a vertical stack of cards with no through-line.
private struct NarrativeBlock<Content: View>: View {
    let eyebrow: String
    let summary: String  // Named to avoid colliding with SwiftUI `body`.
    @ViewBuilder let content: () -> Content
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text(eyebrow)
                    .font(.caption).fontWeight(.bold)
                    .foregroundStyle(brand.accent)
                    .textCase(.uppercase)
                Text(summary)
                    .font(.caption)
                    .foregroundStyle(brand.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            content()
        }
    }
}

/// Merges the existing Service-Visit + Recent-Issues surfaces into a
/// single chronological feed. Previously these were two adjacent
/// sections making the same narrative point. One block, one story.
private struct AssistantActivityBlock: View {
    @ObservedObject var context: CustomerContext
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(spacing: 12) {
            if context.serviceAppointment != nil {
                ServiceVisitSection(context: context)
            }
            if !context.profile.recentIssues.isEmpty {
                RecentIssuesSection(issues: context.profile.recentIssues)
            }
            if context.serviceAppointment == nil && context.profile.recentIssues.isEmpty {
                Text("No on-device actions yet. Ask the assistant to restart something in Chat and it'll log here.")
                    .font(.caption)
                    .foregroundStyle(brand.textSecondary)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(brand.surfaceElevated.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }
}

private struct HomeOperationsSection: View {
    @ObservedObject var context: CustomerContext
    @Environment(\.brand) private var brand

    private var healthLabel: String {
        if context.profile.equipment.contains(where: { $0.kind == .extender && $0.status != .online }) {
            return "Extender path needs attention"
        }
        return "Home network stable"
    }

    private var subheadline: String {
        let pausedCount = context.managedDevices.filter { $0.accessState == .paused }.count
        let visitText = context.serviceAppointment == nil ? "No truck roll booked" : "Technician booked"
        return "\(pausedCount) family-device policies active · \(visitText)"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Home Operations")
                        .font(.caption)
                        .foregroundStyle(brand.textSecondary)
                        .textCase(.uppercase)
                    Text(healthLabel)
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundStyle(brand.textPrimary)
                    Text(subheadline)
                        .font(.caption)
                        .foregroundStyle(brand.textSecondary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Text("On-device actions")
                        .font(.caption2)
                        .foregroundStyle(brand.textSecondary)
                    Text("\(context.managedDevices.count + (context.serviceAppointment == nil ? 0 : 1)) surfaces live")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(brand.textPrimary)
                }
            }

            HStack(spacing: 12) {
                OpsStat(
                    label: "Family devices",
                    value: "\(context.managedDevices.count)",
                    note: "voice + chat controllable"
                )
                OpsStat(
                    label: "Paused now",
                    value: "\(context.managedDevices.filter { $0.accessState == .paused }.count)",
                    note: "parental-control actions"
                )
                OpsStat(
                    label: "Dispatch",
                    value: context.serviceAppointment == nil ? "none" : "scheduled",
                    note: "truck roll workflow"
                )
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.primary.opacity(0.08), in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
    }
}

private struct OpsStat: View {
    let label: String
    let value: String
    let note: String
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
            Text(value)
                .font(.subheadline)
                .fontWeight(.bold)
                .foregroundStyle(brand.textPrimary)
            Text(note)
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Next-Best-Action region at the top of the Plan tab. Shows up to three
/// personalization cards — the "For You" pattern every carrier is racing
/// to put in front of customers in 2026.
private struct ForYouSection: View {
    @ObservedObject var engine: NextBestActionEngine
    @Environment(\.brand) private var brand

    var body: some View {
        let visible = engine.topActions
            .filter { !engine.hasOutcome(for: $0.id) }
            .prefix(3)

        if visible.isEmpty {
            EmptyView()
        } else {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 6) {
                    Image(systemName: "drop.fill")
                        .font(.caption2)
                        .foregroundStyle(brand.accent)
                    Text("For You")
                        .font(.caption)
                        .foregroundStyle(brand.textPrimary)
                        .textCase(.uppercase)
                        .fontWeight(.semibold)
                    Text("· Scored by LFM2.5-350M on-device")
                        .font(.caption2)
                        .foregroundStyle(brand.textSecondary)
                    Spacer()
                }
                Text("Your assistant watches for upsell, retention, and proactive-support moments that fit your usage — and only surfaces the ones worth your attention. Nothing is sent to a cloud to pick these.")
                    .font(.caption)
                    .foregroundStyle(brand.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                ForEach(Array(visible.enumerated()), id: \.element.id) { _, action in
                    NextBestActionCard(
                        action: action,
                        onAccept: {
                            engine.record(outcome: NBAOutcome(actionID: action.id, verdict: .accepted))
                        },
                        onDecline: {
                            engine.record(outcome: NBAOutcome(actionID: action.id, verdict: .declined))
                        }
                    )
                }
            }
        }
    }
}

private struct UsageSummaryCard: View {
    let usage: CustomerProfile.UsageSnapshot
    let plan: CustomerProfile.Plan
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Last \(usage.periodDays) days")
                .font(.caption).foregroundStyle(brand.textSecondary).textCase(.uppercase)
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                UsageTile(label: "Downloaded", value: "\(usage.downloadedGB) GB")
                UsageTile(label: "Uploaded", value: "\(usage.uploadedGB) GB")
                UsageTile(label: "Avg down", value: "\(usage.avgDownMbps) Mbps")
                UsageTile(label: "Devices (peak)", value: "\(usage.peakDeviceCount)")
                UsageTile(label: "Speed test fails", value: "\(usage.speedTestFailures)")
                UsageTile(label: "Outage minutes", value: "\(usage.outageMinutes)")
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
    }
}

private struct UsageTile: View {
    let label: String
    let value: String
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(brand.textSecondary)
            Text(value)
                .font(.subheadline).fontWeight(.semibold)
                .monospacedDigit()
                .foregroundStyle(brand.textPrimary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct HeroCard: View {
    let profile: CustomerProfile
    @Environment(\.brand) private var brand

    var body: some View {
        HStack(spacing: 14) {
            Text(String(profile.firstName.prefix(1)) + String(profile.lastName.prefix(1)))
                .font(.title).fontWeight(.bold)
                .frame(width: 56, height: 56)
                .background(brand.primary, in: Circle())
                .foregroundStyle(brand.onPrimary)
            VStack(alignment: .leading, spacing: 2) {
                Text("\(profile.firstName) \(profile.lastName)")
                    .font(.headline)
                Text(profile.address.line1)
                    .font(.caption).foregroundStyle(brand.textSecondary)
                Text("\(profile.address.city), \(profile.address.state) \(profile.address.zip)")
                    .font(.caption).foregroundStyle(brand.textSecondary)
            }
            Spacer()
        }
    }
}

private struct PlanCard: View {
    let plan: CustomerProfile.Plan
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Current Plan")
                    .font(.caption).foregroundStyle(brand.textSecondary).textCase(.uppercase)
                Spacer()
                Text(String(format: "$%.2f/mo", plan.monthlyPrice))
                    .font(.caption).fontWeight(.semibold)
                    .foregroundStyle(brand.textPrimary)
            }
            Text(plan.name)
                .font(.title3).fontWeight(.bold)
            HStack(spacing: 20) {
                SpeedTile(label: "Down", value: "\(plan.downSpeedMbps) Mbps")
                SpeedTile(label: "Up", value: "\(plan.upSpeedMbps) Mbps")
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
    }
}

private struct SpeedTile: View {
    let label: String
    let value: String
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption).foregroundStyle(brand.textSecondary)
            Text(value).font(.subheadline).fontWeight(.semibold)
                .monospacedDigit()
        }
    }
}

// Equipment / Family devices / Service visit / Recent issues
// sub-sections live in PlanView+NetworkSections.swift so this file
// stays under the 600-line CLAUDE.md guideline.
