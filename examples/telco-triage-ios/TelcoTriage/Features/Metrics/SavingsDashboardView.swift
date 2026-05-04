import SwiftUI

/// Shared money formatters. SwiftUI re-evaluates card bodies on every
/// published change — allocating a fresh NumberFormatter inside `body`
/// spikes CPU during scroll.
enum DashboardFormatters {
    static let whole: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .currency
        f.currencyCode = "USD"
        f.maximumFractionDigits = 0
        return f
    }()

    static let precise: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .currency
        f.currencyCode = "USD"
        f.maximumFractionDigits = 4
        return f
    }()

    /// Auto-pick: under $1 → 4 fractional digits, over → whole dollars.
    static func auto(_ value: Double) -> String {
        let formatter = value.magnitude < 1.0 ? precise : whole
        return formatter.string(from: NSNumber(value: value)) ?? "$0"
    }
}

struct SavingsDashboardView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.brand) private var brand

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    ExecutiveSummaryCard(toolCount: appState.toolRegistry.all.count)
                    ExecutiveBusinessCaseCard()
                    HeaderTile(ledger: appState.tokenLedger)
                    OperatorImpactCard(ledger: appState.tokenLedger, stats: appState.sessionStats)
                    ARPUSignalCard(engine: appState.nbaEngine)
                    SessionTiles(ledger: appState.tokenLedger, stats: appState.sessionStats)
                    BoardroomScriptCard()
                    PricingFootnote()
                }
                .padding(16)
            }
            .background(brand.surfaceBackground.ignoresSafeArea())
            .navigationTitle("ROI")
        }
    }
}

/// Opening CFO-facing card. Frames the rest of the ROI tab as the
/// business case, not just a metrics dashboard. The four proof stats
/// are the "one-line pitch" the exec carries out of the demo.
private struct ExecutiveSummaryCard: View {
    let toolCount: Int
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Executive summary")
                    .font(.caption)
                    .foregroundStyle(brand.textSecondary)
                    .textCase(.uppercase)
                Text("Turn tier-1 support into a button press")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundStyle(brand.textPrimary)
                Text("Liquid edge models contain support on the phone, fill structured tool arguments from plain English, and only escalate to cloud after a scrubbed customer-approved preview.")
                    .font(.caption)
                    .foregroundStyle(brand.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 10) {
                ROIProofStat(value: "<200ms", note: "local response")
                ROIProofStat(value: "\(toolCount)", note: "telco actions")
                ROIProofStat(value: "0 PII", note: "leaves by default")
                ROIProofStat(value: "1 tap", note: "confirm + execute")
            }

            HStack(spacing: 8) {
                Label("Airplane-mode ready", systemImage: "airplane")
                    .font(.caption2)
                    .fontWeight(.semibold)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(brand.surfaceElevated, in: Capsule())
                Spacer()
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.primary.opacity(0.08), in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: brand.cardCornerRadius)
                .stroke(brand.primary.opacity(0.18), lineWidth: 1)
        )
    }
}

private struct ROIProofStat: View {
    let value: String
    let note: String
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(.headline)
                .fontWeight(.bold)
                .foregroundStyle(brand.textPrimary)
            Text(note)
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: 12))
    }
}

/// Each scene is one exec takeaway. Keeping this on the ROI tab instead
/// of the customer surface prevents slideware from bleeding into the
/// chat UX while making the demo script legible to the sales engineer.
private struct BoardroomScriptCard: View {
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Boardroom script")
                    .font(.caption)
                    .foregroundStyle(brand.textSecondary)
                    .textCase(.uppercase)
                Text("Each scene demonstrates a business outcome, not just a chatbot reply")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(brand.textPrimary)
            }
            ForEach(ConversationStarter.all) { starter in
                HStack(spacing: 12) {
                    Image(systemName: starter.icon)
                        .font(.system(size: 16))
                        .frame(width: 32, height: 32)
                        .background(brand.primary.opacity(0.12), in: Circle())
                        .foregroundStyle(brand.primary)
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            Text(starter.label)
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundStyle(brand.textPrimary)
                            Spacer()
                        }
                        Text("\u{201C}\(starter.prompt)\u{201D}")
                            .font(.caption)
                            .foregroundStyle(brand.textSecondary)
                        Text(starter.primitive)
                            .font(.caption)
                            .foregroundStyle(brand.primary)
                    }
                }
                .padding(10)
                .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: 12))
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.surfaceElevated.opacity(0.4), in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
    }
}

private struct ExecutiveBusinessCaseCard: View {
    @Environment(\.brand) private var brand

    private let monthlySupportContacts = 1_000_000.0
    private let oneTapContainmentRate = 0.20
    private let savedPerContainedContact = 7.50
    private let avoidedTruckRollsPerMonth = 500.0
    private let truckRollCost = 150.0
    private let broadbandBase = 500_000.0
    private let planFitConversionRate = 0.01
    private let planFitMonthlyValue = 30.0

    private var annualSupportSavings: Double {
        monthlySupportContacts * oneTapContainmentRate * savedPerContainedContact * 12
    }

    private var annualTruckRollSavings: Double {
        avoidedTruckRollsPerMonth * truckRollCost * 12
    }

    private var annualPlanFitValue: Double {
        broadbandBase * planFitConversionRate * planFitMonthlyValue * 12
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Illustrative annual impact")
                        .font(.caption)
                        .foregroundStyle(brand.textSecondary)
                        .textCase(.uppercase)
                    Text("This is the CFO slide in product form")
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundStyle(brand.textPrimary)
                    Text("Assumptions: 1M monthly support contacts, 20% one-tap containment, 500 avoided truck rolls per month, and 1% plan-fit conversion on a 500k broadband base.")
                        .font(.caption)
                        .foregroundStyle(brand.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Text("replace with carrier numbers live")
                    .font(.caption2)
                    .foregroundStyle(brand.textSecondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(brand.surfaceElevated, in: Capsule())
            }

            HStack(spacing: 12) {
                AnnualValueStat(
                    label: "Support opex",
                    value: format(annualSupportSavings),
                    note: "one-tap tier-1 containment"
                )
                AnnualValueStat(
                    label: "Truck rolls",
                    value: format(annualTruckRollSavings),
                    note: "fewer unnecessary dispatches"
                )
                AnnualValueStat(
                    label: "ARPU upside",
                    value: format(annualPlanFitValue),
                    note: "plan-fit and save offers"
                )
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.primary.opacity(0.08), in: RoundedRectangle(cornerRadius: 20))
        .overlay(
            RoundedRectangle(cornerRadius: 20)
                .stroke(brand.primary.opacity(0.18), lineWidth: 1)
        )
    }

    private func format(_ value: Double) -> String {
        DashboardFormatters.whole.string(from: NSNumber(value: value)) ?? "$0"
    }
}

private struct AnnualValueStat: View {
    let label: String
    let value: String
    let note: String

    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
            Text(value)
                .font(.title3)
                .fontWeight(.bold)
                .monospacedDigit()
                .foregroundStyle(brand.textPrimary)
            Text(note)
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: 14))
    }
}

private struct OperatorImpactCard: View {
    @ObservedObject var ledger: TokenLedger
    @ObservedObject var stats: SessionStats
    @Environment(\.brand) private var brand

    private let truckRollValueUSD = 150.0

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Operator impact")
                    .font(.caption)
                    .foregroundStyle(brand.textSecondary)
                    .textCase(.uppercase)
                Spacer()
                Text("support ops + compliance")
                    .font(.caption2)
                    .foregroundStyle(brand.textSecondary)
            }

            HStack(spacing: 20) {
                ARPUStat(
                    label: "Contained",
                    value: "\(ledger.messagesOnDevice + ledger.messagesDeflected)",
                    sub: "interactions kept local"
                )
                ARPUStat(
                    label: "Agentic actions",
                    value: "\(stats.toolExecutions)",
                    sub: "tools completed"
                )
                ARPUStat(
                    label: "Truck-roll risk",
                    value: DashboardFormatters.whole.string(from: NSNumber(value: Double(stats.truckRollRisksAvoided) * truckRollValueUSD)) ?? "$0",
                    sub: "\(stats.truckRollRisksAvoided) extender saves"
                )
            }

            HStack(spacing: 20) {
                ARPUStat(
                    label: "Dispatches booked",
                    value: "\(stats.appointmentsScheduled)",
                    sub: "field visits scheduled"
                )
                ARPUStat(
                    label: "Privacy approved",
                    value: "\(stats.privacyPreflightsApproved)",
                    sub: "\(stats.privacyPreflightsCancelled) canceled"
                )
                ARPUStat(
                    label: "PII protected",
                    value: "\(stats.piiInstancesCaught)",
                    sub: "elements scrubbed on device"
                )
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
    }
}

/// ARPU / retention card — the number carrier finance teams care about:
/// how many dollars of revenue (or retention credit) this on-device
/// AI surfaced to the customer this session.
private struct ARPUSignalCard: View {
    @ObservedObject var engine: NextBestActionEngine
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("ARPU signal")
                    .font(.caption).foregroundStyle(brand.textSecondary).textCase(.uppercase)
                Spacer()
                Text("Personalization impact")
                    .font(.caption2).foregroundStyle(brand.textSecondary)
            }

            HStack(spacing: 20) {
                ARPUStat(
                    label: "Revenue surfaced",
                    value: formatDollars(engine.surfacedMonthlyValueDollars),
                    sub: "monthly upsell offered"
                )
                ARPUStat(
                    label: "Retention offered",
                    value: formatDollars(engine.surfacedRetentionCostDollars),
                    sub: "credits on the table"
                )
                ARPUStat(
                    label: "Accepted (net)",
                    value: formatDollars(engine.acceptedMonthlyValueDollars),
                    sub: "\(engine.acceptedCount) offers taken"
                )
            }

            Text("Next-Best-Action engine scores upsells, retention credits, plan fit, add-ons, and proactive support against on-device usage telemetry. A 5–15% ARPU uplift is industry-reported for personalization of this type (BCG, 2026).")
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
    }

    private func formatDollars(_ value: Double) -> String {
        DashboardFormatters.whole.string(from: NSNumber(value: value)) ?? "$0"
    }
}

private struct ARPUStat: View {
    let label: String
    let value: String
    let sub: String
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(brand.textSecondary)
            Text(value)
                .font(.title3).fontWeight(.bold)
                .monospacedDigit()
                .foregroundStyle(brand.textPrimary)
            Text(sub).font(.caption2).foregroundStyle(brand.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct HeaderTile: View {
    @ObservedObject var ledger: TokenLedger
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(spacing: 10) {
            Text("Live session proof")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .textCase(.uppercase)
            Text(formatDollars(ledger.netDollarsSaved))
                .font(.system(size: 44, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(ledger.netDollarsSaved >= 0 ? brand.success : brand.danger)
            HStack(spacing: 16) {
                SplitLabel(
                    icon: "iphone",
                    label: "Saved on-device",
                    value: formatDollars(ledger.estimatedDollarsSaved),
                    tint: brand.success
                )
                SplitLabel(
                    icon: "cloud.fill",
                    label: "Spent in cloud",
                    value: formatDollars(ledger.estimatedDollarsSpentInCloud),
                    tint: brand.info
                )
            }
            Text("Saved at competitor-cloud rates · spent at Liquid cloud rates")
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
        }
        .frame(maxWidth: .infinity)
        .padding(20)
        .background(brand.primary.opacity(0.08), in: RoundedRectangle(cornerRadius: 20))
    }

    private func formatDollars(_ value: Double) -> String {
        DashboardFormatters.auto(value)
    }
}

private struct SessionTiles: View {
    @ObservedObject var ledger: TokenLedger
    @ObservedObject var stats: SessionStats

    var body: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            MetricTile(
                label: "On-device messages",
                value: "\(ledger.messagesOnDevice + ledger.messagesDeflected)",
                sublabel: "of \(ledger.messagesOnDevice + ledger.messagesDeflected + ledger.messagesCloudEscalated) total"
            )
            MetricTile(
                label: "% on-device",
                value: String(format: "%.0f%%", ledger.percentOnDevice),
                sublabel: "share of traffic"
            )
            MetricTile(
                label: "Avg latency",
                value: "\(stats.averageLatencyMS)ms",
                sublabel: "p95: \(stats.p95LatencyMS)ms"
            )
            MetricTile(
                label: "PII caught",
                value: "\(stats.piiInstancesCaught)",
                sublabel: "redacted before any cloud call"
            )
            MetricTile(
                label: "Tool calls",
                value: "\(ledger.messagesDeflected)",
                sublabel: "executed on-device"
            )
            MetricTile(
                label: "Cloud escalations",
                value: "\(ledger.messagesCloudEscalated)",
                sublabel: "long-tail → cloud"
            )
            MetricTile(
                label: "Tokens saved",
                value: "\(ledger.totalTokensSaved)",
                sublabel: "in ↑ \(ledger.inputTokensSaved) / out ↓ \(ledger.outputTokensSaved)"
            )
            MetricTile(
                label: "Tokens spent (cloud)",
                value: "\(ledger.totalTokensSpentInCloud)",
                sublabel: "in ↑ \(ledger.inputTokensSpentInCloud) / out ↓ \(ledger.outputTokensSpentInCloud)"
            )
        }
    }
}

private struct SplitLabel: View {
    let icon: String
    let label: String
    let value: String
    let tint: Color

    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption2)
                Text(label)
                    .font(.caption2)
            }
            .foregroundStyle(tint)
            Text(value)
                .font(.subheadline).fontWeight(.semibold)
                .monospacedDigit()
        }
    }
}

private struct MetricTile: View {
    let label: String
    let value: String
    let sublabel: String
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.caption).foregroundStyle(brand.textSecondary).textCase(.uppercase)
            Text(value).font(.system(size: 22, weight: .bold, design: .rounded)).monospacedDigit()
                .foregroundStyle(brand.textPrimary)
            Text(sublabel).font(.caption2).foregroundStyle(brand.textSecondary)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
    }
}

private struct PricingFootnote: View {
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Pricing basis").font(.caption).fontWeight(.semibold).foregroundStyle(brand.textPrimary)
            Text("**Savings side** · competitor cloud: $5.00 in / $15.00 out per 1M tokens (GPT-4-class, what you'd pay without Liquid).")
                .font(.caption2).foregroundStyle(brand.textSecondary)
            Text("**Spend side** · Liquid cloud LFM: $0.50 in / $1.50 out per 1M tokens (cloud-class LFM pricing).")
                .font(.caption2).foregroundStyle(brand.textSecondary)
            Text("Net savings = saved at competitor rates − spent at Liquid rates. Configurable in `TokenLedger.Pricing`.")
                .font(.caption2).foregroundStyle(brand.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(brand.surfaceElevated.opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
    }
}
