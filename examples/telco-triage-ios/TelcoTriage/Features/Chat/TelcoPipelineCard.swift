import SwiftUI

/// Engineering-mode pipeline trace card. RBC-style: header with the
/// On-Device pill + total latency + intent confidence; an expandable
/// detail panel with header grid, primary classifyAll row, optional
/// downstream stage row, and a Reason footer.
///
/// UX notes:
///  - The whole header is tappable (the chevron is visual; the hit
///    target is the entire row via `.contentShape(Rectangle())`).
///  - Long values truncate to one line with `.middle` truncation —
///    Dynamic Type at large sizes never overflows the bubble.
///  - All raw schema labels (`agent_handoff`, `local_answer`) are
///    converted via `TelcoLabelDisplay` so the demo reads in plain
///    English.
///  - Latency tint is reserved for *non-zero* latency; downstream
///    rows that ride the same embedding render as `0ms` in the
///    secondary color, never the warning hue.
///  - Confidence chips use color *only* when the band is informative
///    (warning / danger). A 100% confidence reads as a calm "100%",
///    not a victory-green badge.
struct TelcoPipelineCard: View {
    let trace: TelcoPipelineTrace
    /// Owned by the parent (`ChatViewModel.expandedTraceMessageIDs`) so
    /// the expand/collapse state survives `LazyVStack` cell recycling.
    @Binding var expanded: Bool

    @Environment(\.brand) private var brand

    private let cornerRadius: CGFloat = 12
    private let edgePadding: CGFloat = 14
    private let rowVerticalPadding: CGFloat = 10

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
                .contentShape(Rectangle())
                .onTapGesture {
                    withAnimation(.easeInOut(duration: 0.2)) { expanded.toggle() }
                }
                .accessibilityElement(children: .combine)
                .accessibilityAddTraits(.isButton)
                .accessibilityLabel(headerAccessibilityLabel)
                .accessibilityHint(expanded ? "Tap to collapse" : "Tap to expand details")

            if expanded {
                separator
                stepRow(trace.primaryStep)
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel(stepAccessibilityLabel(trace.primaryStep))

                if let downstream = trace.downstreamStep {
                    separator
                    stepRow(downstream)
                        .accessibilityElement(children: .combine)
                        .accessibilityLabel(stepAccessibilityLabel(downstream))
                }

                if let reason = trace.laneReason {
                    separator
                    reasonRow(reason: reason)
                        .accessibilityElement(children: .combine)
                        .accessibilityLabel("Reason. \(reason)")
                }
            }
        }
        .background(brand.surfaceElevated.opacity(0.6),
                    in: RoundedRectangle(cornerRadius: cornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: cornerRadius)
                .stroke(brand.textSecondary.opacity(0.18), lineWidth: 0.5)
        )
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius))
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                onDevicePill
                Text(formatMs(trace.totalLatencyMs))
                    .font(brand.monoFont.weight(.medium))
                    .foregroundStyle(latencyTint(trace.totalLatencyMs))
                    .accessibilityLabel("Latency \(formatMs(trace.totalLatencyMs))")

                Text("\(percentText(trace.intentConfidence)) conf")
                    .font(.caption)
                    .foregroundStyle(brand.textSecondary)

                Spacer(minLength: 8)

                Image(systemName: "chevron.down")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(brand.textSecondary)
                    .rotationEffect(.degrees(expanded ? 180 : 0))
                    .animation(.easeInOut(duration: 0.2), value: expanded)
                    .frame(width: 20, height: 20)
            }

            if expanded {
                headerGrid
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(edgePadding)
    }

    private var onDevicePill: some View {
        Label("On-Device", systemImage: "iphone")
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 8).padding(.vertical, 3)
            .overlay(
                Capsule().stroke(brand.success.opacity(0.6), lineWidth: 1)
            )
            .foregroundStyle(brand.success)
            .accessibilityLabel("Running on device")
    }

    private var headerGrid: some View {
        // Two-column key/value layout. Values truncate cleanly; Dynamic
        // Type up to AX2 stays readable thanks to minimumScaleFactor.
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
            gridRow("Source", "Multi-head shared classifier")
            gridRow("Intent", trace.intent)
            gridRow("Model",  trace.modelName)
            gridRow("Answer", trace.answerSummary)
            if let target = trace.target {
                gridRow("Target", target)
            }
        }
        .font(.caption)
        .padding(.top, 2)
    }

    @ViewBuilder
    private func gridRow(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label)
                .foregroundStyle(brand.textSecondary)
                .gridColumnAlignment(.leading)
                .accessibilityHidden(true)

            Text(value)
                .font(brand.monoFont)
                .foregroundStyle(brand.textPrimary)
                .multilineTextAlignment(.trailing)
                .lineLimit(2)
                .truncationMode(.tail)
                .minimumScaleFactor(0.85)
                .frame(maxWidth: .infinity, alignment: .trailing)
                .accessibilityLabel("\(label): \(value)")
        }
    }

    // MARK: - Step rows

    private func stepRow(_ step: TelcoPipelineTrace.Step) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(step.title)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(brand.textPrimary)
                Spacer(minLength: 8)
                Text(latencyText(step.latencyMs))
                    .font(brand.monoFont.weight(.medium))
                    .foregroundStyle(latencyTint(step.latencyMs))
            }

            HStack(spacing: 6) {
                Text(step.modelTag)
                    .font(.caption2)
                    .foregroundStyle(brand.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                if let conf = step.confidence {
                    Text(percentText(conf))
                        .font(brand.monoFont.weight(.medium))
                        .foregroundStyle(confidenceTint(conf))
                }
            }

            Text(step.detail)
                .font(brand.monoFont)
                .foregroundStyle(brand.textPrimary)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, edgePadding)
        .padding(.vertical, rowVerticalPadding)
    }

    private func reasonRow(reason: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text("Reason")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
            Spacer(minLength: 8)
            Text(reason)
                .font(brand.monoFont)
                .foregroundStyle(brand.textPrimary)
                .multilineTextAlignment(.trailing)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, edgePadding)
        .padding(.vertical, rowVerticalPadding)
    }

    // MARK: - Reusable separator

    /// 0.5pt hairline rendered as a Color rect — `Divider().background(...)`
    /// is unreliable across iOS versions and ignores some color schemes.
    private var separator: some View {
        brand.textSecondary
            .opacity(0.18)
            .frame(height: 0.5)
    }

    // MARK: - Formatting

    private func formatMs(_ ms: Double) -> String {
        if ms <= 0 { return "0ms" }
        if ms < 10 { return String(format: "%.1fms", ms) }
        return String(format: "%.0fms", ms)
    }

    private func latencyText(_ ms: Double) -> String {
        formatMs(ms)
    }

    private func percentText(_ ratio: Double) -> String {
        let clamped = max(0, min(1, ratio))
        return String(format: "%.0f%%", clamped * 100)
    }

    /// Latency tint reserved for measurable cost. `0ms` rows ride the
    /// shared embedding and shouldn't compete visually with the
    /// primary classifier latency.
    private func latencyTint(_ ms: Double) -> Color {
        ms > 0 ? brand.warning : brand.textSecondary
    }

    /// Confidence color rule: tint only when the score is informative.
    /// High-confidence (>=80%) reads as plain primary text — green
    /// badges on every percent become noise. Low/medium bands warrant
    /// a hue because they signal the model is uncertain.
    private func confidenceTint(_ c: Double) -> Color {
        switch ConfidenceBand.classify(c) {
        case .neutral: return brand.textSecondary
        case .high:    return brand.textPrimary
        case .medium:  return brand.warning
        case .low:     return brand.danger
        }
    }

    // MARK: - Accessibility

    private var headerAccessibilityLabel: String {
        let lat = formatMs(trace.totalLatencyMs)
        let conf = percentText(trace.intentConfidence)
        return "Pipeline trace, on device. Intent \(trace.intent). \(lat) total. \(conf) confidence. \(expanded ? "Expanded" : "Collapsed")."
    }

    private func stepAccessibilityLabel(_ step: TelcoPipelineTrace.Step) -> String {
        var parts: [String] = [step.title]
        if let conf = step.confidence {
            parts.append("\(percentText(conf)) confidence")
        }
        parts.append(formatMs(step.latencyMs))
        parts.append(step.detail)
        return parts.joined(separator: ". ")
    }
}
