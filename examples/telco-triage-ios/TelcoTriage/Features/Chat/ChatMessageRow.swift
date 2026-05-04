import SwiftUI
import UIKit

struct ChatMessageRow: View {
    let message: ChatMessage
    let onTapPII: () -> Void
    let onExecuteVisionTool: ((String, [String: String]) -> Void)?
    let nbaForMessage: (String) -> (any NextBestAction)?
    let onAcceptNBA: (String) -> Void
    let onDeclineNBA: (String) -> Void
    let onConfirmTool: (UUID) -> Void
    let onDeclineTool: (UUID) -> Void
    let onOpenArticle: (KBEntry) -> Void
    /// Binding to the per-message pipeline-card expand state. The
    /// canonical store lives in `ChatViewModel.expandedTraceMessageIDs`
    /// so collapsed cards stay collapsed across `LazyVStack` recycling.
    let traceExpandedBinding: Binding<Bool>

    @Environment(\.brand) private var brand
    @Environment(\.appMode) private var appMode

    init(
        message: ChatMessage,
        onTapPII: @escaping () -> Void,
        onExecuteVisionTool: ((String, [String: String]) -> Void)? = nil,
        nbaForMessage: @escaping (String) -> (any NextBestAction)?,
        onAcceptNBA: @escaping (String) -> Void,
        onDeclineNBA: @escaping (String) -> Void,
        onConfirmTool: @escaping (UUID) -> Void,
        onDeclineTool: @escaping (UUID) -> Void,
        onOpenArticle: @escaping (KBEntry) -> Void,
        traceExpandedBinding: Binding<Bool> = .constant(true)
    ) {
        self.message = message
        self.onTapPII = onTapPII
        self.onExecuteVisionTool = onExecuteVisionTool
        self.nbaForMessage = nbaForMessage
        self.onAcceptNBA = onAcceptNBA
        self.onDeclineNBA = onDeclineNBA
        self.onConfirmTool = onConfirmTool
        self.onDeclineTool = onDeclineTool
        self.onOpenArticle = onOpenArticle
        self.traceExpandedBinding = traceExpandedBinding
    }

    var body: some View {
        HStack(alignment: .top) {
            if message.role == .user { Spacer(minLength: 60) }
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
                bubble
                if message.role == .assistant {
                    // Engineering mode: pipeline card when the multi-head
                    // classifier produced a 9-head trace. Otherwise fall
                    // back to the flat 4-cell trace row (vision, voice,
                    // generative fallback paths).
                    if appMode == .engineering,
                       let trace = message.trace, let routing = message.routing {
                        if let pipeline = trace.telcoPipeline {
                            TelcoPipelineCard(trace: pipeline,
                                              expanded: traceExpandedBinding)
                        } else {
                            TraceRow(trace: trace, routingPath: routing.path)
                        }
                    }
                    // Customer mode: subtle "on-device" badge instead of full trace
                    if appMode == .customer, message.trace != nil {
                        onDeviceBadge
                    }
                    // Action first, then supplemental reference. A user who
                    // reads "How do I restart?" + sees a "Restart Router"
                    // button should tap the button. The "Read full article"
                    // chip is the fallback for people who want to read
                    // instead of act — it belongs below the primary CTA.
                    if !message.deepLinks.isEmpty {
                        deepLinkRow
                    }
                    if let entry = message.sourceEntry {
                        readFullArticleChip(entry: entry)
                    }
                    if let diagnosis = message.visionDiagnosis {
                        VisionDiagnosisCard(
                            diagnosis: diagnosis,
                            onExecuteTool: onExecuteVisionTool.map { handler in
                                { toolID in handler(toolID, diagnosis.proposedArguments) }
                            }
                        )
                    }
                    // Engineering mode: inline tool decision card
                    if appMode == .engineering, let decision = message.toolDecision {
                        ToolDecisionCard(
                            decision: decision,
                            onConfirm: { onConfirmTool(message.id) },
                            onDecline: { onDeclineTool(message.id) }
                        )
                    }
                    if let nbaID = message.attachedNBAID, let nba = nbaForMessage(nbaID) {
                        NextBestActionCard(
                            action: nba,
                            onAccept: { onAcceptNBA(nbaID) },
                            onDecline: { onDeclineNBA(nbaID) }
                        )
                    }
                }
            }
            if message.role == .assistant { Spacer(minLength: 60) }
        }
        .padding(.horizontal, 14)
    }

    private var bubble: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let image = message.attachedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(maxWidth: 220, maxHeight: 220)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            if message.role == .user, !message.piiSpans.isEmpty {
                PIIWarningChip(count: message.piiSpans.count, action: onTapPII)
            }
            if message.voiceInput {
                HStack(spacing: 4) {
                    Image(systemName: "mic.fill").font(.caption2)
                    Text("voice").font(.caption2)
                }
                .opacity(0.8)
            }
            Text(LocalizedStringKey(message.text))
                .font(brand.bodyFont)
                .foregroundStyle(message.role == .user ? brand.onPrimary : brand.textPrimary)
                .textSelection(.enabled)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(bubbleFill, in: RoundedRectangle(cornerRadius: brand.bubbleCornerRadius))
    }

    private var bubbleFill: Color {
        message.role == .user ? brand.primary : brand.surfaceElevated
    }

    private func readFullArticleChip(entry: KBEntry) -> some View {
        Button {
            onOpenArticle(entry)
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "doc.text.magnifyingglass")
                Text("Read full article")
                    .font(.caption)
                    .fontWeight(.semibold)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .overlay(Capsule().stroke(brand.border, lineWidth: 1))
            .foregroundStyle(brand.textPrimary)
        }
        .buttonStyle(.plain)
    }

    /// Minimal customer-facing badge confirming the response was generated
    /// on-device. Replaces the full TraceRow in customer mode — the "aha"
    /// without the noise.
    private var onDeviceBadge: some View {
        HStack(spacing: 4) {
            Image(systemName: "iphone")
                .font(.caption2)
            Text("On-device")
                .font(.caption2)
                .fontWeight(.medium)
            if let ms = message.trace?.customerVisibleMS {
                Text("· \(ms)ms")
                    .font(.caption2)
                    .monospacedDigit()
            }
        }
        .foregroundStyle(brand.textSecondary)
        .padding(.leading, 4)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(onDeviceBadgeAccessibilityLabel)
    }

    private var onDeviceBadgeAccessibilityLabel: String {
        if let ms = message.trace?.customerVisibleMS {
            return "Generated on device in \(ms) milliseconds"
        }
        return "Generated on device"
    }

    private var deepLinkRow: some View {
        HStack(spacing: 8) {
            ForEach(message.deepLinks, id: \.url) { link in
                DeepLinkChip(link: link)
            }
        }
        .padding(.leading, 4)
    }
}

struct PIIWarningChip: View {
    let count: Int
    let action: () -> Void

    @Environment(\.brand) private var brand

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: "shield.lefthalf.filled")
                Text("\(count) PII caught • Tap to inspect")
                    .font(.caption2)
                    .fontWeight(.semibold)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(brand.warning.opacity(0.2), in: Capsule())
            .foregroundStyle(brand.warning)
        }
        .buttonStyle(.plain)
    }
}

/// Static capsule showing where in the carrier home internet app the assistant
/// would navigate. Non-interactive by design — we surface the route
/// visually without attempting `UIApplication.shared.open`, which
/// would fail on a demo phone without the carrier app installed.
struct DeepLinkChip: View {
    let link: DeepLink
    @Environment(\.brand) private var brand

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: "arrow.up.right.square")
            Text(link.label)
                .font(.caption)
                .fontWeight(.medium)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .overlay(Capsule().stroke(brand.border, lineWidth: 1))
        .foregroundStyle(brand.textPrimary)
        // Decorative non-interactive route hint — VoiceOver should
        // announce its purpose so users on screen readers don't
        // mistake the chip for a tappable button.
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Suggested destination: \(link.label)")
        .accessibilityHint("Navigate in the carrier app to access this section")
    }
}

/// Shows what the on-device model understood: which tool it selected,
/// what arguments it extracted, and how confident it is. Confirm runs
/// the tool via `ToolExecutor`; Decline drops the proposal.
struct ToolDecisionCard: View {
    let decision: ToolDecision
    let onConfirm: () -> Void
    let onDecline: () -> Void
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            if !decision.arguments.isEmpty {
                extractedArgumentsSection
            }
            if let reasoning = decision.reasoning {
                reasoningLine(reasoning)
            }
            actionRow
        }
        .padding(12)
        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: brand.cardCornerRadius)
                .stroke(brand.textSecondary.opacity(0.2), lineWidth: 1)
        )
    }

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: decision.icon)
                .font(.system(size: 18))
                .frame(width: 34, height: 34)
                .background(brand.textSecondary.opacity(0.12), in: Circle())
                .foregroundStyle(brand.textPrimary)
            VStack(alignment: .leading, spacing: 2) {
                Text(decision.displayName)
                    .font(.callout)
                    .fontWeight(.semibold)
                    .foregroundStyle(brand.textPrimary)
                HStack(spacing: 6) {
                    Text("Tool selected")
                        .font(.caption2)
                        .foregroundStyle(brand.textSecondary)
                    if decision.isDestructive {
                        Text("DESTRUCTIVE")
                            .font(.caption2).fontWeight(.bold)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(brand.warning.opacity(0.15), in: Capsule())
                            .foregroundStyle(brand.warning)
                    }
                }
            }
            Spacer()
            Text(String(format: "%.0f%%", decision.confidence * 100))
                .font(.caption).fontWeight(.bold)
                .monospacedDigit()
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(confidenceTint.opacity(0.12), in: Capsule())
                .foregroundStyle(confidenceTint)
        }
    }

    private var extractedArgumentsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Extracted arguments")
                .font(.caption2)
                .foregroundStyle(brand.textSecondary)
                .textCase(.uppercase)
            ForEach(decision.arguments) { arg in
                argumentRow(arg)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.surfaceBackground, in: RoundedRectangle(cornerRadius: 8))
    }

    /// Argument row that switches from horizontal to vertical layout
    /// at accessibility-size Dynamic Type. The original layout used a
    /// fixed `minWidth: 80` for the label column, which collided with
    /// the value column at AX2+ and clipped both. `ViewThatFits`
    /// chooses the horizontal HStack when the content fits and falls
    /// back to a stacked label/value when the system text scale grows.
    @ViewBuilder
    private func argumentRow(_ arg: ToolDecisionArgument) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) {
                Text(arg.label)
                    .font(brand.monoFont)
                    .foregroundStyle(brand.textSecondary)
                    .layoutPriority(0)
                Text(arg.value)
                    .font(brand.monoFont)
                    .fontWeight(.medium)
                    .foregroundStyle(brand.textPrimary)
                    .layoutPriority(1)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(arg.label)
                    .font(.caption2)
                    .foregroundStyle(brand.textSecondary)
                    .textCase(.uppercase)
                Text(arg.value)
                    .font(brand.monoFont)
                    .fontWeight(.medium)
                    .foregroundStyle(brand.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(arg.label): \(arg.value)")
    }

    private func reasoningLine(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "brain")
                .font(.caption2)
                .foregroundStyle(brand.accent)
            Text(text)
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var actionRow: some View {
        HStack(spacing: 8) {
            Button(action: onDecline) {
                Text("Not now")
                    .font(.callout)
                    .fontWeight(.medium)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 9)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(brand.border, lineWidth: 1)
                    )
                    .foregroundStyle(brand.textPrimary)
            }
            .buttonStyle(.plain)

            Button(action: onConfirm) {
                Text(decision.isDestructive ? "Confirm" : "Run")
                    .font(.callout)
                    .fontWeight(.semibold)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 9)
                    .background(brand.primary, in: RoundedRectangle(cornerRadius: 10))
                    .foregroundStyle(brand.onPrimary)
            }
            .buttonStyle(.plain)
        }
    }

    private var confidenceTint: Color {
        if decision.confidence >= 0.8 { return brand.success }
        if decision.confidence >= 0.5 { return brand.warning }
        return brand.danger
    }
}
