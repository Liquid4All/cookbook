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
    /// Binding for the ADR-026 LFM understanding card. This disclosure
    /// is collapsed by default so engineering mode stays readable until
    /// someone explicitly asks for all nine head outputs.
    let telcoUnderstandingExpandedBinding: Binding<Bool>

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
        traceExpandedBinding: Binding<Bool> = .constant(true),
        telcoUnderstandingExpandedBinding: Binding<Bool> = .constant(false)
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
        self.telcoUnderstandingExpandedBinding = telcoUnderstandingExpandedBinding
    }

    var body: some View {
        HStack(alignment: .top) {
            if message.role == .user { Spacer(minLength: 54) }
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 7) {
                bubble
                if message.role == .assistant {
                    if appMode == .customer, message.trace != nil {
                        onDeviceBadge
                    }
                    if let entry = message.sourceEntry {
                        readFullArticleChip(entry: entry)
                    }
                    if !message.deepLinks.isEmpty {
                        deepLinkRow
                    }
                    // Engineering mode: trace row showing routing path,
                    // chat mode + confidence, latency, etc. The legacy
                    // 9-head TelcoPipelineCard was deleted along with
                    // the multi-head decision engine that fed it.
                    if appMode == .engineering,
                       let trace = message.trace, let routing = message.routing {
                        TraceRow(trace: trace, routingPath: routing.path)
                        if let telcoUnderstanding = trace.telcoUnderstanding {
                            TelcoUnderstandingTraceDisclosure(
                                understanding: telcoUnderstanding,
                                isExpanded: telcoUnderstandingExpandedBinding
                            )
                        }
                        // ADR-022 §4.3 Layer 4 — full 5-head understanding
                        // vector. Renders below TraceRow so the trace
                        // reads top→down as: routing summary, then the
                        // signal the router consumed to make that choice.
                        if let understanding = trace.understanding {
                            UnderstandingTraceCard(understanding: understanding)
                        }
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
            if message.role == .assistant { Spacer(minLength: 42) }
        }
        .padding(.horizontal, 16)
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
            // Internal scheme (telcohome://) is rewritten to the brand
            // scheme at the render boundary — Stage B + page-link table
            // + grammar all use the internal scheme so the trained model
            // stays on distribution. See DeepLinkRebrand.swift.
            messageContent
        }
        .padding(.horizontal, message.role == .user ? 14 : 15)
        .padding(.vertical, message.role == .user ? 10 : 12)
        .background(bubbleFill, in: RoundedRectangle(cornerRadius: brand.bubbleCornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: brand.bubbleCornerRadius)
                .stroke(bubbleBorder, lineWidth: 1)
        )
        .shadow(
            color: message.role == .assistant ? .black.opacity(0.035) : .clear,
            radius: 8,
            y: 3
        )
    }

    private var bubbleFill: Color {
        message.role == .user ? brand.primary : brand.surfaceElevated
    }

    private var bubbleBorder: Color {
        message.role == .user ? Color.clear : brand.border
    }

    @ViewBuilder
    private var messageContent: some View {
        if let presentation = composerStepPresentation {
            ComposerStepAnswerView(presentation: presentation)
        } else if let presentation = composerSummaryPresentation {
            ComposerSummaryAnswerView(presentation: presentation)
        } else {
            Text(LocalizedStringKey(DeepLinkRebrand.forDisplay(message.text, brand: brand)))
                .font(brand.bodyFont)
                .foregroundStyle(message.role == .user ? brand.onPrimary : brand.textPrimary)
                .lineSpacing(2)
                .textSelection(.enabled)
        }
    }

    private var composerSummaryPresentation: ComposerSummaryPresentation? {
        guard message.role == .assistant,
              message.trace?.composerRoute != nil,
              !message.text.contains(" > ") else {
            return nil
        }

        let paragraphs = message.text
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !paragraphs.isEmpty else { return nil }

        var lead: String?
        var details: [String] = []
        var confirmation: String?

        for paragraph in paragraphs {
            if paragraph.localizedCaseInsensitiveContains("reply 'yes'") {
                confirmation = strippedComposerMarkdown(paragraph)
            } else if paragraph.hasPrefix("Key details:") {
                details.append(contentsOf: keyDetailLines(from: paragraph))
            } else if !isComposerNavigationCTA(paragraph) {
                let clean = strippedComposerMarkdown(paragraph)
                if lead == nil {
                    lead = clean
                } else if !clean.isEmpty {
                    details.append(clean)
                }
            }
        }

        guard let lead, !lead.isEmpty else { return nil }
        return ComposerSummaryPresentation(
            lead: lead,
            details: details,
            confirmation: confirmation
        )
    }

    private var composerStepPresentation: ComposerStepPresentation? {
        guard message.role == .assistant,
              message.trace?.composerRoute != nil,
              message.text.contains(" > ") else {
            return nil
        }
        let parts = message.text
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard parts.count >= 2 else { return nil }
        let chain = parts[1]
        guard let firstSeparator = chain.range(of: " > ") else { return nil }
        var stepText = String(chain[firstSeparator.upperBound...])
        if stepText.hasSuffix(".") {
            stepText.removeLast()
        }
        let steps = stepText
            .components(separatedBy: " > ")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !steps.isEmpty else { return nil }
        return ComposerStepPresentation(
            intro: parts[0],
            steps: steps,
            confirmation: parts.dropFirst(2).first
        )
    }

    private func readFullArticleChip(entry: KBEntry) -> some View {
        Button {
            onOpenArticle(entry)
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "doc.text")
                Text("Source: \(entry.topic)")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .lineLimit(1)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(brand.surfaceElevated, in: Capsule())
            .overlay(Capsule().stroke(brand.border, lineWidth: 1))
            .foregroundStyle(brand.textSecondary)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Open source article, \(entry.topic)")
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
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(brand.surfaceElevated, in: Capsule())
        .overlay(Capsule().stroke(brand.border.opacity(0.7), lineWidth: 1))
        .padding(.leading, 2)
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

private struct ComposerStepPresentation: Equatable {
    let intro: String
    let steps: [String]
    let confirmation: String?
}

private struct ComposerSummaryPresentation: Equatable {
    let lead: String
    let details: [String]
    let confirmation: String?
}

private struct ComposerStepAnswerView: View {
    let presentation: ComposerStepPresentation

    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(presentation.intro)
                .font(brand.bodyFont)
                .foregroundStyle(brand.textPrimary)
                .lineSpacing(2)
                .textSelection(.enabled)

            VStack(alignment: .leading, spacing: 9) {
                Text("Steps")
                    .font(.caption2)
                    .fontWeight(.semibold)
                    .foregroundStyle(brand.textSecondary)
                    .textCase(.uppercase)
                ForEach(Array(presentation.steps.enumerated()), id: \.offset) { index, step in
                    HStack(alignment: .top, spacing: 9) {
                        Text("\(index + 1)")
                            .font(.caption2)
                            .fontWeight(.bold)
                            .foregroundStyle(brand.textPrimary)
                            .frame(width: 20, height: 20)
                            .background(brand.textPrimary.opacity(0.06), in: Circle())
                        Text(step)
                            .font(brand.bodyFont)
                            .foregroundStyle(brand.textPrimary)
                            .fixedSize(horizontal: false, vertical: true)
                            .textSelection(.enabled)
                    }
                }
            }
            .padding(11)
            .background(brand.textPrimary.opacity(0.035),
                        in: RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(brand.border.opacity(0.7), lineWidth: 1)
            )

            if let confirmation = presentation.confirmation {
                HStack(alignment: .top, spacing: 7) {
                    Image(systemName: "checkmark.shield")
                        .font(.caption)
                        .foregroundStyle(brand.success)
                        .padding(.top, 2)
                    Text(confirmation)
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundStyle(brand.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.top, 1)
            }
        }
    }
}

private struct ComposerSummaryAnswerView: View {
    let presentation: ComposerSummaryPresentation

    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(presentation.lead)
                .font(brand.bodyFont)
                .foregroundStyle(brand.textPrimary)
                .lineSpacing(2)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)

            if !presentation.details.isEmpty {
                VStack(alignment: .leading, spacing: 9) {
                    Text("Key details")
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundStyle(brand.textSecondary)
                        .textCase(.uppercase)
                    ForEach(Array(presentation.details.enumerated()), id: \.offset) { _, detail in
                        HStack(alignment: .top, spacing: 9) {
                            Circle()
                                .fill(brand.textPrimary.opacity(0.16))
                                .frame(width: 7, height: 7)
                                .padding(.top, 8)
                            Text(detail)
                                .font(brand.bodyFont)
                                .foregroundStyle(brand.textPrimary)
                                .fixedSize(horizontal: false, vertical: true)
                                .textSelection(.enabled)
                        }
                    }
                }
                .padding(11)
                .background(brand.textPrimary.opacity(0.035),
                            in: RoundedRectangle(cornerRadius: 10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(brand.border.opacity(0.7), lineWidth: 1)
                )
            }

            if let confirmation = presentation.confirmation {
                HStack(alignment: .top, spacing: 7) {
                    Image(systemName: "checkmark.shield")
                        .font(.caption)
                        .foregroundStyle(brand.success)
                        .padding(.top, 2)
                    Text(confirmation)
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundStyle(brand.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.top, 1)
            }
        }
    }
}

private func keyDetailLines(from paragraph: String) -> [String] {
    paragraph
        .components(separatedBy: .newlines)
        .dropFirst()
        .map { raw in
            var line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.hasPrefix("-") {
                line.removeFirst()
            }
            return strippedComposerMarkdown(line)
        }
        .filter { !$0.isEmpty }
}

private func strippedComposerMarkdown(_ text: String) -> String {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    let nsText = trimmed as NSString
    guard let regex = try? NSRegularExpression(pattern: #"\[([^\]\n]+)\]\([^)]+\)"#) else {
        return trimmed
    }
    let mutable = NSMutableString(string: trimmed)
    let range = NSRange(location: 0, length: nsText.length)
    for match in regex.matches(in: trimmed, range: range).reversed() {
        guard match.numberOfRanges >= 2 else { continue }
        mutable.replaceCharacters(in: match.range(at: 0),
                                  with: nsText.substring(with: match.range(at: 1)))
    }
    return String(mutable).trimmingCharacters(in: .whitespacesAndNewlines)
}

private func isComposerNavigationCTA(_ paragraph: String) -> Bool {
    let lowered = paragraph.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    return paragraph.contains("](") &&
        (lowered.hasPrefix("open ") ||
         lowered.hasPrefix("you can continue") ||
         lowered.hasPrefix("to manage your account"))
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
            Image(systemName: "arrow.turn.down.right")
                .font(.caption2)
            Text(link.label == "Open in app" ? "Open in app" : "Open \(link.label)")
                .font(.caption)
                .fontWeight(.semibold)
                .lineLimit(1)
        }
        .padding(.horizontal, 11)
        .padding(.vertical, 6)
        .background(brand.textPrimary.opacity(0.055), in: Capsule())
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
            if decision.isCompoundAttachment {
                compoundCaption
            }
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

    /// Caption that re-frames the card as a one-tap shortcut beneath a
    /// how-to article, rather than the primary CTA. ADR-022 §4.3
    /// compound-response review — when the imperative tool detector
    /// fires on a RAG / unknown-feature / clarification turn, we render
    /// BOTH the article AND the action affordance so the user can read
    /// the explanation OR tap once. This caption is the visual cue.
    private var compoundCaption: some View {
        HStack(spacing: 6) {
            Image(systemName: "wand.and.stars")
                .font(.caption2)
                .foregroundStyle(brand.accent)
            Text("Or, want me to do this for you?")
                .font(.caption).fontWeight(.semibold)
                .foregroundStyle(brand.textSecondary)
        }
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
