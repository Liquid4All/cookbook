import SwiftUI

/// Renders a `SupportSignal` above the input bar in chat. Tapping the
/// card auto-sends the signal's suggested prompt through the normal
/// NLU pipeline — so the demo preserves the "model parses natural
/// language" story even when the customer takes the one-tap path.
struct SupportSignalCard: View {
    let signal: SupportSignal
    let onTap: () -> Void

    @Environment(\.brand) private var brand

    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: signal.icon)
                    .font(.system(size: 16))
                    .foregroundStyle(tint)
                    .frame(width: 32, height: 32)
                    .background(tint.opacity(0.12), in: Circle())

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(severityLabel)
                            .font(.caption2)
                            .fontWeight(.bold)
                            .foregroundStyle(tint)
                            .textCase(.uppercase)
                        Text("·")
                            .font(.caption2)
                            .foregroundStyle(brand.textSecondary)
                        Text(signal.domain.rawValue)
                            .font(.caption2)
                            .foregroundStyle(brand.textSecondary)
                            .textCase(.uppercase)
                        Spacer()
                    }
                    Text(signal.title)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundStyle(brand.textPrimary)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(signal.summary)
                        .font(.caption)
                        .foregroundStyle(brand.textSecondary)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 6) {
                        Image(systemName: "arrow.up.forward")
                            .font(.caption2)
                        Text("\u{201C}\(signal.suggestedPrompt)\u{201D}")
                            .font(.caption2)
                            .italic()
                    }
                    .foregroundStyle(brand.textSecondary)
                    .padding(.top, 2)
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(tint.opacity(0.06), in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
            .overlay(
                RoundedRectangle(cornerRadius: brand.cardCornerRadius)
                    .stroke(tint.opacity(0.25), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(severityLabel) support signal: \(signal.title)")
        .accessibilityHint("Double-tap to ask the assistant about this")
    }

    private var tint: Color {
        switch signal.severity {
        case .urgent: return brand.danger
        case .attention: return brand.warning
        case .info: return brand.primary
        }
    }

    private var severityLabel: String {
        switch signal.severity {
        case .urgent: return "Needs attention"
        case .attention: return "Heads up"
        case .info: return "Suggestion"
        }
    }
}
