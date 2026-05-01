import SwiftUI

/// Customer-mode tool confirmation. Presented as a half-sheet when the
/// LFM selects a tool action. Clean, friendly, no engineering jargon —
/// just "here's what I'd like to do, OK?"
///
/// Replaces the inline `ToolDecisionCard` in customer mode. Engineering
/// mode still shows the full inline card with confidence scores and
/// extracted argument labels.
struct ToolConfirmationSheet: View {
    let decision: ToolDecision
    let onConfirm: () -> Void
    let onDecline: () -> Void

    @Environment(\.brand) private var brand

    var body: some View {
        VStack(spacing: 24) {
            // Icon + title
            VStack(spacing: 12) {
                Image(systemName: decision.icon)
                    .font(.system(size: 36))
                    .foregroundStyle(brand.primary)
                    .frame(width: 64, height: 64)
                    .background(brand.primary.opacity(0.1), in: Circle())

                Text(decision.displayName)
                    .font(.title3)
                    .fontWeight(.semibold)
                    .foregroundStyle(brand.textPrimary)
                    .multilineTextAlignment(.center)
            }

            // Plain-English summary of what will happen
            if let reasoning = decision.reasoning {
                Text(reasoning)
                    .font(.body)
                    .foregroundStyle(brand.textSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 16)
            }

            // Arguments as a simple key-value list (no "Extracted arguments" header)
            if !decision.arguments.isEmpty {
                VStack(spacing: 8) {
                    ForEach(decision.arguments) { arg in
                        HStack {
                            Text(arg.label)
                                .font(.subheadline)
                                .foregroundStyle(brand.textSecondary)
                            Spacer()
                            Text(arg.value)
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundStyle(brand.textPrimary)
                        }
                    }
                }
                .padding(16)
                .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: 12))
            }

            // Destructive warning
            if decision.isDestructive {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(brand.warning)
                    Text("This action cannot be undone")
                        .font(.caption)
                        .foregroundStyle(brand.warning)
                }
            }

            // Action buttons
            VStack(spacing: 10) {
                Button(action: onConfirm) {
                    Text(decision.isDestructive ? "Confirm" : "Sounds good")
                        .font(.body)
                        .fontWeight(.semibold)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(brand.primary, in: RoundedRectangle(cornerRadius: 12))
                        .foregroundStyle(brand.onPrimary)
                }
                .buttonStyle(.plain)

                Button(action: onDecline) {
                    Text("Not now")
                        .font(.body)
                        .fontWeight(.medium)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .foregroundStyle(brand.textSecondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(24)
        .padding(.top, 8)
    }
}
