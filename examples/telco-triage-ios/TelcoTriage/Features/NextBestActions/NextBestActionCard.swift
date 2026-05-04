import SwiftUI

/// Renders a single NBA. Used on the Plan tab's "For You" section and
/// attached inline to chat replies when the intent matches. Accept /
/// decline route back through the engine so the Savings dashboard can
/// show "revenue surfaced" vs "revenue accepted".
struct NextBestActionCard: View {
    let action: any NextBestAction
    let onAccept: () -> Void
    let onDecline: () -> Void

    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Text(action.body)
                .font(.callout)
                .foregroundStyle(brand.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
            buttons
        }
        .padding(16)
        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: brand.cardCornerRadius)
                .stroke(tintForCategory.opacity(0.35), lineWidth: 1)
        )
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: action.icon)
                .font(.system(size: 22))
                .frame(width: 40, height: 40)
                .background(tintForCategory.opacity(0.15), in: RoundedRectangle(cornerRadius: 10))
                .foregroundStyle(tintForCategory)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    // Small AI-powered marker so the customer knows this
                    // card was picked by the on-device LFM, not a
                    // hardcoded banner.
                    HStack(spacing: 3) {
                        Image(systemName: "drop.fill")
                            .font(.system(size: 9))
                        Text("AI")
                            .font(.caption2).fontWeight(.bold)
                    }
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(brand.accent.opacity(0.12), in: Capsule())
                    .foregroundStyle(brand.accent)

                    Text(action.category.displayName.uppercased())
                        .font(.caption2).fontWeight(.bold)
                        .foregroundStyle(tintForCategory)
                    if let tag = action.impactTag {
                        Text(tag)
                            .font(.caption2).fontWeight(.semibold)
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(tintForCategory.opacity(0.12), in: Capsule())
                            .foregroundStyle(tintForCategory)
                    }
                }
                Text(action.headline)
                    .font(.headline)
                    .foregroundStyle(brand.textPrimary)
            }
            Spacer()
        }
    }

    private var buttons: some View {
        HStack(spacing: 10) {
            Button(action: onDecline) {
                Text(action.declineLabel)
                    .font(.subheadline).fontWeight(.medium)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .foregroundStyle(brand.textSecondary)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(brand.border)
                    )
            }
            Button(action: onAccept) {
                Text(action.acceptLabel)
                    .font(.subheadline).fontWeight(.semibold)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .foregroundStyle(brand.onPrimary)
                    .background(tintForCategory, in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    private var tintForCategory: Color {
        switch action.category {
        case .upsell: return brand.primary
        case .retention: return brand.success
        case .planOptimize: return brand.info
        case .boltOn: return brand.primary
        case .proactiveSupport: return brand.warning
        }
    }
}
