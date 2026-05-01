import SwiftUI

/// A Siri-inspired status pill shown during inference in customer mode.
/// Gives users just enough feedback about what the on-device AI is doing
/// without exposing engineering internals. Replaces `ProcessingRow` in
/// customer mode.
///
/// States map to the LFM pipeline stages:
///  - "Understanding..." → ChatModeRouter is classifying
///  - "Searching..." → KBExtractor is retrieving
///  - "Preparing action..." → ToolSelector is selecting
///  - "Composing..." → LFMChatProvider is generating
struct RoutingStatusPill: View {
    let stage: RoutingStage

    @Environment(\.brand) private var brand
    @State private var isPulsing = false

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(brand.primary)
                .frame(width: 8, height: 8)
                .scaleEffect(isPulsing ? 1.3 : 1.0)
                .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isPulsing)
            Text(stage.displayText)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundStyle(brand.textSecondary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(brand.surfaceElevated, in: Capsule())
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .onAppear { isPulsing = true }
    }
}

/// Pipeline stage visible to customer mode. The ViewModel publishes
/// this as processing progresses.
public enum RoutingStage: String, Sendable {
    case understanding
    case searching
    case preparingAction
    case composing

    var displayText: String {
        switch self {
        case .understanding:   return "Understanding..."
        case .searching:       return "Searching knowledge base..."
        case .preparingAction: return "Preparing action..."
        case .composing:       return "Composing response..."
        }
    }
}
