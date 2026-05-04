import SwiftUI

/// Liquid AI default theme. Monochrome, minimal, technical. Used when the
/// Settings toggle flips away from the telco skin, and for any future carrier
/// that hasn't been branded yet.
public struct LiquidTheme: BrandTheme {
    public let id = "liquid"
    public let displayName = "Liquid AI"
    public let tagline = "On-Device Reference"

    public let primary = Color.black
    public let onPrimary = Color.white
    public let accent = Color(red: 0.55, green: 0.35, blue: 0.85)   // aurora-ish
    public let surfaceBackground = Color(.systemBackground)
    public let surfaceElevated = Color(.secondarySystemBackground)
    public let border = Color.black.opacity(0.1)
    public let textPrimary = Color.primary
    public let textSecondary = Color.secondary

    public let success = Color(red: 0.12, green: 0.7, blue: 0.4)
    public let warning = Color(red: 0.95, green: 0.65, blue: 0.0)
    public let danger = Color(red: 0.85, green: 0.2, blue: 0.2)
    public let info = Color(red: 0.2, green: 0.5, blue: 0.9)

    public let titleFont = Font.system(.title3, design: .monospaced).weight(.semibold)
    public let bodyFont = Font.system(.body, design: .monospaced)
    public let monoFont = Font.system(.caption, design: .monospaced)

    public let bubbleCornerRadius: CGFloat = 16
    public let cardCornerRadius: CGFloat = 12

    public let appName = "Liquid Support"
    public let appSubtitle = "On-Device Reference"
    public let assistantName = "Liquid Assistant"
    public let chatPlaceholder = "Ask anything\u{2026}"
    public let deepLinkScheme = "liquid"
    public let wordmarkSystemImage = "drop.fill"

    public var welcomeGreeting: @Sendable (String) -> String {
        { name in
            "Hi — ask anything, I run on-device. Retrieval-grounded, no cloud call for local queries."
        }
    }

    public init() {}
}
