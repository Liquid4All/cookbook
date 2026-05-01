import SwiftUI

/// Generic telco triage theme. Carrier-specific skins can still register
/// their own brand, but the default demo now mirrors Edge Banking: a
/// neutral solution pattern first, customer logo second.
public struct TelcoTriageTheme: BrandTheme {
    public let id = "telco-triage"
    public let displayName = "Telco Triage"
    public let tagline = "by Liquid AI"

    public let primary = Color(red: 0.08, green: 0.08, blue: 0.09)
    public let onPrimary = Color.white
    public let accent = Color(red: 0.24, green: 0.45, blue: 0.95)
    public let surfaceBackground = Color(.systemBackground)
    public let surfaceElevated = Color(.secondarySystemBackground)
    public let border = Color.black.opacity(0.1)
    public let textPrimary = Color.primary
    public let textSecondary = Color.secondary

    public let success = Color(red: 0.12, green: 0.7, blue: 0.4)
    public let warning = Color(red: 0.95, green: 0.65, blue: 0.0)
    public let danger = Color(red: 0.85, green: 0.2, blue: 0.2)
    public let info = Color(red: 0.2, green: 0.5, blue: 0.9)

    public let titleFont = Font.system(.title3, design: .default).weight(.bold)
    public let bodyFont = Font.system(.body)
    public let monoFont = Font.system(.caption, design: .monospaced)

    public let bubbleCornerRadius: CGFloat = 18
    public let cardCornerRadius: CGFloat = 14

    public let appName = "Telco Triage"
    public let appSubtitle = "by Liquid AI"
    public let assistantName = "Telco Assistant"
    public let chatPlaceholder = "Ask about Wi-Fi, router, bill..."
    public let deepLinkScheme = "telco"
    public let wordmarkSystemImage = "antenna.radiowaves.left.and.right"

    public var welcomeGreeting: @Sendable (String) -> String {
        { name in
            if name.isEmpty {
                return "Hi - I'm your telco support assistant running on-device with Liquid AI. I can answer home internet questions, triage Wi-Fi issues, prepare safe tool actions, and hand off complex requests to cloud only with approval."
            }
            return "Hi \(name) - I'm your telco support assistant running on-device with Liquid AI. I can answer home internet questions, triage Wi-Fi issues, prepare safe tool actions, and hand off complex requests to cloud only with approval."
        }
    }

    public init() {}
}

public typealias VerizonTheme = TelcoTriageTheme
