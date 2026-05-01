import SwiftUI

/// The contract every brand must satisfy. All UI colors, copy, and shape
/// primitives come from here — views must never hardcode `.red`, `.black`,
/// or literal strings for app-name / assistant-name. That keeps the generic
/// Telco skin one-swap-removed from any carrier (T-Mobile, AT&T, Bell,
/// Rogers, Vodafone, Verizon) that wants the same POC reskinned.
public protocol BrandTheme: Sendable {
    var id: String { get }
    var displayName: String { get }
    var tagline: String { get }

    // Colors
    var primary: Color { get }
    var onPrimary: Color { get }
    var accent: Color { get }
    var surfaceBackground: Color { get }
    var surfaceElevated: Color { get }
    var border: Color { get }
    var textPrimary: Color { get }
    var textSecondary: Color { get }

    // Semantic status colors — shared shape but themeable
    var success: Color { get }
    var warning: Color { get }
    var danger: Color { get }
    var info: Color { get }

    // Typography
    var titleFont: Font { get }
    var bodyFont: Font { get }
    var monoFont: Font { get }

    // Shape
    var bubbleCornerRadius: CGFloat { get }
    var cardCornerRadius: CGFloat { get }

    // Copy
    var appName: String { get }            // "Telco Triage"
    var appSubtitle: String { get }        // "by Liquid AI"
    var assistantName: String { get }      // "Telco Assistant" / "Liquid Support"
    var welcomeGreeting: @Sendable (String) -> String { get } // personalized greeting
    var chatPlaceholder: String { get }     // Input bar hint text
    var deepLinkScheme: String { get }     // "telco", etc.

    // Asset hooks (SF Symbols by default — brands can override with image names)
    var wordmarkSystemImage: String { get }
}

public extension BrandTheme {
    /// Colored overlay used when we need a subtle brand-tinted background.
    func primaryTint(_ opacity: Double) -> Color {
        primary.opacity(opacity)
    }
}

// MARK: - Environment wiring

private struct BrandThemeKey: EnvironmentKey {
    static let defaultValue: any BrandTheme = TelcoTriageTheme()
}

public extension EnvironmentValues {
    var brand: any BrandTheme {
        get { self[BrandThemeKey.self] }
        set { self[BrandThemeKey.self] = newValue }
    }
}

public extension View {
    func brand(_ theme: any BrandTheme) -> some View {
        environment(\.brand, theme)
            .tint(theme.primary)
    }
}
