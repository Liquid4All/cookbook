import SwiftUI

/// Liquid AI-styled mark used in the navigation header.
///
/// Renders a small flowing droplet glyph with an aurora gradient
/// (purple → blue → cyan) — the visual cue that this app is built
/// on Liquid AI's on-device foundation. Sized for a `subheadline`
/// title bar; scales with Dynamic Type via `relativeTo:`.
///
/// Pairs with a lockup view (`LiquidAITitleLockup`) that places this
/// mark next to the app name + subtitle. Both are theme-agnostic and
/// safe to drop into any branded surface.
public struct LiquidAIMark: View {
    public enum Size: CGFloat {
        case small = 18
        case medium = 24
        case large = 32
    }

    let size: Size

    public init(size: Size = .small) {
        self.size = size
    }

    public var body: some View {
        ZStack {
            // Soft halo behind the glyph — gives it the "liquid"
            // feel without painting the whole nav bar.
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            Color(red: 0.35, green: 0.55, blue: 1.00).opacity(0.35),
                            .clear,
                        ],
                        center: .center,
                        startRadius: 0,
                        endRadius: size.rawValue
                    )
                )
                .frame(width: size.rawValue * 1.6, height: size.rawValue * 1.6)
                .blur(radius: 2)

            // The droplet glyph itself, painted with the Liquid
            // aurora — purple to blue to cyan, rotated slightly so
            // it reads as flowing rather than top-lit.
            Image(systemName: "drop.fill")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: size.rawValue, height: size.rawValue)
                .foregroundStyle(Self.auroraGradient)
                .rotationEffect(.degrees(-12))
                .shadow(color: Color(red: 0.35, green: 0.55, blue: 1.0).opacity(0.5), radius: 3)
        }
        .accessibilityHidden(true)
    }

    /// Aurora gradient — Liquid AI's signature flowing color tone.
    /// Tuned for legibility on both light and dark navigation bars
    /// (saturation pulled in slightly so it doesn't oversaturate
    /// against monochrome text).
    static let auroraGradient = LinearGradient(
        colors: [
            Color(red: 0.55, green: 0.35, blue: 0.95),  // violet
            Color(red: 0.30, green: 0.55, blue: 1.00),  // azure
            Color(red: 0.20, green: 0.80, blue: 0.95),  // cyan
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
}

/// Title lockup for the chat navigation bar: Liquid mark + app name
/// + subtitle. The "Liquid AI" tokens in the subtitle (when present)
/// receive the aurora tint so the brand reads even at glance.
public struct LiquidAITitleLockup: View {
    let appName: String
    let subtitle: String
    /// True when the engineering-mode subtitle is in effect; we tint
    /// it with `brand.accent` instead of the aurora gradient to
    /// keep the engineering surface visually distinct.
    let isEngineeringSubtitle: Bool

    @Environment(\.brand) private var brand

    public init(appName: String, subtitle: String, isEngineeringSubtitle: Bool) {
        self.appName = appName
        self.subtitle = subtitle
        self.isEngineeringSubtitle = isEngineeringSubtitle
    }

    public var body: some View {
        HStack(spacing: 8) {
            LiquidAIMark(size: .small)
            VStack(spacing: 0) {
                Text(appName)
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .foregroundStyle(brand.textPrimary)

                subtitleView
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(appName), \(subtitle)")
    }

    @ViewBuilder
    private var subtitleView: some View {
        if isEngineeringSubtitle {
            Text(subtitle)
                .font(.caption2)
                .foregroundStyle(brand.accent)
        } else {
            // Aurora-tint the "Liquid AI" tokens; everything else
            // stays neutral so the mark stands out on its own.
            highlightedSubtitle
                .font(.caption2)
        }
    }

    /// Rebuild the subtitle as a Text composition where the literal
    /// "Liquid AI" gets the aurora foreground style. Falls back to a
    /// plain styled Text when the substring isn't present.
    private var highlightedSubtitle: Text {
        let token = "Liquid AI"
        guard let range = subtitle.range(of: token) else {
            return Text(subtitle).foregroundStyle(brand.textSecondary)
        }
        let before = String(subtitle[..<range.lowerBound])
        let after = String(subtitle[range.upperBound...])
        return Text(before).foregroundStyle(brand.textSecondary)
            + Text(token).foregroundStyle(LiquidAIMark.auroraGradient).fontWeight(.semibold)
            + Text(after).foregroundStyle(brand.textSecondary)
    }
}
