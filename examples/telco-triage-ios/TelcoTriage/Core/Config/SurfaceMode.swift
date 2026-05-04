import Foundation

/// Which audience the current view is serving.
///
/// The app exposes two deliberately separate surfaces:
/// - `.customer` — what a home internet customer actually uses (chat, household).
/// - `.operator_` — what a telco operator buys (ROI, architecture, packs).
///
/// The demo narrative lives in the pivot between the two: a sales engineer
/// runs a live support flow on the customer surface, then flips to the
/// operator surface to show what happened under the hood economically and
/// technically. Keeping them as separate surfaces (not mixed tabs) prevents
/// slideware from leaking into customer-facing UX, and prevents the pitch
/// from being hidden inside a support app.
///
/// Not persisted across launches — every fresh launch starts on the
/// customer surface (the default path a subscriber would take). If we
/// ever need stickiness during long demos, wire an `@AppStorage`
/// binding in `RootView`; the stable `rawValue` is compatible with that.
public enum SurfaceMode: String, CaseIterable, Identifiable, Sendable {
    case customer
    case operator_ = "operator"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .customer: return "Customer"
        case .operator_: return "Operator"
        }
    }

    public var longDisplayName: String {
        switch self {
        case .customer: return "Customer view"
        case .operator_: return "Operator view"
        }
    }

    public var accessibilityHint: String {
        switch self {
        case .customer: return "What a home internet customer sees in the app"
        case .operator_: return "What an operator buys — ROI, architecture, intelligence packs"
        }
    }
}
