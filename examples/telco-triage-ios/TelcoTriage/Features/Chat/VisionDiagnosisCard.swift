import SwiftUI

/// Renders a `VisionDiagnosis` below the assistant's vision reply.
/// Shows the headline, explanation, and — when a proposed tool is
/// attached at confidence ≥ 0.7 — a one-tap action button that
/// executes through the normal tool-confirmation flow.
struct VisionDiagnosisCard: View {
    let diagnosis: VisionDiagnosis
    let onExecuteTool: ((String) -> Void)?

    @Environment(\.brand) private var brand

    // Don't push a tool action when confidence is low or the visual
    // pack isn't installed — the mock preview would be misleading.
    private var canProposeTool: Bool {
        diagnosis.proposedToolID != nil
            && diagnosis.confidence >= 0.7
            && diagnosis.usedPack
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 16))
                    .foregroundStyle(tint)
                    .frame(width: 30, height: 30)
                    .background(tint.opacity(0.12), in: Circle())
                VStack(alignment: .leading, spacing: 2) {
                    Text(kindLabel)
                        .font(.caption2)
                        .fontWeight(.bold)
                        .foregroundStyle(tint)
                        .textCase(.uppercase)
                    Text(diagnosis.headline)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundStyle(brand.textPrimary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                if !diagnosis.usedPack {
                    Text("preview")
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(brand.surfaceElevated, in: Capsule())
                        .foregroundStyle(brand.textSecondary)
                }
            }

            Text(diagnosis.explanation)
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 6) {
                HStack(spacing: 4) {
                    Image(systemName: "gauge.with.dots.needle.67percent")
                        .font(.caption2)
                    Text(String(format: "conf %.2f", diagnosis.confidence))
                        .font(.caption2)
                        .monospacedDigit()
                }
                .foregroundStyle(brand.textSecondary)
                HStack(spacing: 4) {
                    Image(systemName: "clock")
                        .font(.caption2)
                    Text("\(diagnosis.latencyMS) ms")
                        .font(.caption2)
                        .monospacedDigit()
                }
                .foregroundStyle(brand.textSecondary)
                Spacer()
            }

            if canProposeTool, let toolID = diagnosis.proposedToolID, let onExecuteTool {
                Button { onExecuteTool(toolID) } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "play.fill")
                            .font(.caption)
                        Text("Try: \(proposedToolLabel(toolID))")
                            .font(.caption)
                            .fontWeight(.semibold)
                        Spacer()
                        Image(systemName: "arrow.right")
                            .font(.caption2)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .background(brand.primary, in: RoundedRectangle(cornerRadius: 10))
                    .foregroundStyle(brand.onPrimary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Run proposed tool: \(proposedToolLabel(toolID))")
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(brand.surfaceElevated.opacity(0.6), in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: brand.cardCornerRadius)
                .stroke(tint.opacity(0.2), lineWidth: 1)
        )
    }

    private var icon: String {
        switch diagnosis.kind {
        case .routerLights: return "wifi.router"
        case .billScreenshot: return "doc.text"
        case .errorScreen: return "exclamationmark.triangle"
        case .unknown: return "questionmark.circle"
        }
    }

    private var kindLabel: String {
        switch diagnosis.kind {
        case .routerLights: return "Router lights"
        case .billScreenshot: return "Bill screenshot"
        case .errorScreen: return "Error screen"
        case .unknown: return "Image received"
        }
    }

    private var tint: Color {
        switch diagnosis.severity {
        case .critical: return brand.danger
        case .degraded: return brand.warning
        case .normal: return brand.success
        case .unclear: return brand.textSecondary
        }
    }

    private func proposedToolLabel(_ toolID: String) -> String {
        ToolIntent(toolID: toolID)?.displayName ?? toolID
    }
}
