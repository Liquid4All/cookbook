import SwiftUI

/// Read-only diff between the raw user query and the redacted form,
/// surfacing the PII spans the on-device scanner caught. No cloud
/// approval gate — this sheet is pure "show me what the scanner saw".
struct PIIInspectSheet: View {
    let state: PrivacyShieldState
    @Environment(\.dismiss) private var dismiss
    @Environment(\.brand) private var brand

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    header
                    rawSection
                    redactedSection
                    if !state.spans.isEmpty {
                        spansSection
                    }
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(brand.surfaceBackground.ignoresSafeArea())
            .navigationTitle("On-device PII scan")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Nothing from this query left the device.")
                .font(.headline)
                .foregroundStyle(brand.textPrimary)
            Text("The PII scanner runs entirely on-device before any further processing. It found the spans below.")
                .font(.subheadline)
                .foregroundStyle(brand.textSecondary)
        }
    }

    private var rawSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Raw query")
            Text(state.original)
                .font(brand.bodyFont)
                .foregroundStyle(brand.textPrimary)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: 12))
        }
    }

    private var redactedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Redacted view")
            Text(state.sanitized)
                .font(brand.bodyFont)
                .foregroundStyle(brand.textPrimary)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: 12))
        }
    }

    private var spansSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Detections")
            // Each detection renders the kind-aware masked form (e.g.
            // "***-**-6789" for an SSN). The full raw value is still
            // visible in the "Raw query" section above — that's where
            // the demo shows the input the scanner saw. Masking here
            // prevents a screenshot of this sheet from leaking the
            // plaintext PII.
            ForEach(Array(state.spans.enumerated()), id: \.offset) { _, span in
                HStack(spacing: 8) {
                    Text(span.kind.rawValue.uppercased())
                        .font(.caption2).fontWeight(.bold)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(brand.warning.opacity(0.15), in: Capsule())
                        .foregroundStyle(brand.warning)
                    Text(PIIMasker.masked(span))
                        .font(brand.monoFont)
                        .foregroundStyle(brand.textPrimary)
                    Spacer()
                }
            }
        }
    }

    private func sectionLabel(_ text: String) -> some View {
        Text(text)
            .font(.caption2).fontWeight(.semibold)
            .textCase(.uppercase)
            .foregroundStyle(brand.textSecondary)
    }
}
