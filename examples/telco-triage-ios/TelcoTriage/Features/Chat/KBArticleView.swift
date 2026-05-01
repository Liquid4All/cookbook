import SwiftUI

/// Full-article sheet shown when the customer taps the "Read full
/// article" chip under a grounded-QA response. Renders the complete
/// `KBEntry.answer` markdown (steps, links, etc.) rather than the
/// compressed LFM-composed summary in the chat bubble.
///
/// Deliberately in-app (not a `telco://` `UIApplication.shared.open`
/// call): the demo phone has no carrier app installed, and the
/// point is to show the LFM did a good job summarizing a real KB
/// article that lives on-device — not to launch an external surface.
struct KBArticleView: View {
    let entry: KBEntry
    @Environment(\.dismiss) private var dismiss
    @Environment(\.brand) private var brand

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    header
                    Text(LocalizedStringKey(entry.answer))
                        .font(brand.bodyFont)
                        .foregroundStyle(brand.textPrimary)
                        .fixedSize(horizontal: false, vertical: true)
                    if !entry.deepLinks.isEmpty {
                        deepLinksSection
                    }
                    provenance
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(brand.surfaceBackground.ignoresSafeArea())
            .navigationTitle(entry.topic)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(entry.category.uppercased())
                .font(.caption2).fontWeight(.semibold)
                .foregroundStyle(brand.textSecondary)
            Text(entry.topic)
                .font(.title3).fontWeight(.semibold)
                .foregroundStyle(brand.textPrimary)
        }
    }

    private var deepLinksSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Where in the app")
                .font(.caption2).fontWeight(.semibold).textCase(.uppercase)
                .foregroundStyle(brand.textSecondary)
            ForEach(entry.deepLinks, id: \.url) { link in
                HStack(spacing: 6) {
                    Image(systemName: "arrow.up.right.square")
                    Text(link.label)
                        .font(.callout)
                    Spacer()
                    Text(link.url)
                        .font(brand.monoFont)
                        .foregroundStyle(brand.textSecondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: 10))
                .foregroundStyle(brand.textPrimary)
            }
        }
    }

    private var provenance: some View {
        HStack(spacing: 6) {
            Image(systemName: "doc.text")
            Text("KB entry: \(entry.id)")
                .font(brand.monoFont)
        }
        .foregroundStyle(brand.textSecondary)
        .padding(.top, 8)
    }
}
