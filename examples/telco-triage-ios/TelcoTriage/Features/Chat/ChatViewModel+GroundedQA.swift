import Foundation

extension ChatViewModel {
    /// Keep customer-visible RAG on the low-latency path by default.
    ///
    /// The model stack still does the on-device work that matters for the
    /// demo: classify the request, arbitrate the support lane, select local
    /// tools, and choose whether the request can be answered from device
    /// knowledge. The final KB wording is rendered from the retrieved local
    /// article so the customer path does not pay autoregressive decode cost.
    ///
    /// Set TELCO_TRIAGE_ENABLE_GENERATIVE_GROUNDED_QA=1 when deliberately
    /// testing freeform local answer generation; that is an experiment path,
    /// not the latency-critical demo default.
    nonisolated static var shouldUseFastGroundedQA: Bool {
        ProcessInfo.processInfo.environment["TELCO_TRIAGE_ENABLE_GENERATIVE_GROUNDED_QA"] != "1"
    }

    /// Customer-readable KB answer used by the fast path after LFM routing
    /// + KB selection have already run. This keeps demos responsive while
    /// preserving the same local grounding source.
    static func compactGroundedAnswer(_ answer: String, maxDetailLines: Int = 4) -> String {
        let blocks = answer
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard let firstBlock = blocks.first else { return "" }

        let intro = ensureTerminalPunctuation(cleanKBLine(firstBlock))
        let detailLines = blocks
            .dropFirst()
            .flatMap { $0.components(separatedBy: .newlines) }
            .map(cleanKBLine)
            .filter { !$0.isEmpty }
            .prefix(maxDetailLines)
            .map(ensureTerminalPunctuation)

        guard !detailLines.isEmpty else {
            return firstParagraph(of: answer)
        }

        return ([intro] + Array(detailLines)).joined(separator: "\n")
    }

    private static func cleanKBLine(_ raw: String) -> String {
        var line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        line = line.replacingOccurrences(
            of: #"^\*\*Step\s+\d+:\*\*\s*"#,
            with: "",
            options: .regularExpression
        )
        line = line.replacingOccurrences(
            of: #"^[-*]\s+"#,
            with: "",
            options: .regularExpression
        )
        line = line.replacingOccurrences(of: "**", with: "")
        return line.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func ensureTerminalPunctuation(_ text: String) -> String {
        guard let last = text.last else { return text }
        if ".!?".contains(last) { return text }
        return "\(text)."
    }
}
