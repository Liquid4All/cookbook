import Foundation

extension ChatViewModel {
    /// The iOS Simulator cannot safely use Metal offload for this GGUF
    /// stack, so it runs llama.cpp on CPU. A base-model grounded-QA decode
    /// that runs to the 256-token cap takes ~5s there; real iPhones keep
    /// the generative path. Set VERIZON_DISABLE_SIMULATOR_FAST_RAG=1 when
    /// validating simulator generative behavior directly.
    nonisolated static var shouldUseSimulatorFastGroundedQA: Bool {
        #if targetEnvironment(simulator)
        return ProcessInfo.processInfo.environment["VERIZON_DISABLE_SIMULATOR_FAST_RAG"] != "1"
        #else
        return false
        #endif
    }

    /// Customer-readable KB answer used by the simulator fast path after
    /// LFM routing + KB selection have already run. This keeps simulator
    /// demos responsive while preserving the same local grounding source.
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
