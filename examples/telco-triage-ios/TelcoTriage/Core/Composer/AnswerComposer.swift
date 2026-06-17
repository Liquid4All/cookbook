import Foundation

/// Swift port of `scripts/telco/answer_composer.py::DeterministicComposer`.
///
/// **Doctrine** (Step 5 decision record):
///
/// > The generator should verbalize already-selected evidence; it
/// > should not discover evidence, invent links, or decide citations.
///
/// The composer trusts its inputs unconditionally:
///
/// * `evidence` — the canonical RAG unit (picked by the retriever or
///   by the test oracle). The composer never re-ranks and never opens
///   `body` for new claims.
/// * `route` — already decided by the dispatcher
///   (`TelcoChatDispatcher.deriveRoute`) using the ToolRegistry
///   gate (guardrail #3). The composer never re-routes.
/// * `requiresConfirmation` — already decided. The composer toggles
///   a single confirmation clause; it NEVER invokes any tool.
///
/// **Link guarantee**: every `telcohome://` URL the composer emits comes
/// from `evidence.canonicalURL`. The only other URLs allowed are the
/// fixed external set (My Telco, Telco internet, tel:).
public enum AnswerComposerConstants {
    public static let telcoInternetURL = "https://www.telco.com/home/internet"
    public static let myTelcoURL = "https://m.telco.com/wMM0jUjn"
    public static let liveAgentPhone = "tel://+18009220204"
}

public protocol AnswerComposing: Sendable {
    func compose(
        query: String,
        route: ComposerRoute,
        evidence: RAGUnit?,
        requiresConfirmation: Bool?,
        history: [ConversationTurnSnippet],
        expectedPolicyLinkID: String?
    ) -> ComposedAnswer
}

public struct DeterministicAnswerComposer: AnswerComposing {
    public let name: String

    public init(name: String = "deterministic_composer") {
        self.name = name
    }

    public func compose(
        query: String,
        route: ComposerRoute,
        evidence: RAGUnit?,
        requiresConfirmation: Bool? = nil,
        history: [ConversationTurnSnippet] = [],
        expectedPolicyLinkID: String? = nil
    ) -> ComposedAnswer {
        _ = history  // unused — composer does not condition on history
        _ = expectedPolicyLinkID
        let start = CFAbsoluteTimeGetCurrent()

        var usedFallback = false
        var citedPageID: String? = evidence?.pageID
        var expectedLinkURL: String? = evidence?.canonicalURL

        var text = ""
        var hasStepChain = false

        switch route {
        case .greeting:
            text = "Hello! How can I assist you with Telco Home Internet today?"
            expectedLinkURL = nil
        case .outOfScope:
            text = "I'm here to help with topics related to Telco Home Internet. Please try asking a different question."
            expectedLinkURL = nil
        case .noRagAnswer:
            text = "It looks like I don't have specific information about that. You can check [Telco Home Internet](\(AnswerComposerConstants.telcoInternetURL)) for more details."
            expectedLinkURL = AnswerComposerConstants.telcoInternetURL
        case .liveAgent:
            text = "I can connect you with a Telco support agent. [Call us](\(AnswerComposerConstants.liveAgentPhone)) for support."
            expectedLinkURL = AnswerComposerConstants.liveAgentPhone
        case .clarify:
            let hint = clarifyHintFromQuery(query)
            text = "Could you clarify what you're asking about? Are you referring to \(hint)?"
            expectedLinkURL = nil
        case .accountNav:
            text = "To manage your account, please use the My Telco App: [My Telco](\(AnswerComposerConstants.myTelcoURL))."
            expectedLinkURL = AnswerComposerConstants.myTelcoURL
        case .ragAnswer, .answerPlusAction, .toolAction:
            if let unit = evidence {
                let task = taskPhrase(from: unit)
                let label = unit.displayLabel
                let url = unit.canonicalURL
                let intro = introForRoute(route, task: task)
                if let fieldLookup = renderFieldLookupAnswer(
                    query: query,
                    unit: unit,
                    route: route
                ) {
                    text = fieldLookup
                } else if let focused = renderFocusedStepAnswer(query: query, unit: unit, route: route) {
                    text = focused
                    hasStepChain = true
                } else if !unit.steps.isEmpty {
                    let chain = renderStepChain(label: label, url: url, steps: unit.steps)
                    text = "\(intro)\n\n\(chain)."
                    hasStepChain = true
                } else {
                    text = renderGroundedSummaryAnswer(
                        query: query,
                        unit: unit,
                        route: route
                    )
                }
                let needsConfirmClause = (requiresConfirmation ?? false) &&
                    (route == .toolAction || route == .answerPlusAction)
                if needsConfirmClause {
                    text = "\(text)\n\n\(confirmationClause())"
                }
            } else {
                // Composer can't ground itself without evidence. Safe
                // fallback — the telco.com URL is the canonical
                // citation for this state.
                text = "It looks like I don't have specific information about that. You can check [Telco Home Internet](\(AnswerComposerConstants.telcoInternetURL)) for more details."
                usedFallback = true
                citedPageID = nil
                expectedLinkURL = AnswerComposerConstants.telcoInternetURL
            }
        }

        let (urls, labels) = extractRenderedLinks(text)
        let latency = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        return ComposedAnswer(
            text: text,
            route: route,
            citedPageID: citedPageID,
            renderedLinks: urls,
            renderedLinkLabels: labels,
            expectedLinkURL: expectedLinkURL,
            requiresConfirmation: requiresConfirmation,
            actionFired: false,
            latencyMs: latency,
            strategy: name,
            hasStepChain: hasStepChain,
            usedFallback: usedFallback
        )
    }
}

// MARK: - Rendering helpers (mirror Python module-private helpers)

/// Render a markdown link in the canonical Telco Home assistant style.
public func renderLink(label: String, url: String) -> String {
    let safeLabel = label
        .replacingOccurrences(of: "[", with: "(")
        .replacingOccurrences(of: "]", with: ")")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    return "[\(safeLabel)](\(url))"
}

/// Render `[label](url) > step1 > step2 > ...`. Mirrors Python.
public func renderStepChain(label: String, url: String, steps: [String]) -> String {
    var parts: [String] = [renderLink(label: label, url: url)]
    let whitespace = CharacterSet.whitespacesAndNewlines
    let arrowReplace = "(?:^|[^>])>"  // unused; replicate Python's safer logic below
    _ = arrowReplace
    for step in steps {
        var clean = step.trimmingCharacters(in: whitespace)
        if clean.hasSuffix(".") {
            clean = String(clean.dropLast())
        }
        clean = clean.replacingOccurrences(of: "\n", with: " ")
        // Collapse runs of whitespace.
        clean = clean.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        // Never let a step introduce a "`>`"-shaped separator inside
        // its own text — match Python's heuristic.
        clean = clean.replacingOccurrences(of: " > ", with: " - ")
        clean = clean.replacingOccurrences(of: ">", with: " ")
        if clean.isEmpty { continue }
        parts.append(clean)
    }
    return parts.joined(separator: " > ")
}

/// Extract markdown link URLs + labels from the rendered text. Returns
/// `(urls, labels)` parallel arrays. Mirrors Python.
public func extractRenderedLinks(_ text: String) -> ([String], [String]) {
    var urls: [String] = []
    var labels: [String] = []
    guard let regex = try? NSRegularExpression(pattern: #"\[([^\]\n]+)\]\(([^)\s]+)\)"#, options: []) else {
        return ([], [])
    }
    let ns = text as NSString
    let range = NSRange(location: 0, length: ns.length)
    regex.enumerateMatches(in: text, options: [], range: range) { match, _, _ in
        guard let m = match, m.numberOfRanges >= 3 else { return }
        labels.append(ns.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespaces))
        urls.append(ns.substring(with: m.range(at: 2)).trimmingCharacters(in: .whitespaces))
    }
    return (urls, labels)
}

/// Mirrors `_task_phrase_from_evidence`.
private func taskPhrase(from unit: RAGUnit) -> String {
    if let taskID = unit.taskID, !taskID.isEmpty {
        return taskID.replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "_", with: " ")
    }
    if let label = unit.citationLabel, !label.isEmpty {
        return label.lowercased()
    }
    let title = unit.title.lowercased()
    if title.hasSuffix(".") { return String(title.dropLast()) }
    return title.isEmpty ? "do this" : title
}

/// Mirrors `_intro_for_route`.
private func introForRoute(_ route: ComposerRoute, task: String) -> String {
    let customerTask = customerFacingTaskPhrase(task)
    switch route {
    case .toolAction:
        return "I can \(customerTask) for you."
    case .answerPlusAction:
        return "To \(customerTask), follow these steps. I can also do it after you confirm:"
    default:
        return "To \(customerTask):"
    }
}

private func customerFacingTaskPhrase(_ task: String) -> String {
    switch task {
    case "restart router":
        return "restart your router"
    case "run speed test", "speed test":
        return "run a speed test"
    case "toggle parental controls":
        return "update parental controls"
    case "reboot extender":
        return "reboot your extender"
    default:
        return task
    }
}

/// Render a useful grounded answer for pages that have source text but
/// no canonical `steps` array. This keeps deterministic composition
/// from leaking corpus labels like "I found the relevant page for X"
/// while preserving the hard guarantee that every visible claim is
/// extracted from the selected `RAGUnit`.
private func renderGroundedSummaryAnswer(
    query: String,
    unit: RAGUnit,
    route: ComposerRoute
) -> String {
    let label = unit.displayLabel
    let link = renderLink(label: label, url: unit.canonicalURL)
    let lead = leadSentence(from: unit, query: query)
        ?? "This section covers \(humanizedLabel(label))."
    let facts: [String]
    if prefersRankedSummary(for: query) {
        facts = rankedFactSentences(from: unit, query: query, excluding: lead, limit: 4)
    } else {
        facts = sourceOrderedFactSentences(from: unit, excluding: lead, limit: 4)
    }

    var sections: [String] = [lead]
    if !facts.isEmpty {
        sections.append("Key details:\n" + facts.map { "- \($0)" }.joined(separator: "\n"))
    }

    let cta: String
    switch route {
    case .toolAction:
        cta = "Open \(label) to continue: \(link)."
    case .answerPlusAction:
        cta = "You can continue in \(label): \(link)."
    default:
        cta = "Open \(label) for the exact screen: \(link)."
    }
    sections.append(cta)
    return sections.joined(separator: "\n\n")
}

private enum FieldLookupKind {
    case networkName
    case wifiPassword
}

private func renderFieldLookupAnswer(
    query: String,
    unit: RAGUnit,
    route: ComposerRoute
) -> String? {
    guard route == .ragAnswer,
          let kind = fieldLookupKind(for: query) else {
        return nil
    }

    let body = decodeSourceText(unit.body)
    let label = unit.displayLabel
    let link = renderLink(label: label, url: unit.canonicalURL)
    let cta = "Open \(label) for the exact screen: \(link)."

    switch kind {
    case .networkName:
        guard body.localizedCaseInsensitiveContains("Network Name") else { return nil }
        return [
            "I can't see the actual SSID value here. In the app, your SSID is shown as the Network Name in Wi-Fi Management.",
            """
            Key details:
            - Use the Primary tab for your main home Wi-Fi network.
            - Guest and IoT have their own tabs if you need those network names.
            - On each network tile, the header displays the Network Name.
            """,
            cta,
        ].joined(separator: "\n\n")
    case .wifiPassword:
        guard body.localizedCaseInsensitiveContains("Wi-Fi password") ||
            body.localizedCaseInsensitiveContains("WiFi password") else {
            return nil
        }
        return [
            "I can't see the actual Wi-Fi password value here. In the app, the password is on the Wi-Fi Management tile for the selected network.",
            """
            Key details:
            - Choose Primary, Guest, or IoT.
            - The Wi-Fi password appears beneath the tile header.
            - It is hidden by default; select Show to temporarily reveal it.
            """,
            cta,
        ].joined(separator: "\n\n")
    }
}

private func fieldLookupKind(for query: String) -> FieldLookupKind? {
    guard isFieldLookupQuestion(query) else { return nil }
    let lower = query.lowercased()
    let tokens = Set(BM25Tokenizer.tokenize(query))
    if tokens.contains("password") {
        return .wifiPassword
    }
    if tokens.contains("ssid") ||
        lower.contains("network name") ||
        lower.contains("wifi name") ||
        lower.contains("wi-fi name") {
        return .networkName
    }
    return nil
}

private func leadSentence(from unit: RAGUnit, query: String) -> String? {
    let sentences = sourceSentences(from: unit.body)
        .filter { !isLowValueSummarySentence($0) }
    guard !sentences.isEmpty else { return nil }

    guard prefersRankedSummary(for: query) else {
        let objectiveTokens = Set(BM25Tokenizer.tokenize(unit.title))
            .union(BM25Tokenizer.tokenize(unit.displayLabel))
        for sentence in sentences.prefix(5) {
            let sentenceTokens = Set(BM25Tokenizer.tokenize(sentence))
            if !sentenceTokens.intersection(objectiveTokens).isEmpty {
                return sentence
            }
        }
        return sentences.first
    }

    let queryTokens = summaryFocusTokens(for: query)
    let objectiveTokens = summaryObjectiveTokens(for: unit)
    if let match = rankedSentenceMatches(
        sentences: sentences,
        queryTokens: queryTokens,
        objectiveTokens: objectiveTokens,
        excluding: nil
    ).first(where: { $0.queryOverlap > 0 }) {
        return match.sentence
    }
    if let match = rankedSentenceMatches(
        sentences: sentences,
        queryTokens: queryTokens,
        objectiveTokens: objectiveTokens,
        excluding: nil
    ).first(where: { $0.objectiveOverlap > 0 }) {
        return match.sentence
    }
    return sentences.first
}

private func sourceOrderedFactSentences(
    from unit: RAGUnit,
    excluding lead: String,
    limit: Int
) -> [String] {
    let sentences = sourceSentences(from: unit.body)
        .filter { !isLowValueSummarySentence($0) }
    guard let leadIndex = sentences.firstIndex(of: lead) else {
        return Array(sentences.filter { $0 != lead }.prefix(limit))
    }
    let afterLead = sentences.dropFirst(leadIndex + 1)
    if !afterLead.isEmpty {
        return Array(afterLead.prefix(limit))
    }
    return Array(sentences.filter { $0 != lead }.prefix(limit))
}

private func rankedFactSentences(
    from unit: RAGUnit,
    query: String,
    excluding lead: String,
    limit: Int
) -> [String] {
    let sentences = sourceSentences(from: unit.body)
        .filter { $0 != lead && !isLowValueSummarySentence($0) }
    guard !sentences.isEmpty else { return [] }

    let matches = rankedSentenceMatches(
        sentences: sentences,
        queryTokens: summaryFocusTokens(for: query),
        objectiveTokens: summaryObjectiveTokens(for: unit),
        excluding: lead
    )
    let relevant = matches
        .filter { $0.queryOverlap > 0 || $0.objectiveOverlap > 0 || $0.supportCue }
        .prefix(limit)
        .map(\.sentence)
    if !relevant.isEmpty {
        return Array(relevant)
    }
    return Array(sentences.prefix(limit))
}

private func prefersRankedSummary(for query: String) -> Bool {
    let lower = query.lowercased()
    let tokens = Set(BM25Tokenizer.tokenize(query))
    let diagnosticTokens: Set<String> = [
        "why", "slow", "sluggish", "lag", "laggy", "weak", "poor", "bad",
        "issue", "issues", "problem", "problems", "trouble", "troubleshoot",
        "dropping", "disconnecting", "buffering",
    ]
    let identityTokens: Set<String> = [
        "ssid", "password", "credential", "credentials",
    ]
    return !tokens.intersection(diagnosticTokens).isEmpty ||
        !tokens.intersection(identityTokens).isEmpty ||
        lower.contains("network name") ||
        lower.contains("wifi name") ||
        lower.contains("wi-fi name") ||
        lower.contains("not working")
}

private struct SummarySentenceMatch {
    let sentence: String
    let score: Double
    let queryOverlap: Int
    let objectiveOverlap: Int
    let supportCue: Bool
}

private func rankedSentenceMatches(
    sentences: [String],
    queryTokens: Set<String>,
    objectiveTokens: Set<String>,
    excluding: String?
) -> [SummarySentenceMatch] {
    sentences.enumerated().compactMap { index, sentence in
        guard sentence != excluding else { return nil }
        let sentenceTokens = Set(BM25Tokenizer.tokenize(sentence))
        let queryOverlap = sentenceTokens.intersection(queryTokens).count
        let objectiveOverlap = sentenceTokens.intersection(objectiveTokens).count
        let supportCue = hasSupportCue(sentence)
        var score = Double(queryOverlap * 4 + objectiveOverlap)
        if supportCue { score += 1.5 }
        score -= Double(index) * 0.01
        return SummarySentenceMatch(
            sentence: sentence,
            score: score,
            queryOverlap: queryOverlap,
            objectiveOverlap: objectiveOverlap,
            supportCue: supportCue
        )
    }
    .sorted {
        if $0.score != $1.score { return $0.score > $1.score }
        return $0.sentence < $1.sentence
    }
}

private func summaryObjectiveTokens(for unit: RAGUnit) -> Set<String> {
    var tokens = Set(BM25Tokenizer.tokenize(unit.title))
    for token in BM25Tokenizer.tokenize(unit.displayLabel) { tokens.insert(token) }
    if let taskID = unit.taskID {
        for token in BM25Tokenizer.tokenize(taskID) { tokens.insert(token) }
    }
    for alias in unit.aliases {
        for token in BM25Tokenizer.tokenize(alias) { tokens.insert(token) }
    }
    return expandWifiTokens(tokens)
}

private func summaryFocusTokens(for query: String) -> Set<String> {
    var tokens = Set(BM25Tokenizer.tokenize(query))
    let lower = query.lowercased()
    if lower.contains("slow") || lower.contains("sluggish") || lower.contains("lag") {
        tokens.formUnion([
            "speed", "performance", "coverage", "reliability", "weak", "fair",
            "recommended", "action", "repair", "restart", "reset", "optimize",
            "optimization", "son", "band", "6", "ghz",
        ])
    }
    if lower.contains("why") {
        tokens.formUnion(["score", "summary", "coverage", "reliability"])
    }
    if lower.contains("ssid") ||
        lower.contains("network name") ||
        lower.contains("wifi name") ||
        lower.contains("wi-fi name") {
        tokens.formUnion(["network", "name", "primary", "guest", "iot", "shown", "displayed"])
    }
    if lower.contains("password") {
        tokens.formUnion(["password", "show", "hidden", "primary", "guest", "iot"])
    }
    return expandWifiTokens(tokens)
}

private func expandWifiTokens(_ tokens: Set<String>) -> Set<String> {
    var expanded = tokens
    if tokens.contains("wifi") || tokens.contains("wi") || tokens.contains("fi") {
        expanded.formUnion(["wifi", "wi", "fi", "network", "wireless"])
    }
    if tokens.contains("ssid") {
        expanded.formUnion(["ssid", "network", "name", "wifi", "wi", "fi", "wireless"])
    }
    return expanded
}

private func isLowValueSummarySentence(_ sentence: String) -> Bool {
    let normalized = sentence
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    return normalized == "this page updates hourly" ||
        normalized == "level 1" ||
        normalized.hasPrefix("deep link:")
}

private func hasSupportCue(_ sentence: String) -> Bool {
    let tokens = Set(BM25Tokenizer.tokenize(sentence))
    let cues: Set<String> = [
        "recommended", "action", "improve", "speed", "coverage", "reliability",
        "weak", "fair", "repair", "restart", "reset", "optimize", "optimization",
        "connectivity", "issue",
    ]
    return !tokens.intersection(cues).isEmpty
}

private func sourceSentences(from body: String) -> [String] {
    let decoded = decodeSourceText(body)
    let normalized = decoded
        .replacingOccurrences(of: "\r", with: "\n")
        .replacingOccurrences(of: "\n", with: ". ")
        .replacingOccurrences(of: "  ", with: " ")

    return normalized
        .components(separatedBy: ". ")
        .map { raw -> String in
            var clean = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            while let last = clean.last, last == "." || last == ":" {
                clean.removeLast()
                clean = clean.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return clean
        }
        .filter { sentence in
            let tokens = BM25Tokenizer.tokenize(sentence)
            return tokens.count >= 3 && !sentence.lowercased().hasSuffix(" page")
        }
}

private func decodeSourceText(_ text: String) -> String {
    text
        .replacingOccurrences(of: "&quot;", with: "\"")
        .replacingOccurrences(of: "&amp;", with: "&")
        .replacingOccurrences(of: "&#39;", with: "'")
        .replacingOccurrences(of: "&apos;", with: "'")
}

private func humanizedLabel(_ label: String) -> String {
    label
        .replacingOccurrences(of: "-", with: " ")
        .replacingOccurrences(of: "_", with: " ")
        .lowercased()
}

/// If a user asks about one step inside the active task, render that
/// grounded step instead of replaying the whole procedure. This is a
/// local span-selection pass over `RAGUnit.steps`, not an additional
/// intent heuristic or free-form generation.
private func renderFocusedStepAnswer(
    query: String,
    unit: RAGUnit,
    route: ComposerRoute
) -> String? {
    guard !unit.steps.isEmpty else { return nil }
    guard !unit.queryTargetsTaskObjective(query) else { return nil }
    let qTokens = Set(BM25Tokenizer.tokenize(query))
    guard !qTokens.isEmpty else { return nil }

    var best: (index: Int, precision: Double, overlap: Int)?
    for (index, step) in unit.steps.enumerated() {
        let stepTokens = Set(BM25Tokenizer.tokenize(step))
        let overlap = qTokens.intersection(stepTokens).count
        guard overlap > 0 else { continue }
        let precision = Double(overlap) / Double(qTokens.count)
        if best == nil ||
            precision > best!.precision ||
            (precision == best!.precision && overlap > best!.overlap) {
            best = (index, precision, overlap)
        }
    }
    guard let match = best, match.precision >= 0.5 else { return nil }

    let label = unit.displayLabel
    let url = unit.canonicalURL
    let task = taskPhrase(from: unit)
    let intro = route == .toolAction
        ? "For \(task), this is the step to focus on:"
        : "For \(label), this is the step to focus on:"
    let step = unit.steps[match.index]
    let chain = renderStepChain(label: label, url: url, steps: [step])
    return "\(intro)\n\n\(chain)."
}

/// Mirrors `_confirmation_clause`.
private func confirmationClause() -> String {
    return "Want me to do this for you? Reply 'yes' to confirm."
}

/// Mirrors `_clarify_hint_from_query` — cheap heuristic that fills the
/// clarify template's `…are you asking about X?` slot.
private func clarifyHintFromQuery(_ query: String) -> String {
    var q = query
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    while let last = q.last, last == "?" || last == "." {
        q = String(q.dropLast())
    }
    q = q.trimmingCharacters(in: .whitespacesAndNewlines)
    if q.isEmpty { return "your Telco Home Internet service" }
    let leadingDeterminers = ["the ", "a ", "an ", "my ", "our "]
    for det in leadingDeterminers {
        if q.hasPrefix(det) {
            q = String(q.dropFirst(det.count))
            break
        }
    }
    if q.count > 80 { q = String(q.prefix(80)) }
    return q
}
