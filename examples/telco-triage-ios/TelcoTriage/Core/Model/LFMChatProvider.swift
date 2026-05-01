import Foundation

/// On-device chat response generator backed by the base LFM2.5-350M
/// GGUF (no LoRA adapter on this path — the model is greedy-decoded
/// directly with a tight prompt).
///
/// Five prompt modes drive the five chat behaviors the demo exposes:
///
///  - `.groundedQA`       : RAG — short answer grounded in the top KB entry
///  - `.toolProposal`     : one-line framing above a ToolDecisionCard
///  - `.toolConfirmation` : one-line result summary after a tool executes
///  - `.profileSummary`   : plain-English summary of the CustomerContext
///  - `.outOfScope`       : polite on-device-only decline (privacy boundary)
///
/// Every mode caps output at ~120 tokens by default to keep warm latency
/// under a second on iPhone 15. The caller injects an
/// `AdapterInferenceBackend`; this provider passes `adapterPath: ""`
/// which the bridge treats as "base model only, no adapter swap."
///
/// There is no fallback path. If the backend throws, the caller sees
/// the thrown `LFMChatError`; the ChatViewModel surfaces it as a
/// short, system-labeled error message (not a canned "Phase 2" string).
public final class LFMChatProvider: ChatModelProvider {
    public enum Mode: Sendable {
        case groundedQA(topEntry: KBEntry)
        case toolProposal(tool: Tool, arguments: [String: String])
        case toolConfirmation(tool: Tool, result: ToolResult)
        case profileSummary(profile: CustomerProfile)
        case outOfScope(query: String)
    }

    private let backend: AdapterInferenceBackend
    private let maxTokens: Int

    public init(backend: AdapterInferenceBackend, maxTokens: Int = 120) {
        self.backend = backend
        self.maxTokens = maxTokens
    }

    // MARK: - ChatModelProvider

    /// Legacy protocol conformance: treat any call through the generic
    /// chat interface as a grounded-QA request over the first KB entry
    /// (or out-of-scope if context is empty). The new chips drive the
    /// richer `generate(mode:)` entrypoint directly via ChatViewModel.
    public func respond(
        to message: String,
        context: [KBEntry],
        history: [ChatTurn]
    ) async throws -> ChatModelResponse {
        let mode: Mode
        if let top = context.first {
            mode = .groundedQA(topEntry: top)
        } else {
            mode = .outOfScope(query: message)
        }
        return try await generate(query: message, mode: mode)
    }

    /// Primary entrypoint used by the new chat flow. Picks a prompt
    /// template, runs greedy decoding on the base GGUF, returns the
    /// structured response.
    public func generate(query: String, mode: Mode) async throws -> ChatModelResponse {
        let start = Date()
        let prompt = PromptBuilder.build(query: query, mode: mode)
        let raw: String
        do {
            raw = try await backend.generate(
                prompt: prompt,
                adapterPath: "",
                maxTokens: Self.maxTokens(for: mode, fallback: maxTokens),
                stopSequences: Self.stopSequences(for: mode)
            )
        } catch {
            throw LFMChatError.backendFailed(underlying: error)
        }

        let text = Self.cleanResponseText(raw)
        guard !text.isEmpty else { throw LFMChatError.emptyResponse }

        let elapsed = Int(Date().timeIntervalSince(start) * 1000)
        // Confidence is intentionally left uncalibrated on the base-model
        // chat path — the LFM intent classifier and tool selector emit
        // real softmax-style confidence scores; the base model doesn't.
        // `ChatModelResponse.confidence` carries a sentinel 0.0 here so
        // the trace row's confidence cell stays blank rather than
        // displaying a made-up number.
        return ChatModelResponse(
            text: text,
            confidence: 0.0,
            latencyMS: elapsed,
            usedContextIDs: Self.contextIDs(for: mode),
            deepLinks: Self.deepLinks(for: mode),
            inputTokens: TokenEstimator.estimate(prompt),
            outputTokens: TokenEstimator.estimate(text)
        )
    }

    // MARK: - Per-mode token budgets

    /// Token budgets per mode. 120 is tight enough for one-liners (tool
    /// proposal, confirmation, out-of-scope) but KB articles with multi-step
    /// instructions consistently exceed it — "why is my wifi slow" truncated
    /// at "Here are the steps for conducting a router speed test:" in build 21.
    /// groundedQA and profileSummary both need headroom for complete answers.
    static func maxTokens(for mode: Mode, fallback: Int) -> Int {
        switch mode {
        case .groundedQA:      return 256
        case .profileSummary:  return 256
        case .toolProposal, .toolConfirmation, .outOfScope:
            return fallback
        }
    }

    // MARK: - Stop sequences

    /// Stops fed to the backend per chat mode. Kept as a `switch`
    /// over the exhaustive `Mode` enum so callers never forget to
    /// pair a new mode with its stop markers — adding a case to
    /// `Mode` forces a compile error here until stops are defined.
    ///
    /// Today all five modes share the same fake-turn markers because
    /// every prompt ends with a section label ("Short answer:",
    /// "Summary:", "One-sentence reply:", …) and the 350M base model
    /// consistently emits one of the fake-turn prefixes below after
    /// its real answer — regardless of mode. If a future mode needs
    /// different stops (e.g. a JSON-in-chat mode where "}" is the
    /// terminator), branch here.
    ///
    /// The `"\nReply:"` marker is the specific fake-turn prefix
    /// observed in session-048 on "how do I restart my router"
    /// (→ "Reply:\n\nReply:\n\nReply:…").
    static func stopSequences(for mode: Mode) -> [String] {
        // Role-marker fake turns the 350M base tends to emit after its
        // real answer.
        let roleMarkers = [
            "\nCustomer:",
            "\nUser:",
            "\nAssistant:",
            "\nQuery:",
            "\nReply:",
        ]
        // Prompt-template echoes the model drops when it rolls past the
        // end of the intended answer (observed on TestFlight build 15
        // for "why is my wifi slow" — model produced a grounded answer,
        // then re-emitted "You are Telco Home Support. Answer the
        // customer…" verbatim). Catching these as stops prevents the
        // leak from reaching `cleanResponseText`.
        let promptLeakMarkers = [
            "\nYou are Telco",
            "\nReference:",
            "\nShort answer:",     // legacy terminator — kept as safety
            "\nComplete answer:",  // current groundedQA terminator
            "\nSummary:",
            "\nFact:",
            "\nTool:",
        ]
        switch mode {
        case .groundedQA, .toolProposal, .toolConfirmation,
             .profileSummary, .outOfScope:
            return roleMarkers + promptLeakMarkers
        }
    }

    // MARK: - Text cleanup

    /// Strip common base-model artifacts: trailing role-marker echoes,
    /// ``` fences, and any 6-word n-gram that repeats (a deterministic
    /// signature of greedy-decoding collapse).
    ///
    /// The n-gram guard is a belt-and-suspenders complement to the
    /// backend's `stopSequences`: some loops ("the router is connected
    /// to the router G3100 and the router is connected to…") happen
    /// inside the answer, before any fake-turn marker — the model
    /// never emits one because it's still inside the "Summary:" body.
    /// Truncating at the second occurrence of a 6-word sequence
    /// recovers a clean partial answer instead of a collapsed one.
    static func cleanResponseText(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // Strip code fences if the model wrapped the answer.
        if s.hasPrefix("```") {
            if let nl = s.firstIndex(of: "\n") {
                s = String(s[s.index(after: nl)...])
            }
            if s.hasSuffix("```") {
                s = String(s.dropLast(3))
            }
            s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        // If the model kept writing past our intent (e.g. started a
        // second fake turn OR re-emitted the prompt template), cut at
        // the first marker. The prompt-leak markers cover both the
        // instruction line ("You are Telco …") and the reference /
        // summary section labels every template uses.
        let cutMarkers = [
            "\nCustomer:",
            "\nUser:",
            "\nQuery:",
            "\nAssistant:",
            "\nReply:",
            "\nYou are Telco",
            "\nReference:",
            "\nShort answer:",
            "\nComplete answer:",
            "\nSummary:",
            "\nFact:",
            "\nTool:",
        ]
        for marker in cutMarkers {
            if let range = s.range(of: marker) {
                s = String(s[..<range.lowerBound])
            }
        }
        s = truncateAtRepeatedNgram(s, ngramSize: 6)
        s = truncateAtRepeatedSentenceStart(s)
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Cut output as soon as a sentence-starting stem appears 3 or more
    /// times. Catches the class of loop the 6-word n-gram guard misses:
    /// structural repetition where the prefix is fixed but the tail
    /// varies.
    ///
    /// Seen on "How do I do this?" → "The top-right button is
    /// Notifications. … The top-right button is Troubleshoot. The
    /// top-right button is Speed Test. The top-right button is Restart
    /// Router." The 5-word prefix "the top-right button is" never
    /// repeats as a full 6-gram because word 6 varies, so the n-gram
    /// guard is blind to it. Counting sentence stems catches it.
    ///
    /// Threshold is 3 (not 2) so legitimate "If X then Y. If X then Z."
    /// patterns stay intact.
    static func truncateAtRepeatedSentenceStart(
        _ text: String,
        stemWords: Int = 4,
        threshold: Int = 3
    ) -> String {
        // Walk the text once, recording (startIndex, sentence) for each
        // sentence that carries enough words to form a stem. Short
        // interjections ("OK.", "Sure.") are skipped so they don't
        // perturb the index math when cutting.
        var starts: [String.Index] = []
        var sentences: [String] = []
        var sentenceStart = text.startIndex
        var cursor = text.startIndex
        while cursor < text.endIndex {
            let ch = text[cursor]
            if ".!?".contains(ch) {
                let raw = String(text[sentenceStart..<cursor])
                    .trimmingCharacters(in: .whitespaces)
                if raw.split(whereSeparator: { $0.isWhitespace }).count >= stemWords {
                    starts.append(sentenceStart)
                    sentences.append(raw)
                }
                cursor = text.index(after: cursor)
                while cursor < text.endIndex, text[cursor].isWhitespace {
                    cursor = text.index(after: cursor)
                }
                sentenceStart = cursor
                continue
            }
            cursor = text.index(after: cursor)
        }

        guard sentences.count >= threshold else { return text }

        var stemCounts: [String: Int] = [:]
        for (idx, sentence) in sentences.enumerated() {
            let stem = sentence
                .split(whereSeparator: { $0.isWhitespace })
                .prefix(stemWords)
                .joined(separator: " ")
                .lowercased()
            stemCounts[stem, default: 0] += 1
            if stemCounts[stem]! >= threshold {
                // Cut at the start of the 3rd-repeated sentence so
                // the first two occurrences are preserved (they look
                // like legitimate structure; only beyond that is loop).
                return String(text[..<starts[idx]])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return text
    }

    /// If any whitespace-delimited `ngramSize`-word sequence appears a
    /// second time, return the prefix that ends right before that
    /// second occurrence. Returns the input unchanged if no repetition
    /// is found or the text is too short.
    ///
    /// Conservative by design: 6-word runs rarely repeat in legitimate
    /// English, and the check is case-insensitive so "Router" and
    /// "router" share a signature.
    static func truncateAtRepeatedNgram(_ text: String, ngramSize: Int) -> String {
        precondition(ngramSize >= 2, "n-gram size must be at least 2")
        let words = text.split(
            whereSeparator: { $0.isWhitespace }
        ).map(String.init)
        guard words.count >= ngramSize * 2 else { return text }

        var firstStart: [String: Int] = [:]
        var secondStart: Int?
        for i in 0...(words.count - ngramSize) {
            let gram = words[i..<(i + ngramSize)]
                .joined(separator: " ")
                .lowercased()
            if firstStart[gram] != nil {
                secondStart = i
                break
            }
            firstStart[gram] = i
        }
        guard let cutAtWord = secondStart else { return text }

        // Find the byte offset in the original text at which the
        // `cutAtWord`-th word starts. Walk the original string
        // once so Unicode boundaries are respected.
        var wordIndex = 0
        var cursor = text.startIndex
        var inWord = false
        while cursor < text.endIndex {
            let ch = text[cursor]
            if ch.isWhitespace {
                if inWord {
                    wordIndex += 1
                    if wordIndex == cutAtWord {
                        return String(text[..<cursor])
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                    inWord = false
                }
            } else {
                inWord = true
            }
            cursor = text.index(after: cursor)
        }
        return text
    }

    private static func contextIDs(for mode: Mode) -> [String] {
        switch mode {
        case .groundedQA(let entry): return [entry.id]
        case .toolProposal, .toolConfirmation, .profileSummary, .outOfScope: return []
        }
    }

    private static func deepLinks(for mode: Mode) -> [DeepLink] {
        switch mode {
        case .groundedQA(let entry): return entry.deepLinks
        case .toolProposal(let tool, _), .toolConfirmation(let tool, _):
            return tool.deepLink.map { [$0] } ?? []
        case .profileSummary, .outOfScope: return []
        }
    }
}

// MARK: - Errors

public enum LFMChatError: Error, LocalizedError {
    case backendFailed(underlying: Error)
    case emptyResponse

    public var errorDescription: String? {
        switch self {
        case .backendFailed(let u): return "On-device inference failed: \(u.localizedDescription)"
        case .emptyResponse: return "On-device inference returned an empty response."
        }
    }
}

// MARK: - Prompt templates

/// Prompt construction for LFMChatProvider. Templates live as static
/// strings so they're grep-friendly and can be unit-tested without
/// running the model.
enum PromptBuilder {
    static func build(query: String, mode: LFMChatProvider.Mode) -> String {
        switch mode {
        case .groundedQA(let entry):
            return groundedQA(query: query, entry: entry)
        case .toolProposal(let tool, let arguments):
            return toolProposal(query: query, tool: tool, arguments: arguments)
        case .toolConfirmation(let tool, let result):
            return toolConfirmation(tool: tool, result: result)
        case .profileSummary(let profile):
            return profileSummary(query: query, profile: profile)
        case .outOfScope(let q):
            return outOfScope(query: q)
        }
    }

    // MARK: RAG

    static func groundedQA(query: String, entry: KBEntry) -> String {
        // The word "short" dominated the model's conditioning on terse
        // queries like "How do I restart?" — the base emitted two words
        // ("Restart router.") and stopped. Instruction now requires
        // *complete* sentences, and the terminator label mirrors that.
        return """
        You are Telco Home Support. Answer the customer's question in 2-3 complete sentences using only the reference below. Do not invent steps. Write full, natural sentences; do not truncate.

        Customer: \(query)

        Reference: \(entry.topic)
        \(entry.answer)

        Complete answer:
        """
    }

    // MARK: Tool proposal (one-liner above ToolDecisionCard)

    static func toolProposal(query: String, tool: Tool, arguments: [String: String]) -> String {
        let argsLine = arguments.isEmpty
            ? "(no arguments)"
            : arguments.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: ", ")
        return """
        You are Telco Home Support. The on-device tool selector chose \(tool.displayName) for the customer's request.
        Confirm the action you are about to take in ONE short sentence. Mention the key argument if any.
        Do not describe steps. Do not say "I will." Keep it under 20 words.

        Customer: \(query)

        Tool: \(tool.id) (\(tool.displayName))
        Arguments: \(argsLine)

        One-sentence confirmation prompt:
        """
    }

    // MARK: Tool confirmation (shown after the tool runs)

    static func toolConfirmation(tool: Tool, result: ToolResult) -> String {
        let payload = result.structuredPayload.isEmpty
            ? "(no structured output)"
            : result.structuredPayload.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: ", ")
        return """
        You are Telco Home Support. The \(tool.displayName) tool just ran.
        Summarize the outcome for the customer in ONE short sentence. Use the structured output below.
        Be specific (mention device names, timings, values). Don't repeat the raw JSON.

        Tool: \(tool.id)
        Status: \(result.status.rawValue)
        Human summary (machine-written): \(result.humanSummary)
        Structured output: \(payload)

        Customer-facing summary:
        """
    }

    // MARK: Profile summary

    /// DEAD CODE as of build 16 — retained so the `Mode.profileSummary`
    /// case still compiles and so the `respond(to:context:history:)` legacy
    /// protocol path still has a prompt shape to fall through to. The
    /// 350M base pattern-locks on the key:value data block at greedy
    /// decode and echoes the fields verbatim; local harness
    /// (scripts/test_telco_chat_pipeline_local.py) and TestFlight both showed
    /// the failure. ChatViewModel.runPersonalizedSummary now renders a
    /// deterministic prose summary directly from the profile. See F8
    /// in FEATURES.yaml for the v2 plan to re-enable LFM generation once
    /// we have a 1.2B-class summarizer adapter trained on household
    /// summarization pairs.
    static func profileSummary(query: String, profile: CustomerProfile) -> String {
        return """
        Summarize the home internet customer's household in 2-3 short sentences.

        Query: \(query)
        Plan: \(profile.plan.name) (\(profile.plan.downSpeedMbps)/\(profile.plan.upSpeedMbps) Mbps)

        Summary:
        """
    }

    // MARK: Out of scope

    static func outOfScope(query: String) -> String {
        return """
        You are Telco Home Support. The customer asked something outside the scope of home internet support (e.g. general knowledge, unrelated product).
        Reply in ONE short, friendly sentence that (a) you only handle home internet support and (b) nothing about their query is sent off-device.
        Do not attempt to answer the question.

        Customer: \(query)

        One-sentence reply:
        """
    }
}
