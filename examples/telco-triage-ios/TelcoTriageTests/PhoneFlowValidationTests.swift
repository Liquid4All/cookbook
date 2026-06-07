import XCTest
@testable import TelcoTriage

@MainActor
final class PhoneFlowValidationTests: XCTestCase {
    private let evalMarkerPath = "/tmp/enable-phone-flow-eval"
    private let realModelMarkerPath = "/tmp/enable-phone-flow-real-models"
    private let ablationModeMarkerPath = "/tmp/phone-flow-eval-ablation"
    private let minPassRateMarkerPath = "/tmp/phone-flow-eval-min-pass-rate"

    private enum RuntimeMode: String {
        case headlessFallback = "headless_fallback"
        case realBundledModels = "real_bundled_models"
        case sharedUnderstandingOnly = "shared_understanding_only"
        case turnRelationOnly = "turn_relation_only"

        var usesBackend: Bool {
            switch self {
            case .headlessFallback: return false
            case .realBundledModels, .sharedUnderstandingOnly, .turnRelationOnly: return true
            }
        }

        var usesSharedUnderstanding: Bool {
            switch self {
            case .realBundledModels, .sharedUnderstandingOnly: return true
            case .headlessFallback, .turnRelationOnly: return false
            }
        }

        var usesTurnRelation: Bool {
            switch self {
            case .realBundledModels, .turnRelationOnly: return true
            case .headlessFallback, .sharedUnderstandingOnly: return false
            }
        }

        var sendTimeout: TimeInterval {
            switch self {
            case .headlessFallback:
                return 5.0
            case .sharedUnderstandingOnly:
                return 8.0
            case .realBundledModels, .turnRelationOnly:
                return 12.0
            }
        }
    }

    private struct Fixture: Decodable {
        let version: Int
        let n_conversations: Int
        let n_turns: Int
        let conversations: [Conversation]
    }

    private struct Conversation: Decodable {
        let conversation_id: String
        let title: String
        let turns: [ExpectedTurn]
    }

    private struct ExpectedTurn: Decodable {
        let turn_index: Int
        let user_query: String
        let expected_route: String?
        let expected_content_page_id: String?
        let expected_page_section: String?
        let expected_link_id: String?
        let must_include: [String]
        let must_not_include: [String]
        let label_confidence: String
    }

    private struct EvalFailure: Encodable {
        let conversation_id: String
        let title: String
        let turn_index: Int
        let user_query: String
        let expected_route: String?
        let actual_route: String?
        let expected_page_id: String?
        let actual_page_id: String?
        let expected_section: String?
        let actual_section: String?
        let expected_link_id: String?
        let actual_link_id: String?
        let missing_terms: [String]
        let forbidden_terms: [String]
        let assistant_text: String
    }

    /// Per-turn decision trace for EVERY graded turn (pass or fail), so the
    /// situation-level decision-quality scorer (`scripts/vz/eval/situation_eval.py`)
    /// can map the app's actual decision against the situation taxonomy. This is
    /// the raw material for the product-readiness metric; the strict pass/fail
    /// rows above remain the regression-debugging layer.
    private struct Decision: Encodable {
        let conversation_id: String
        let turn_index: Int
        let user_query: String
        let route: String?
        let page_id: String?
        let link_id: String?
        let requires_confirmation: Bool?
        // ADR-029 §6: explicit policy-engine ground truth (no page-id inference).
        let reuse_active_evidence: Bool?
        let reason: String?
        // ADR-029 §7: the explicit dialogue-state operation + its audit reason, so
        // `situation_eval` scores the state decision directly.
        let state_operation: String?
        let state_operation_reason: String?
    }

    private struct EvalReport: Encodable {
        let runtime_mode: String
        let fixture_version: Int
        let conversations: Int
        let turns: Int
        let graded_turns: Int
        let failed_turns: Int
        let pass_rate: Double
        let structural_pass_rate: Double
        let text_pass_rate: Double
        let per_turn_latency_p50_ms: Double
        let per_turn_latency_p95_ms: Double
        let route_failures: Int
        let page_failures: Int
        let section_failures: Int
        let link_failures: Int
        let text_failures: Int
        let forbidden_failures: Int
        let failures: [EvalFailure]
        let decisions: [Decision]
    }

    private struct EvalRuntime {
        let mode: String
        let sendTimeout: TimeInterval
        let backend: LlamaBackend?
        let telcoUnderstandingClassifier: TelcoSharedUnderstandingClassifying?
        let relationalStrategy: RelationalHeadsStrategy

        func unload() async {
            await backend?.unload()
        }
    }

    func test_phoneFlowProbeFrom50Conversations() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["PHONE_FLOW_EVAL"] == "1" ||
            FileManager.default.fileExists(atPath: evalMarkerPath) else {
            throw XCTSkip(
                "Set PHONE_FLOW_EVAL=1 or create \(evalMarkerPath) to run the 50-conversation phone-flow harness."
            )
        }

        let fixture = try loadFixture()
        let runtime = try await makeRuntime()
        var failures: [EvalFailure] = []
        var gradedTurns = 0
        var structuralFailureTurns = 0
        var textFailureTurns = 0
        var routeFailures = 0
        var pageFailures = 0
        var sectionFailures = 0
        var linkFailures = 0
        var textFailures = 0
        var forbiddenFailures = 0
        var perTurnLatenciesMs: [Double] = []
        var decisions: [Decision] = []

        for conversation in fixture.conversations {
            let harness = try makeHarness(runtime: runtime)
            for turn in conversation.turns where turn.label_confidence != "skip" {
                gradedTurns += 1
                let turnStart = Date()
                await harness.send(turn.user_query, timeout: runtime.sendTimeout)
                perTurnLatenciesMs.append(Date().timeIntervalSince(turnStart) * 1000)
                guard let message = harness.lastAssistantMessage else {
                    decisions.append(
                        Decision(
                            conversation_id: conversation.conversation_id,
                            turn_index: turn.turn_index,
                            user_query: turn.user_query,
                            route: nil,
                            page_id: nil,
                            link_id: nil,
                            requires_confirmation: nil,
                            reuse_active_evidence: nil,
                            reason: nil,
                            state_operation: nil,
                            state_operation_reason: nil
                        )
                    )
                    failures.append(
                        EvalFailure(
                            conversation_id: conversation.conversation_id,
                            title: conversation.title,
                            turn_index: turn.turn_index,
                            user_query: turn.user_query,
                            expected_route: turn.expected_route,
                            actual_route: nil,
                            expected_page_id: turn.expected_content_page_id,
                            actual_page_id: nil,
                            expected_section: turn.expected_page_section,
                            actual_section: nil,
                            expected_link_id: turn.expected_link_id,
                            actual_link_id: nil,
                            missing_terms: turn.must_include,
                            forbidden_terms: [],
                            assistant_text: "<no assistant message>"
                        )
                    )
                    continue
                }

                let trace = message.trace
                let actualRoute = trace?.composerRoute
                let actualPageID = trace?.composerCitedPageID
                let actualSection = section(from: actualPageID)
                let actualLinkID = trace?.composerRenderedLinkID
                decisions.append(
                    Decision(
                        conversation_id: conversation.conversation_id,
                        turn_index: turn.turn_index,
                        user_query: turn.user_query,
                        route: actualRoute,
                        page_id: actualPageID,
                        link_id: actualLinkID,
                        requires_confirmation: trace?.composerConfirmationShown,
                        reuse_active_evidence: trace?.reuseActiveEvidence,
                        reason: trace?.policyReason,
                        state_operation: trace?.stateOperation,
                        state_operation_reason: trace?.stateOperationReason
                    )
                )

                let missingTerms = turn.must_include.filter {
                    !containsNormalized(message.text, $0)
                }
                let forbiddenTerms = turn.must_not_include.filter {
                    containsNormalized(message.text, $0)
                }

                let routeOK = turn.expected_route == nil || turn.expected_route == actualRoute
                let pageOK = turn.expected_content_page_id == nil ||
                    turn.expected_content_page_id == actualPageID
                let sectionOK = turn.expected_page_section == nil ||
                    turn.expected_page_section == actualSection
                let linkOK = turn.expected_link_id == nil || turn.expected_link_id == actualLinkID
                let textOK = missingTerms.isEmpty && forbiddenTerms.isEmpty
                let structuralOK = routeOK && pageOK && sectionOK && linkOK && forbiddenTerms.isEmpty

                if !routeOK { routeFailures += 1 }
                if !pageOK { pageFailures += 1 }
                if !sectionOK { sectionFailures += 1 }
                if !linkOK { linkFailures += 1 }
                if !missingTerms.isEmpty { textFailures += 1 }
                if !forbiddenTerms.isEmpty { forbiddenFailures += 1 }
                if !structuralOK { structuralFailureTurns += 1 }
                if !textOK { textFailureTurns += 1 }

                if !(routeOK && pageOK && sectionOK && linkOK && textOK) {
                    failures.append(
                        EvalFailure(
                            conversation_id: conversation.conversation_id,
                            title: conversation.title,
                            turn_index: turn.turn_index,
                            user_query: turn.user_query,
                            expected_route: turn.expected_route,
                            actual_route: actualRoute,
                            expected_page_id: turn.expected_content_page_id,
                            actual_page_id: actualPageID,
                            expected_section: turn.expected_page_section,
                            actual_section: actualSection,
                            expected_link_id: turn.expected_link_id,
                            actual_link_id: actualLinkID,
                            missing_terms: missingTerms,
                            forbidden_terms: forbiddenTerms,
                            assistant_text: message.text
                        )
                    )
                }
            }
        }

        let passRate = gradedTurns == 0
            ? 0
            : Double(gradedTurns - failures.count) / Double(gradedTurns)
        let structuralPassRate = gradedTurns == 0
            ? 0
            : Double(gradedTurns - structuralFailureTurns) / Double(gradedTurns)
        let textPassRate = gradedTurns == 0
            ? 0
            : Double(gradedTurns - textFailureTurns) / Double(gradedTurns)
        let report = EvalReport(
            runtime_mode: runtime.mode,
            fixture_version: fixture.version,
            conversations: fixture.n_conversations,
            turns: fixture.n_turns,
            graded_turns: gradedTurns,
            failed_turns: failures.count,
            pass_rate: passRate,
            structural_pass_rate: structuralPassRate,
            text_pass_rate: textPassRate,
            per_turn_latency_p50_ms: Self.percentile(perTurnLatenciesMs, 0.50),
            per_turn_latency_p95_ms: Self.percentile(perTurnLatenciesMs, 0.95),
            route_failures: routeFailures,
            page_failures: pageFailures,
            section_failures: sectionFailures,
            link_failures: linkFailures,
            text_failures: textFailures,
            forbidden_failures: forbiddenFailures,
            failures: failures,
            decisions: decisions
        )
        let reportURL = try writeReport(report)
        print("PHONE_FLOW_EVAL_REPORT=\(reportURL.path)")
        print(String(
            format: "PHONE_FLOW_LATENCY mode=%@ p50=%.0fms p95=%.0fms",
            runtime.mode, report.per_turn_latency_p50_ms, report.per_turn_latency_p95_ms
        ))
        await runtime.unload()

        let minPassRate = configuredMinPassRate()
        XCTAssertGreaterThanOrEqual(
            passRate,
            minPassRate,
            "Phone-flow pass rate \(String(format: "%.3f", passRate)) below \(minPassRate). Report: \(reportURL.path)"
        )
    }

    private func makeRuntime() async throws -> EvalRuntime {
        let mode = try requestedRuntimeMode()
        guard mode.usesBackend else {
            return EvalRuntime(
                mode: mode.rawValue,
                sendTimeout: mode.sendTimeout,
                backend: nil,
                telcoUnderstandingClassifier: nil,
                relationalStrategy: UnavailableRelationalStrategy()
            )
        }

        guard let basePath = TelcoModelBundle.basePath() else {
            throw XCTSkip("PHONE_FLOW_EVAL_REAL_MODELS=1 requires bundled \(TelcoModelBundle.baseModelName).")
        }

        let backend = LlamaBackend()
        try await backend.loadModel(
            path: basePath,
            contextLength: 8192,
            gpuLayers: 0,
            temperature: 0
        )
        let telcoUnderstandingClassifier = mode.usesSharedUnderstanding
            ? try TelcoSharedUnderstandingClassifier.bundled(backend: backend)
            : nil
        let relationalStrategy: RelationalHeadsStrategy
        if mode.usesTurnRelation,
           let bundledStrategy = try TelcoTurnRelationV4Strategy.bundled(backend: backend) {
            relationalStrategy = bundledStrategy
        } else {
            relationalStrategy = UnavailableRelationalStrategy()
        }

        return EvalRuntime(
            mode: mode.rawValue,
            sendTimeout: mode.sendTimeout,
            backend: backend,
            telcoUnderstandingClassifier: telcoUnderstandingClassifier,
            relationalStrategy: relationalStrategy
        )
    }

    private func requestedRuntimeMode() throws -> RuntimeMode {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["PHONE_FLOW_EVAL_ABLATION"],
           let mode = RuntimeMode(rawValue: raw.trimmingCharacters(in: .whitespacesAndNewlines)) {
            return mode
        }
        if let marker = try? String(contentsOfFile: ablationModeMarkerPath, encoding: .utf8) {
            let raw = marker.trimmingCharacters(in: .whitespacesAndNewlines)
            if let mode = RuntimeMode(rawValue: raw) {
                return mode
            }
            if !raw.isEmpty {
                throw XCTSkip("Unknown phone-flow ablation mode: \(raw)")
            }
        }
        if env["PHONE_FLOW_EVAL_REAL_MODELS"] == "1" ||
            FileManager.default.fileExists(atPath: realModelMarkerPath) {
            return .realBundledModels
        }
        return .headlessFallback
    }

    private func configuredMinPassRate() -> Double {
        if let marker = try? String(contentsOfFile: minPassRateMarkerPath, encoding: .utf8),
           let value = Double(marker.trimmingCharacters(in: .whitespacesAndNewlines)) {
            return value
        }
        return Double(
            ProcessInfo.processInfo.environment["PHONE_FLOW_EVAL_MIN_PASS_RATE"] ?? "1.0"
        ) ?? 1.0
    }

    private func makeHarness(runtime: EvalRuntime) throws -> TestChatHarness {
        _ = runtime.backend
        let corpus = try RAGUnitCorpus.loadFromBundle()
        let dispatcher = VerizonChatDispatcher(
            stageA: nil,
            stageB: nil,
            kbFallback: StubKBExtractor(),
            kb: [],
            retriever: nil,
            modelHost: nil,
            composer: DeterministicAnswerComposer(),
            corpus: corpus,
            lexicalRetriever: BM25HierarchyRetriever(corpus: corpus),
            toolRegistry: ToolRegistry.default(customerContext: CustomerContext()),
            toolAliasMap: ToolAliasMap.default()
        )
        return TestChatHarness(
            verizonDispatcher: dispatcher,
            telcoUnderstandingClassifier: runtime.telcoUnderstandingClassifier,
            relationalStrategy: runtime.relationalStrategy
        )
    }

    private func loadFixture() throws -> Fixture {
        let testSourcePath = #filePath as NSString
        let fixturePath = testSourcePath.deletingLastPathComponent +
            "/Fixtures/phone_flow_validation_v1.json"
        let data = try Data(contentsOf: URL(fileURLWithPath: fixturePath))
        return try JSONDecoder().decode(Fixture.self, from: data)
    }

    private func writeReport(_ report: EvalReport) throws -> URL {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("liquid_telco_phone_flow_eval_report_\(report.runtime_mode).json")
        try data.write(to: url, options: .atomic)
        return url
    }

    /// Nearest-rank percentile over per-turn latencies. Returns 0 for an
    /// empty sample so the report stays well-formed on degenerate runs.
    private static func percentile(_ samples: [Double], _ fraction: Double) -> Double {
        guard !samples.isEmpty else { return 0 }
        let sorted = samples.sorted()
        let rank = Int((fraction * Double(sorted.count)).rounded(.up)) - 1
        let index = min(max(rank, 0), sorted.count - 1)
        return sorted[index]
    }

    private func section(from pageID: String?) -> String? {
        guard let pageID, let dot = pageID.firstIndex(of: ".") else { return nil }
        return String(pageID[..<dot])
    }

    private func containsNormalized(_ haystack: String, _ needle: String) -> Bool {
        let normalizedHaystack = normalize(haystack)
        let normalizedNeedle = normalize(needle)
        guard !normalizedNeedle.isEmpty else { return true }
        return normalizedHaystack.contains(normalizedNeedle)
    }

    private func normalize(_ text: String) -> String {
        let folded = text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
            .lowercased()
            .replacingOccurrences(of: "wi fi", with: "wifi")
            .replacingOccurrences(of: "wi-fi", with: "wifi")
        let scalars = folded.unicodeScalars.map { scalar -> Character in
            CharacterSet.alphanumerics.contains(scalar) ? Character(scalar) : " "
        }
        return String(scalars)
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
    }
}
