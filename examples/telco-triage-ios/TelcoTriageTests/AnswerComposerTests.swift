import XCTest
@testable import TelcoTriage

/// Mirrors `scripts/telco/tests/test_answer_composer.py` 1:1. Any
/// behaviour change in `AnswerComposer.swift` MUST be reflected in
/// both the Python and Swift composers; this suite is the Swift side
/// of that contract.
final class AnswerComposerTests: XCTestCase {
    private var composer: DeterministicAnswerComposer!
    private var units: [String: RAGUnit]!

    override func setUpWithError() throws {
        try super.setUpWithError()
        composer = DeterministicAnswerComposer()
        units = TestUnitFixture.makeUnits()
    }

    // MARK: - Render helpers

    func test_renderLink_basic() {
        XCTAssertEqual(renderLink(label: "Network", url: "telcohome://network"),
                       "[Network](telcohome://network)")
    }

    func test_renderLink_escapes_brackets() {
        XCTAssertEqual(renderLink(label: "Bad [Label]", url: "telcohome://x"),
                       "[Bad (Label)](telcohome://x)")
    }

    func test_renderStepChain_basic() {
        let chain = renderStepChain(
            label: "Restart Router",
            url: "telcohome://restart-router",
            steps: ["Tap Equipment.", "Select Restart router.", "Confirm."]
        )
        XCTAssertEqual(
            chain,
            "[Restart Router](telcohome://restart-router) > Tap Equipment > Select Restart router > Confirm"
        )
    }

    func test_renderStepChain_strips_internal_arrows() {
        let chain = renderStepChain(label: "X", url: "telcohome://x", steps: ["a > b"])
        XCTAssertTrue(chain.contains(" > "), "outer arrow separator present")
        XCTAssertTrue(chain.contains("a - b") || chain.contains("a   b"),
                      "internal `>` defused")
    }

    func test_extractRenderedLinks_pulls_urls_and_labels() {
        let (urls, labels) = extractRenderedLinks(
            "Go to [Network](telcohome://network) and [Speed Test](telcohome://speed-test)."
        )
        XCTAssertEqual(urls, ["telcohome://network", "telcohome://speed-test"])
        XCTAssertEqual(labels, ["Network", "Speed Test"])
    }

    // MARK: - Route templates

    func test_greeting_template() {
        let ans = composer.compose(query: "Hi", route: .greeting, evidence: nil)
        XCTAssertTrue(ans.text.contains("Hello"))
        XCTAssertTrue(ans.renderedLinks.isEmpty)
        XCTAssertNil(ans.citedPageID)
    }

    func test_outOfScope_template() {
        let ans = composer.compose(query: "weather", route: .outOfScope, evidence: nil)
        XCTAssertTrue(ans.text.contains("Telco Home Internet"))
        XCTAssertTrue(ans.renderedLinks.isEmpty)
    }

    func test_noRagAnswer_has_external_url() {
        let ans = composer.compose(query: "??", route: .noRagAnswer, evidence: nil)
        XCTAssertEqual(ans.renderedLinks, [AnswerComposerConstants.telcoInternetURL])
        XCTAssertFalse(ans.renderedLinks.contains { $0.hasPrefix("telcohome://") })
    }

    func test_liveAgent_uses_phone() {
        let ans = composer.compose(query: "help me", route: .liveAgent, evidence: nil)
        XCTAssertEqual(ans.renderedLinks, [AnswerComposerConstants.liveAgentPhone])
    }

    func test_accountNav_uses_my_telco() {
        let unit = units["10.00"]!
        let ans = composer.compose(query: "my bill", route: .accountNav, evidence: unit)
        XCTAssertEqual(ans.renderedLinks, [AnswerComposerConstants.myTelcoURL])
        XCTAssertFalse(ans.renderedLinks.contains { $0.hasPrefix("telcohome://") })
    }

    func test_clarify_no_link() {
        let ans = composer.compose(query: "phone", route: .clarify, evidence: nil)
        XCTAssertTrue(ans.text.lowercased().contains("clarify"))
        XCTAssertTrue(ans.renderedLinks.isEmpty)
    }

    func test_ragAnswer_with_steps_renders_chain() {
        let unit = units["02.07"]!
        let ans = composer.compose(
            query: "restart my router",
            route: .ragAnswer,
            evidence: unit,
            requiresConfirmation: false
        )
        XCTAssertTrue(ans.text.contains("telcohome://restart-router"))
        XCTAssertTrue(ans.text.contains(" > Tap Equipment > "))
        XCTAssertTrue(ans.hasStepChain)
        XCTAssertEqual(ans.renderedLinks.first, "telcohome://restart-router")
    }

    func test_ragAnswer_without_steps_renders_link_only() {
        let unit = units["03.00"]!
        let ans = composer.compose(
            query: "change wifi password",
            route: .ragAnswer,
            evidence: unit
        )
        XCTAssertTrue(ans.text.contains("telcohome://network"))
        XCTAssertFalse(ans.hasStepChain)
    }

    func test_ragAnswer_without_steps_extracts_grounded_summary() {
        let unit = units["13.02"]!
        let ans = composer.compose(
            query: "add a profile for my son",
            route: .ragAnswer,
            evidence: unit,
            requiresConfirmation: false
        )

        XCTAssertFalse(ans.text.contains("I found the relevant page"))
        XCTAssertTrue(ans.text.contains("group children's devices"))
        XCTAssertTrue(ans.text.contains("Name this profile"))
        XCTAssertTrue(ans.text.contains("Select a color"))
        XCTAssertTrue(ans.text.contains("telcohome://home"))
        XCTAssertFalse(ans.text.contains("Reply 'yes'"))
        XCTAssertFalse(ans.hasStepChain)
    }

    func test_toolAction_carries_confirmation() {
        let unit = units["02.07"]!
        let ans = composer.compose(
            query: "restart my router",
            route: .toolAction,
            evidence: unit,
            requiresConfirmation: true
        )
        XCTAssertTrue(ans.text.lowercased().contains("confirm"))
        XCTAssertFalse(ans.actionFired)
        XCTAssertTrue(ComposerGrading.isActionSafe(ans))
    }

    func test_toolAction_no_confirmation_when_not_required() {
        // run-speed-test has ToolIntent.requiresConfirmation == false.
        // The composer must NOT add the "Reply 'yes'" clause in that
        // case (the dispatcher fires the tool immediately).
        let unit = units["01.02"]!
        let ans = composer.compose(
            query: "run a speed test",
            route: .toolAction,
            evidence: unit,
            requiresConfirmation: false
        )
        XCTAssertTrue(ans.text.contains("I can run a speed test for you."))
        XCTAssertFalse(ans.text.contains("Reply 'yes'"))
        XCTAssertFalse(ans.text.lowercased().contains("confirm"))
    }

    func test_answerPlusAction_adds_confirmation_when_required() {
        let unit = units["02.07"]!
        let ans = composer.compose(
            query: "how do I restart my router?",
            route: .answerPlusAction,
            evidence: unit,
            requiresConfirmation: true
        )
        XCTAssertTrue(ans.text.contains("telcohome://restart-router"))
        XCTAssertTrue(ans.text.lowercased().contains("confirm"))
    }

    func test_ragAnswer_without_evidence_uses_safe_fallback() {
        let ans = composer.compose(query: "x", route: .ragAnswer, evidence: nil)
        XCTAssertTrue(ans.usedFallback)
        XCTAssertTrue(ans.text.contains(AnswerComposerConstants.telcoInternetURL))
        XCTAssertFalse(ans.renderedLinks.contains { $0.hasPrefix("telcohome://") })
    }

    // MARK: - Grading helpers

    func test_isLinkValid_accepts_known_telcohome() {
        let ans = ComposedAnswer(
            text: "[Network](telcohome://network)",
            route: .ragAnswer,
            renderedLinks: ["telcohome://network"],
            renderedLinkLabels: ["Network"],
            expectedLinkURL: "telcohome://network"
        )
        XCTAssertTrue(ComposerGrading.isLinkValid(ans, knownTelcoHomeURLs: ["telcohome://network"]))
    }

    func test_isLinkValid_rejects_unknown_telcohome() {
        let ans = ComposedAnswer(
            text: "[Bogus](telcohome://bogus)",
            route: .ragAnswer,
            renderedLinks: ["telcohome://bogus"],
            renderedLinkLabels: ["Bogus"]
        )
        XCTAssertFalse(ComposerGrading.isLinkValid(ans, knownTelcoHomeURLs: ["telcohome://network"]))
    }

    func test_isLinkValid_allows_external_fallback() {
        let ans = ComposedAnswer(
            text: "[Telco Home Internet](\(AnswerComposerConstants.telcoInternetURL))",
            route: .noRagAnswer,
            renderedLinks: [AnswerComposerConstants.telcoInternetURL],
            renderedLinkLabels: ["Telco Home Internet"]
        )
        XCTAssertTrue(ComposerGrading.isLinkValid(ans, knownTelcoHomeURLs: []))
    }

    func test_isCitationCorrect_for_query_suffix_equivalence() {
        let ans = ComposedAnswer(
            text: "[Network](telcohome://network?launchPoint=telcoAssistant)",
            route: .ragAnswer,
            renderedLinks: ["telcohome://network?launchPoint=telcoAssistant"],
            renderedLinkLabels: ["Network"],
            expectedLinkURL: "telcohome://network"
        )
        XCTAssertTrue(ComposerGrading.isCitationCorrect(ans))
    }

    func test_isGrounded_rejects_extra_telcohome_link() {
        let unit = units["02.07"]!
        let base = composer.compose(
            query: "restart my router",
            route: .ragAnswer,
            evidence: unit
        )
        let polluted = ComposedAnswer(
            text: base.text + "\n\nAlso try [Network](telcohome://network).",
            route: base.route,
            citedPageID: base.citedPageID,
            renderedLinks: base.renderedLinks + ["telcohome://network"],
            renderedLinkLabels: base.renderedLinkLabels + ["Network"],
            expectedLinkURL: base.expectedLinkURL,
            requiresConfirmation: base.requiresConfirmation,
            actionFired: false
        )
        XCTAssertFalse(ComposerGrading.isGrounded(polluted, evidence: unit))
    }

    func test_isActionSafe_requires_confirm_for_toolAction() {
        let ans = ComposedAnswer(
            text: "Restarting now.",
            route: .toolAction,
            citedPageID: "02.07",
            renderedLinks: [],
            requiresConfirmation: true
        )
        XCTAssertFalse(ComposerGrading.isActionSafe(ans))
    }

    func test_refusal_templates_pass_grading() {
        for route in [ComposerRoute.greeting, .outOfScope, .noRagAnswer, .liveAgent, .clarify] {
            let ans = composer.compose(query: "x", route: route, evidence: nil)
            XCTAssertTrue(
                ComposerGrading.isRefusalTemplateCorrect(ans),
                "\(route.wireName) template failed"
            )
        }
    }

    func test_isFormatCompliant_for_rag_routes() {
        let unit = units["02.07"]!
        let ans = composer.compose(query: "restart", route: .ragAnswer, evidence: unit)
        XCTAssertTrue(ComposerGrading.isFormatCompliant(ans))
    }

    // MARK: - All-routes-covered sanity

    func test_all_routes_have_template_coverage() {
        // Iterating the enum proves Swift exhaustiveness matches the
        // Python `ALL_ROUTES` set.
        for route in ComposerRoute.allCases {
            let evidence: RAGUnit? = route.requiresEvidence ? units["02.07"] : nil
            let ans = composer.compose(query: "test", route: route, evidence: evidence)
            XCTAssertFalse(ans.text.isEmpty, "route \(route.wireName) produced empty text")
        }
    }
}

// MARK: - Synthetic mini-corpus

enum TestUnitFixture {
    static func makeUnits() -> [String: RAGUnit] {
        return [
            "01.00": _u(page: "01.00", title: "Home page", link: "tab-home"),
            "01.02": _u(
                page: "01.02",
                title: "Router speed test page",
                link: "speed-test",
                url: "telcohome://speed-test?launchPoint=telcoAssistant",
                citation: "Speed Test",
                task: "run-speed-test",
                steps: [
                    "Select Equipment from the Home page.",
                    "Tap Router Speed Test.",
                    "Tap Start.",
                ],
                affordance: "assist"
            ),
            "02.07": _u(
                page: "02.07",
                title: "Restart router",
                link: "restart-router",
                citation: "Restart Router",
                task: "restart-router",
                steps: ["Tap Equipment.", "Select Restart router.", "Confirm."],
                affordance: "tool_action"
            ),
            "03.00": _u(
                page: "03.00",
                title: "Network page",
                link: "network",
                url: "telcohome://network?launchPoint=telcoAssistant",
                citation: "Network",
                aliases: ["wifi", "wifi password", "change wifi password"],
                affordance: "view"
            ),
            "10.00": _u(
                page: "10.00",
                title: "My Telco App",
                link: "my-telco-app",
                citation: "My Telco App",
                aliases: ["my bill", "my account"],
                affordance: "navigate"
            ),
            "13.00": _u(
                page: "13.00",
                title: "Parental Controls profiles page",
                link: "home",
                url: "telcohome://home?launchPoint=telcoAssistant",
                citation: "Parental Controls",
                aliases: ["parental controls", "kid mode"],
                steps: ["Tap Devices.", "Pick a profile.", "Adjust restrictions."],
                affordance: "view"
            ),
            "13.02": _u(
                page: "13.02",
                title: "Create profile page",
                link: "home",
                url: "telcohome://home?launchPoint=telcoAssistant",
                citation: "Create profile",
                aliases: ["add profile", "create profile", "profile for son"],
                body: """
                The Parental Controls "Create Profile" page enables users to group children's devices for managing internet access.
                At the top of the Create profile page, is an Add photo button.
                Under the photo section is the step to Name this profile. This is a required field.
                Below that is Select a color. This allows the user to assign different colors for their children's profiles.
                Selecting Next will bring the user to an Assign devices page.
                """,
                affordance: "tool_action"
            ),
        ]
    }

    private static func _u(
        page: String,
        title: String,
        link: String,
        url: String? = nil,
        citation: String? = nil,
        task: String? = nil,
        aliases: [String] = [],
        steps: [String] = [],
        body: String = "",
        answerSummary: String? = nil,
        answerFacts: [String]? = nil,
        affordance: String = "view"
    ) -> RAGUnit {
        RAGUnit(
            pageID: page,
            taskID: task,
            title: title,
            section: "Home",
            level: 1,
            parentPageID: nil,
            linkID: link,
            canonicalURL: url ?? "telcohome://\(link)",
            aliases: aliases,
            steps: steps,
            body: body,
            sourceDoc: "synthetic.docx",
            citationLabel: citation ?? title,
            actionAffordance: affordance,
            availability: nil,
            answerSummary: answerSummary,
            answerFacts: answerFacts
        )
    }
}
