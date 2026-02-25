# /test-usecase

Run the integration test for a specific use case from the PRD.

**Usage:** `/test-usecase <UC-number>` (e.g., `/test-usecase UC-1` or `/test-usecase 1`)

## Use Case Reference

| UC | Name | Required Servers | Test File |
|----|------|-----------------|-----------|
| UC-1 | Receipt → Reconciliation | filesystem, ocr, data, document | uc1_receipt_reconciliation.test.ts |
| UC-2 | Contract / NDA Copilot | document, knowledge, email | uc2_contract_copilot.test.ts |
| UC-3 | Security & Privacy Steward | filesystem, security, audit, document | uc3_security_steward.test.ts |
| UC-4 | Download Triage | filesystem, ocr, document, data | uc4_download_triage.test.ts |
| UC-5 | Screenshot-to-Action | system, clipboard, ocr, filesystem, document, data | uc5_screenshot_action.test.ts |
| UC-6 | Meeting-to-Execution | meeting, calendar, task, email, knowledge, document | uc6_meeting_pipeline.test.ts |
| UC-7 | Personal Operations OS | task, calendar, email, clipboard | uc7_personal_ops.test.ts |
| UC-8 | Portfolio / Deal Memo | document, knowledge, task, data | uc8_deal_memo.test.ts |
| UC-9 | Local Codebase Navigator | knowledge, filesystem, system, clipboard | uc9_codebase_navigator.test.ts |
| UC-10 | Compliance Pack Generator | audit, document, filesystem | uc10_compliance_pack.test.ts |

## Steps

1. Parse the UC number from `$ARGUMENTS` (accept "UC-1", "UC1", or just "1").
2. Read `docs/PRD.md` Section 6 for the detailed flow of this use case.
3. Check that all required MCP servers for this UC are built:
   - For each required server, check if `mcp-servers/<server>/src/index.ts` (or `__init__.py`) exists.
   - For each required server, check if unit tests pass: run `npm test` or `pytest` in the server directory.
   - If any server is missing or failing, report which ones and stop.
4. Verify test fixtures exist in `tests/fixtures/` for this UC.
   - If fixtures are missing, list what's needed (e.g., "UC-1 needs sample receipt images and PDFs in tests/fixtures/uc1/").
5. Run the integration test: `npx vitest run tests/integration/uc<N>_*.test.ts`
6. Report results:
   - ✅ All assertions passed
   - ❌ Failures with details (which tool chain step failed, expected vs actual)
   - ⏭️ Skipped (server not available)

## Notes

- Integration tests simulate the model's tool-calling by directly invoking MCP server tools in the sequence defined by the PRD flow.
- These tests do NOT require a running LLM — they test the tool chain, not the model's ability to select tools.
- Model-level integration is tested separately via `/model-test`.
