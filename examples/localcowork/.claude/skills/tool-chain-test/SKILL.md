---
name: tool-chain-test
description: >
  Skill for writing and running integration tests that chain multiple MCP tools
  together, simulating real user workflows from the PRD. Use when implementing
  or testing any of the 10 use cases (UC-1 through UC-10), verifying that MCP
  servers compose correctly, or building end-to-end test scenarios.
  MANDATORY TRIGGERS: "integration test", "use case test", "UC-1" through "UC-10",
  "tool chain", "end-to-end test", "workflow test", "compose tools", or any mention
  of testing a specific use case by name (receipt reconciliation, contract copilot,
  download triage, meeting pipeline, etc.).
---

# Tool Chain Integration Testing Skill

## What These Tests Do

Integration tests simulate the model's tool-calling behavior by directly invoking
MCP server tools in the sequence defined by each use case in the PRD. They verify
that the tool chain produces correct results end-to-end — but they do NOT test the
model's ability to select the right tools (that's the model-behavior test suite).

Think of it this way:
- **Unit tests** verify each tool works alone
- **Integration tests** (this skill) verify tools compose correctly
- **Model behavior tests** verify the LLM picks the right tools

## Before You Start

Read these to understand the test you're writing:

1. `docs/PRD.md` Section 6 — the specific use case flow (UC-1 through UC-10)
2. `docs/mcp-tool-registry.yaml` — tool signatures for the servers involved
3. `.claude/commands/test-usecase.md` — the slash command reference for which servers each UC needs

## Test Structure

Every UC integration test follows this structure:

```typescript
// tests/integration/uc<N>_<name>.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { TestHarness } from '../helpers/test-harness';

describe('UC-<N>: <Use Case Name>', () => {
  let harness: TestHarness;

  beforeAll(async () => {
    // 1. Start required MCP servers
    harness = await TestHarness.create({
      servers: ['filesystem', 'ocr', 'data'],  // only servers needed for this UC
      fixtures: 'uc<N>',                        // loads from tests/fixtures/uc<N>/
    });
  });

  afterAll(async () => {
    // Clean up: stop servers, remove temp files
    await harness.teardown();
  });

  it('should complete the full workflow', async () => {
    // Follow the PRD flow step by step
    // Each step calls a tool and verifies the intermediate result
  });

  // Optional: test individual steps in isolation
  it('should handle step 3 when OCR confidence is low', async () => {
    // Edge case for specific step
  });
});
```

## Test Harness

The test harness (`tests/helpers/test-harness.ts`) manages:
- Starting/stopping MCP servers as child processes
- Loading test fixtures from `tests/fixtures/`
- Providing typed wrappers around tool calls
- Cleaning up temp files after tests

```typescript
// tests/helpers/test-harness.ts
export class TestHarness {
  private servers: Map<string, MCPServerProcess>;
  private fixtureDir: string;
  private tempDir: string;

  static async create(config: { servers: string[], fixtures: string }): Promise<TestHarness>;

  // Call a tool on a specific server
  async callTool<T>(serverName: string, toolName: string, params: object): Promise<T>;

  // Get the path to a fixture file
  fixturePath(filename: string): string;

  // Get a temp directory for output files
  tempPath(filename: string): string;

  // Clean up everything
  async teardown(): Promise<void>;
}
```

## Fixture Organization

```
tests/fixtures/
├── uc1/                          # Receipt Reconciliation
│   ├── receipts/
│   │   ├── receipt_coffee.jpg     # Sample receipt image
│   │   ├── receipt_office.pdf     # Sample receipt PDF
│   │   └── invoice_vendor.pdf     # Sample invoice
│   └── expected/
│       └── receipts.csv           # Expected output CSV
│
├── uc2/                          # Contract Copilot
│   ├── original_nda.pdf
│   ├── revised_nda.docx
│   └── expected/
│       └── diff_report.json
│
├── uc3/                          # Security Steward
│   ├── sample_files/
│   │   ├── has_ssn.txt           # Contains fake SSN for testing
│   │   ├── has_api_key.env       # Contains fake API key
│   │   └── clean_file.txt        # No sensitive data
│   └── expected/
│       └── findings.json
│
├── uc4/                          # Download Triage
│   ├── downloads/
│   │   ├── quarterly_report.pdf
│   │   ├── photo_vacation.jpg
│   │   ├── node-v20.pkg
│   │   └── receipt_amazon.pdf
│   └── expected/
│       └── classification.json
│
├── uc6/                          # Meeting Pipeline
│   ├── meeting_audio.wav          # Short sample audio (30 sec)
│   └── expected/
│       ├── transcript.json
│       └── action_items.json
│
└── shared/                       # Fixtures used across multiple UCs
    ├── sample.pdf
    └── sample.docx
```

## Example: UC-1 Receipt Reconciliation

Following the PRD Section 6, UC-1 flow step by step:

```typescript
describe('UC-1: Receipt → Reconciliation', () => {
  let harness: TestHarness;

  beforeAll(async () => {
    harness = await TestHarness.create({
      servers: ['filesystem', 'ocr', 'data', 'document'],
      fixtures: 'uc1',
    });
  });

  it('should process a receipt folder into structured CSV', async () => {
    const receiptDir = harness.fixturePath('receipts');

    // Step 1: List directory
    const files = await harness.callTool<FileInfo[]>(
      'filesystem', 'list_dir', { path: receiptDir }
    );
    expect(files.length).toBe(3);

    // Steps 2-3: OCR + structured extraction per file
    const records = [];
    for (const file of files) {
      // Step 2a: OCR
      let ocrResult;
      if (file.name.endsWith('.jpg')) {
        ocrResult = await harness.callTool(
          'ocr', 'extract_text_from_image', { path: file.path }
        );
      } else {
        ocrResult = await harness.callTool(
          'ocr', 'extract_text_from_pdf', { path: file.path }
        );
      }
      expect(ocrResult.text.length).toBeGreaterThan(0);

      // Step 3: Extract structured data
      const structured = await harness.callTool(
        'ocr', 'extract_structured_data', {
          text: ocrResult.text,
          schema: {
            type: 'object',
            properties: {
              vendor: { type: 'string' },
              date: { type: 'string' },
              amount: { type: 'number' },
              category: { type: 'string' },
            },
            required: ['vendor', 'amount'],
          },
        }
      );
      expect(structured.data).toHaveProperty('vendor');
      expect(structured.data).toHaveProperty('amount');
      records.push(structured.data);
    }

    // Step 4: Deduplicate
    const deduped = await harness.callTool(
      'data', 'deduplicate_records', {
        data: records,
        match_fields: ['vendor', 'amount', 'date'],
        threshold: 0.85,
      }
    );
    expect(deduped.unique.length).toBeGreaterThanOrEqual(1);

    // Step 5: Write CSV
    const csvPath = harness.tempPath('receipts_output.csv');
    const csv = await harness.callTool(
      'data', 'write_csv', {
        data: deduped.unique,
        output_path: csvPath,
      }
    );
    expect(csv.rows).toBe(deduped.unique.length);

    // Verify output
    const outputContent = await harness.callTool(
      'filesystem', 'read_file', { path: csvPath }
    );
    expect(outputContent.content).toContain('vendor');
    expect(outputContent.content).toContain('amount');
  });
});
```

## Example: UC-3 Security Steward

```typescript
describe('UC-3: Security & Privacy Steward', () => {
  it('should detect PII and secrets in sample files', async () => {
    const scanDir = harness.fixturePath('sample_files');

    // Step 1: Search files
    const files = await harness.callTool(
      'filesystem', 'search_files', { path: scanDir, pattern: '*' }
    );

    // Steps 2-3: Scan for PII and secrets
    const allFindings = [];
    for (const file of files) {
      const pii = await harness.callTool(
        'security', 'scan_for_pii', { path: file.path }
      );
      const secrets = await harness.callTool(
        'security', 'scan_for_secrets', { path: file.path }
      );
      allFindings.push(...pii.findings, ...secrets.findings);
    }

    // Verify: should find SSN in has_ssn.txt and API key in has_api_key.env
    expect(allFindings.some(f => f.type === 'ssn')).toBe(true);
    expect(allFindings.some(f => f.type === 'api_key')).toBe(true);

    // Step 4: Propose cleanup
    const proposals = await harness.callTool(
      'security', 'propose_cleanup', { findings: allFindings }
    );
    expect(proposals.actions.length).toBeGreaterThan(0);
  });
});
```

## Running Tests

```bash
# Run a specific use case
/test-usecase UC-1

# Run all integration tests
npx vitest run tests/integration/

# Run with verbose output
npx vitest run tests/integration/ --reporter=verbose
```

## Creating New UC Tests

When a new use case is ready for integration testing:

1. Read the UC flow in `docs/PRD.md` Section 6
2. Identify all MCP servers involved (see `/test-usecase` command for the mapping)
3. Create test fixtures in `tests/fixtures/uc<N>/`
4. Write the test following the step-by-step flow from the PRD
5. Verify each intermediate result, not just the final output
6. Include edge cases (e.g., what happens if OCR confidence is low?)
