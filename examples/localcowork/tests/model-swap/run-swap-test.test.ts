/**
 * Model Swap Test Runner — WS-6C
 *
 * Vitest-compatible test file that:
 * - Reads model config from models.json
 * - If LOCALCOWORK_MODEL_ENDPOINT is set, runs live model tests
 * - If not, runs structural validation only (checks test definitions are valid)
 * - Creates results file in tests/model-swap/.results/
 */

import { describe, it, expect } from "vitest";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  validateTestDefinitions,
  loadToolSelectionTests,
  loadMultiStepTests,
  loadEdgeCaseTests,
  runModelSwap,
} from "./runner";
import type { ModelSwapConfig, SwapResult } from "./runner";
import { compareResults, generateMarkdownReport } from "./compare";
import type { ComparisonThresholds } from "./compare";

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

const currentDir = dirname(fileURLToPath(import.meta.url));

function modelsConfigPath(): string {
  return resolve(currentDir, "models.json");
}

function resultsDir(): string {
  return resolve(currentDir, ".results");
}

function systemPromptsDir(): string {
  return resolve(currentDir, "system-prompts");
}

// ---------------------------------------------------------------------------
// Config loader
// ---------------------------------------------------------------------------

interface ModelEntry {
  readonly name: string;
  readonly endpoint: string;
  readonly systemPromptFile: string;
  readonly temperature: number;
  readonly notes: string;
}

interface ModelsConfig {
  readonly models: readonly ModelEntry[];
  readonly thresholds: {
    readonly tool_selection_min_accuracy: number;
    readonly chain_completion_min_rate: number;
    readonly edge_case_min_accuracy: number;
    readonly max_regression_from_baseline: number;
  };
}

function loadModelsConfig(): ModelsConfig {
  const raw = readFileSync(modelsConfigPath(), "utf-8");
  return JSON.parse(raw) as ModelsConfig;
}

function loadSystemPrompt(promptFile: string): string {
  const filePath = resolve(systemPromptsDir(), promptFile.replace("system-prompts/", ""));
  return readFileSync(filePath, "utf-8");
}

// ---------------------------------------------------------------------------
// Known valid tool names (all 68 MCP tools)
// ---------------------------------------------------------------------------

const VALID_TOOL_NAMES: ReadonlySet<string> = new Set([
  "filesystem.list_dir", "filesystem.read_file", "filesystem.write_file",
  "filesystem.move_file", "filesystem.copy_file", "filesystem.delete_file",
  "filesystem.search_files", "filesystem.get_metadata", "filesystem.watch_folder",
  "document.extract_text", "document.convert_format", "document.diff_documents",
  "document.create_pdf", "document.fill_pdf_form", "document.merge_pdfs",
  "document.create_docx", "document.read_spreadsheet",
  "ocr.extract_text_from_image", "ocr.extract_text_from_pdf",
  "ocr.extract_structured_data", "ocr.extract_table",
  "data.write_csv", "data.write_sqlite", "data.query_sqlite",
  "data.deduplicate_records", "data.summarize_anomalies",
  "audit.get_tool_log", "audit.get_session_summary",
  "audit.generate_audit_report", "audit.export_audit_pdf",
  "knowledge.index_folder", "knowledge.search_documents",
  "knowledge.ask_about_files", "knowledge.update_index",
  "knowledge.get_related_chunks",
  "security.scan_for_pii", "security.scan_for_secrets",
  "security.find_duplicates", "security.propose_cleanup",
  "security.encrypt_file", "security.decrypt_file",
  "task.create_task", "task.list_tasks", "task.update_task",
  "task.get_overdue", "task.daily_briefing",
  "calendar.list_events", "calendar.create_event",
  "calendar.find_free_slots", "calendar.create_time_block",
  "email.draft_email", "email.list_drafts", "email.search_emails",
  "email.summarize_thread", "email.send_draft",
  "meeting.transcribe_audio", "meeting.extract_action_items",
  "meeting.extract_commitments", "meeting.generate_minutes",
  "clipboard.get_clipboard", "clipboard.set_clipboard",
  "clipboard.clipboard_history",
  "system.get_system_info", "system.open_application",
  "system.take_screenshot", "system.list_processes",
  "system.open_file_with",
]);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Model Swap Validation Framework", () => {

  // =========================================================================
  // Structural validation — always runs (no model needed)
  // =========================================================================

  describe("Structural Validation", () => {

    it("should have a valid models.json configuration", () => {
      const config = loadModelsConfig();
      expect(config.models).toBeDefined();
      expect(config.models.length).toBeGreaterThanOrEqual(2);
      expect(config.thresholds).toBeDefined();
      expect(config.thresholds.tool_selection_min_accuracy).toBeGreaterThan(0);
      expect(config.thresholds.chain_completion_min_rate).toBeGreaterThan(0);
      expect(config.thresholds.edge_case_min_accuracy).toBeGreaterThan(0);
    });

    it("should have system prompt files for all configured models", () => {
      const config = loadModelsConfig();
      for (const model of config.models) {
        const promptPath = resolve(
          systemPromptsDir(),
          model.systemPromptFile.replace("system-prompts/", ""),
        );
        expect(
          existsSync(promptPath),
          "Missing system prompt: " + promptPath,
        ).toBe(true);

        const content = readFileSync(promptPath, "utf-8");
        expect(content.length).toBeGreaterThan(100);
      }
    });

    it("should pass test definition validation", () => {
      const result = validateTestDefinitions();
      expect(result.errors).toEqual([]);
      expect(result.valid).toBe(true);
    });

    it("should have exactly 100 tool-selection tests", () => {
      const suite = loadToolSelectionTests();
      expect(suite.tests.length).toBe(100);
    });

    it("should have exactly 50 multi-step tests", () => {
      const suite = loadMultiStepTests();
      expect(suite.tests.length).toBe(50);
    });

    it("should have exactly 30 edge-case tests", () => {
      const suite = loadEdgeCaseTests();
      expect(suite.tests.length).toBe(30);
    });

    it("should have unique test IDs across all suites", () => {
      const ts = loadToolSelectionTests();
      const ms = loadMultiStepTests();
      const ec = loadEdgeCaseTests();

      const allIds = [
        ...ts.tests.map((t) => t.id),
        ...ms.tests.map((t) => t.id),
        ...ec.tests.map((t) => t.id),
      ];

      const uniqueIds = new Set(allIds);
      expect(uniqueIds.size).toBe(allIds.length);
    });

    it("should reference only valid tool names in tool-selection tests", () => {
      const suite = loadToolSelectionTests();
      const invalidTools: string[] = [];

      for (const test of suite.tests) {
        if (!VALID_TOOL_NAMES.has(test.expectedTool)) {
          invalidTools.push(test.id + ": " + test.expectedTool);
        }
      }

      expect(
        invalidTools,
        "Invalid tool names found: " + invalidTools.join(", "),
      ).toEqual([]);
    });

    it("should reference only valid tool names in multi-step test chains", () => {
      const suite = loadMultiStepTests();
      const invalidTools: string[] = [];

      for (const test of suite.tests) {
        for (const tool of test.expectedChain) {
          if (!VALID_TOOL_NAMES.has(tool)) {
            invalidTools.push(test.id + ": " + tool);
          }
        }
      }

      expect(
        invalidTools,
        "Invalid tool names found: " + invalidTools.join(", "),
      ).toEqual([]);
    });

    it("should reference only valid tool names in edge-case tests (when expectedTool is set)", () => {
      const suite = loadEdgeCaseTests();
      const invalidTools: string[] = [];

      for (const test of suite.tests) {
        if (test.expectedTool !== null && !VALID_TOOL_NAMES.has(test.expectedTool)) {
          invalidTools.push(test.id + ": " + test.expectedTool);
        }
      }

      expect(
        invalidTools,
        "Invalid tool names found: " + invalidTools.join(", "),
      ).toEqual([]);
    });

    it("should cover all 13 servers in tool-selection tests", () => {
      const suite = loadToolSelectionTests();
      const servers = new Set<string>();

      for (const test of suite.tests) {
        const server = test.expectedTool.split(".")[0];
        servers.add(server);
      }

      const expectedServers = [
        "filesystem", "document", "ocr", "data", "audit",
        "knowledge", "security", "task", "calendar", "email",
        "meeting", "clipboard", "system",
      ];

      for (const server of expectedServers) {
        expect(
          servers.has(server),
          "Missing server coverage: " + server,
        ).toBe(true);
      }
    });

    it("should have valid edge-case expectedBehavior values", () => {
      const suite = loadEdgeCaseTests();
      const validBehaviors = new Set([
        "tool_call",
        "ask_clarification",
        "explain_limitation",
        "conversational_response",
      ]);

      for (const test of suite.tests) {
        expect(
          validBehaviors.has(test.expectedBehavior),
          "Invalid expectedBehavior in " + test.id + ": " + test.expectedBehavior,
        ).toBe(true);
      }
    });

    it("should have a results directory with .gitkeep", () => {
      expect(existsSync(resolve(resultsDir(), ".gitkeep"))).toBe(true);
    });
  });

  // =========================================================================
  // Comparison logic validation (uses synthetic data, no model needed)
  // =========================================================================

  describe("Comparison Logic", () => {

    function makeSyntheticResult(
      modelName: string,
      tsAccuracy: number,
      msAccuracy: number,
      ecAccuracy: number,
    ): SwapResult {
      const makeResults = (
        prefix: string,
        count: number,
        accuracy: number,
      ) => {
        const passCount = Math.round(count * accuracy);
        const results = [];
        for (let i = 0; i < count; i++) {
          results.push({
            id: prefix + "-" + String(i + 1).padStart(3, "0"),
            prompt: "Test prompt " + String(i + 1),
            expected: "expected.tool",
            actual: i < passCount ? "expected.tool" : "wrong.tool",
            passed: i < passCount,
          });
        }
        return results;
      };

      const tsResults = makeResults("ts", 100, tsAccuracy);
      const msResults = makeResults("ms", 50, msAccuracy);
      const ecResults = makeResults("ec", 30, ecAccuracy);

      return {
        modelName,
        timestamp: new Date().toISOString(),
        toolSelection: {
          total: 100,
          passed: tsResults.filter((r) => r.passed).length,
          accuracy: tsAccuracy,
          results: tsResults,
        },
        multiStep: {
          total: 50,
          passed: msResults.filter((r) => r.passed).length,
          accuracy: msAccuracy,
          results: msResults,
        },
        edgeCases: {
          total: 30,
          passed: ecResults.filter((r) => r.passed).length,
          accuracy: ecAccuracy,
          results: ecResults,
        },
        overall: {
          total: 180,
          passed: tsResults.filter((r) => r.passed).length +
            msResults.filter((r) => r.passed).length +
            ecResults.filter((r) => r.passed).length,
          accuracy:
            (tsAccuracy * 100 + msAccuracy * 50 + ecAccuracy * 30) / 180,
        },
      };
    }

    it("should produce PASS when candidate meets thresholds", () => {
      const baseline = makeSyntheticResult("baseline", 0.95, 0.90, 0.90);
      const candidate = makeSyntheticResult("candidate", 0.93, 0.85, 0.88);

      const report = compareResults(baseline, candidate);
      expect(report.overallVerdict).toBe("PASS");
    });

    it("should produce FAIL when tool selection drops too much", () => {
      const baseline = makeSyntheticResult("baseline", 0.95, 0.90, 0.90);
      const candidate = makeSyntheticResult("candidate", 0.85, 0.88, 0.88);

      const report = compareResults(baseline, candidate);
      expect(report.overallVerdict).toBe("FAIL");
      expect(report.toolSelection.passed).toBe(false);
    });

    it("should produce FAIL when chain completion drops too much", () => {
      const baseline = makeSyntheticResult("baseline", 0.95, 0.90, 0.90);
      const candidate = makeSyntheticResult("candidate", 0.93, 0.70, 0.88);

      const report = compareResults(baseline, candidate);
      expect(report.overallVerdict).toBe("FAIL");
      expect(report.multiStep.passed).toBe(false);
    });

    it("should detect regressions between baseline and candidate", () => {
      const baseline = makeSyntheticResult("baseline", 1.0, 1.0, 1.0);
      const candidate = makeSyntheticResult("candidate", 0.90, 0.90, 0.90);

      const report = compareResults(baseline, candidate);
      const totalRegressions =
        report.toolSelection.regressions.length +
        report.multiStep.regressions.length +
        report.edgeCases.regressions.length;

      expect(totalRegressions).toBeGreaterThan(0);
    });

    it("should detect improvements between baseline and candidate", () => {
      const baseline = makeSyntheticResult("baseline", 0.80, 0.80, 0.80);
      const candidate = makeSyntheticResult("candidate", 0.90, 0.90, 0.90);

      const report = compareResults(baseline, candidate);
      const totalImprovements =
        report.toolSelection.improvements.length +
        report.multiStep.improvements.length +
        report.edgeCases.improvements.length;

      expect(totalImprovements).toBeGreaterThan(0);
    });

    it("should generate a valid markdown report", () => {
      const baseline = makeSyntheticResult("qwen2.5-32b", 0.95, 0.90, 0.90);
      const candidate = makeSyntheticResult("lfm2.5-24b", 0.92, 0.85, 0.87);

      const report = compareResults(baseline, candidate);
      const markdown = generateMarkdownReport(report);

      expect(markdown).toContain("# Model Swap Comparison Report");
      expect(markdown).toContain("qwen2.5-32b");
      expect(markdown).toContain("lfm2.5-24b");
      expect(markdown).toContain("Accuracy Comparison");
    });

    it("should respect custom thresholds", () => {
      const baseline = makeSyntheticResult("baseline", 0.95, 0.90, 0.90);
      const candidate = makeSyntheticResult("candidate", 0.93, 0.85, 0.88);

      // With very tight thresholds, this should fail
      const tightThresholds: ComparisonThresholds = {
        toolSelectionMaxDrop: 0.01,
        chainCompletionMaxDrop: 0.01,
        edgeCaseMaxDrop: 0.01,
      };

      const report = compareResults(baseline, candidate, tightThresholds);
      expect(report.overallVerdict).toBe("FAIL");
    });
  });

  // =========================================================================
  // Live model tests (only when LOCALCOWORK_MODEL_ENDPOINT is set)
  // =========================================================================

  describe("Live Model Tests", () => {
    const endpoint = process.env["LOCALCOWORK_MODEL_ENDPOINT"];
    const modelName = process.env["LOCALCOWORK_MODEL_NAME"] ?? "unknown";

    if (!endpoint) {
      it.skip("skipped — LOCALCOWORK_MODEL_ENDPOINT not set", () => {
        // This test is intentionally skipped when no model endpoint is available.
        // Set LOCALCOWORK_MODEL_ENDPOINT to run live model validation.
      });
      return;
    }

    it("should run the full model swap test suite against " + modelName, async () => {
      const config = loadModelsConfig();
      const modelEntry = config.models.find((m) => m.name === modelName);

      let systemPrompt: string;
      if (modelEntry) {
        systemPrompt = loadSystemPrompt(modelEntry.systemPromptFile);
      } else {
        // Use the Qwen prompt as default
        systemPrompt = loadSystemPrompt("qwen-system-prompt.txt");
      }

      const swapConfig: ModelSwapConfig = {
        modelName,
        endpoint,
        temperature: 0.1,
        systemPrompt,
      };

      const result = await runModelSwap(swapConfig);

      // Write results to file
      const resultsPath = resolve(
        resultsDir(),
        modelName.replace(/[^a-zA-Z0-9.-]/g, "_") + "_" +
          new Date().toISOString().replace(/[:.]/g, "-") + ".json",
      );
      writeFileSync(resultsPath, JSON.stringify(result, null, 2), "utf-8");

      // Validate minimum thresholds from config
      expect(result.toolSelection.accuracy).toBeGreaterThanOrEqual(
        config.thresholds.tool_selection_min_accuracy,
      );
      expect(result.multiStep.accuracy).toBeGreaterThanOrEqual(
        config.thresholds.chain_completion_min_rate,
      );
      expect(result.edgeCases.accuracy).toBeGreaterThanOrEqual(
        config.thresholds.edge_case_min_accuracy,
      );
    }, 600_000); // 10 minute timeout for full suite
  });
});
