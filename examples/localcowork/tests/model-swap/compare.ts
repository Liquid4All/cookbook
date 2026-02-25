/**
 * Comparison Report Generator — WS-6C
 *
 * Takes two SwapResult objects (baseline + candidate) and generates:
 * - Side-by-side accuracy comparison
 * - Regression list (tests that pass on baseline but fail on candidate)
 * - Improvement list (tests that fail on baseline but pass on candidate)
 * - Pass/fail verdict based on configurable thresholds
 */

import type { SwapResult, SuiteResult, TestCaseResult } from "./runner";

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

export interface ComparisonThresholds {
  /** Max allowed accuracy drop for tool selection (e.g. 0.05 = 5%) */
  readonly toolSelectionMaxDrop: number;
  /** Max allowed accuracy drop for chain completion (e.g. 0.10 = 10%) */
  readonly chainCompletionMaxDrop: number;
  /** Max allowed accuracy drop for edge cases (e.g. 0.05 = 5%) */
  readonly edgeCaseMaxDrop: number;
}

export interface TestDelta {
  readonly id: string;
  readonly prompt: string;
  readonly baselineResult: string | null;
  readonly candidateResult: string | null;
  readonly expected: string | null;
}

export interface SuiteComparison {
  readonly suiteName: string;
  readonly baselineAccuracy: number;
  readonly candidateAccuracy: number;
  readonly delta: number;
  readonly regressions: readonly TestDelta[];
  readonly improvements: readonly TestDelta[];
  readonly passed: boolean;
  readonly maxAllowedDrop: number;
}

export interface ComparisonReport {
  readonly baselineModel: string;
  readonly candidateModel: string;
  readonly baselineTimestamp: string;
  readonly candidateTimestamp: string;
  readonly toolSelection: SuiteComparison;
  readonly multiStep: SuiteComparison;
  readonly edgeCases: SuiteComparison;
  readonly overallVerdict: "PASS" | "FAIL";
  readonly summary: string;
}

// ---------------------------------------------------------------------------
// Default thresholds (from models.json spec)
// ---------------------------------------------------------------------------

export const DEFAULT_THRESHOLDS: ComparisonThresholds = {
  toolSelectionMaxDrop: 0.05,
  chainCompletionMaxDrop: 0.10,
  edgeCaseMaxDrop: 0.05,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildResultMap(
  results: readonly TestCaseResult[],
): ReadonlyMap<string, TestCaseResult> {
  const map = new Map<string, TestCaseResult>();
  for (const r of results) {
    map.set(r.id, r);
  }
  return map;
}

function findRegressions(
  baseline: ReadonlyMap<string, TestCaseResult>,
  candidate: ReadonlyMap<string, TestCaseResult>,
): TestDelta[] {
  const regressions: TestDelta[] = [];
  for (const [id, baseResult] of baseline) {
    const candResult = candidate.get(id);
    if (baseResult.passed && candResult && !candResult.passed) {
      regressions.push({
        id,
        prompt: baseResult.prompt,
        baselineResult: baseResult.actual,
        candidateResult: candResult.actual,
        expected: baseResult.expected,
      });
    }
  }
  return regressions;
}

function findImprovements(
  baseline: ReadonlyMap<string, TestCaseResult>,
  candidate: ReadonlyMap<string, TestCaseResult>,
): TestDelta[] {
  const improvements: TestDelta[] = [];
  for (const [id, baseResult] of baseline) {
    const candResult = candidate.get(id);
    if (!baseResult.passed && candResult && candResult.passed) {
      improvements.push({
        id,
        prompt: baseResult.prompt,
        baselineResult: baseResult.actual,
        candidateResult: candResult.actual,
        expected: baseResult.expected,
      });
    }
  }
  return improvements;
}

function compareSuite(
  suiteName: string,
  baselineSuite: SuiteResult,
  candidateSuite: SuiteResult,
  maxAllowedDrop: number,
): SuiteComparison {
  const baselineMap = buildResultMap(baselineSuite.results);
  const candidateMap = buildResultMap(candidateSuite.results);

  const regressions = findRegressions(baselineMap, candidateMap);
  const improvements = findImprovements(baselineMap, candidateMap);

  const delta = candidateSuite.accuracy - baselineSuite.accuracy;
  const passed = delta >= -maxAllowedDrop;

  return {
    suiteName,
    baselineAccuracy: baselineSuite.accuracy,
    candidateAccuracy: candidateSuite.accuracy,
    delta,
    regressions,
    improvements,
    passed,
    maxAllowedDrop,
  };
}

// ---------------------------------------------------------------------------
// Main comparison function
// ---------------------------------------------------------------------------

export function compareResults(
  baseline: SwapResult,
  candidate: SwapResult,
  thresholds: ComparisonThresholds = DEFAULT_THRESHOLDS,
): ComparisonReport {
  const toolSelection = compareSuite(
    "tool-selection",
    baseline.toolSelection,
    candidate.toolSelection,
    thresholds.toolSelectionMaxDrop,
  );

  const multiStep = compareSuite(
    "multi-step",
    baseline.multiStep,
    candidate.multiStep,
    thresholds.chainCompletionMaxDrop,
  );

  const edgeCases = compareSuite(
    "edge-cases",
    baseline.edgeCases,
    candidate.edgeCases,
    thresholds.edgeCaseMaxDrop,
  );

  const overallVerdict: "PASS" | "FAIL" =
    toolSelection.passed && multiStep.passed && edgeCases.passed
      ? "PASS"
      : "FAIL";

  const totalRegressions =
    toolSelection.regressions.length +
    multiStep.regressions.length +
    edgeCases.regressions.length;

  const totalImprovements =
    toolSelection.improvements.length +
    multiStep.improvements.length +
    edgeCases.improvements.length;

  const summary = [
    "Model Swap Comparison: " + baseline.modelName + " -> " + candidate.modelName,
    "Verdict: " + overallVerdict,
    "",
    "Tool Selection: " +
      formatPct(baseline.toolSelection.accuracy) + " -> " +
      formatPct(candidate.toolSelection.accuracy) +
      " (" + formatDelta(toolSelection.delta) + ")",
    "Multi-Step:      " +
      formatPct(baseline.multiStep.accuracy) + " -> " +
      formatPct(candidate.multiStep.accuracy) +
      " (" + formatDelta(multiStep.delta) + ")",
    "Edge Cases:      " +
      formatPct(baseline.edgeCases.accuracy) + " -> " +
      formatPct(candidate.edgeCases.accuracy) +
      " (" + formatDelta(edgeCases.delta) + ")",
    "",
    "Regressions: " + String(totalRegressions),
    "Improvements: " + String(totalImprovements),
  ].join("\n");

  return {
    baselineModel: baseline.modelName,
    candidateModel: candidate.modelName,
    baselineTimestamp: baseline.timestamp,
    candidateTimestamp: candidate.timestamp,
    toolSelection,
    multiStep,
    edgeCases,
    overallVerdict,
    summary,
  };
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function formatPct(value: number): string {
  return (value * 100).toFixed(1) + "%";
}

function formatDelta(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return sign + (value * 100).toFixed(1) + "%";
}

// ---------------------------------------------------------------------------
// Markdown report generator
// ---------------------------------------------------------------------------

export function generateMarkdownReport(report: ComparisonReport): string {
  const lines: string[] = [];

  lines.push("# Model Swap Comparison Report");
  lines.push("");
  lines.push("| Property | Value |");
  lines.push("|----------|-------|");
  lines.push("| Baseline Model | " + report.baselineModel + " |");
  lines.push("| Candidate Model | " + report.candidateModel + " |");
  lines.push("| Baseline Timestamp | " + report.baselineTimestamp + " |");
  lines.push("| Candidate Timestamp | " + report.candidateTimestamp + " |");
  lines.push("| **Verdict** | **" + report.overallVerdict + "** |");
  lines.push("");

  lines.push("## Accuracy Comparison");
  lines.push("");
  lines.push("| Suite | Baseline | Candidate | Delta | Max Drop | Status |");
  lines.push("|-------|----------|-----------|-------|----------|--------|");

  for (const suite of [report.toolSelection, report.multiStep, report.edgeCases]) {
    const status = suite.passed ? "PASS" : "FAIL";
    lines.push(
      "| " + suite.suiteName +
      " | " + formatPct(suite.baselineAccuracy) +
      " | " + formatPct(suite.candidateAccuracy) +
      " | " + formatDelta(suite.delta) +
      " | " + formatPct(suite.maxAllowedDrop) +
      " | " + status + " |",
    );
  }
  lines.push("");

  // Regressions
  const allRegressions = [
    ...report.toolSelection.regressions,
    ...report.multiStep.regressions,
    ...report.edgeCases.regressions,
  ];
  if (allRegressions.length > 0) {
    lines.push("## Regressions (" + String(allRegressions.length) + ")");
    lines.push("");
    lines.push("| ID | Prompt | Expected | Baseline | Candidate |");
    lines.push("|----|--------|----------|----------|----------|");
    for (const r of allRegressions) {
      const promptShort = r.prompt.length > 60 ? r.prompt.substring(0, 57) + "..." : r.prompt;
      lines.push(
        "| " + r.id +
        " | " + promptShort +
        " | " + String(r.expected) +
        " | " + String(r.baselineResult) +
        " | " + String(r.candidateResult) + " |",
      );
    }
    lines.push("");
  }

  // Improvements
  const allImprovements = [
    ...report.toolSelection.improvements,
    ...report.multiStep.improvements,
    ...report.edgeCases.improvements,
  ];
  if (allImprovements.length > 0) {
    lines.push("## Improvements (" + String(allImprovements.length) + ")");
    lines.push("");
    lines.push("| ID | Prompt | Expected | Baseline | Candidate |");
    lines.push("|----|--------|----------|----------|----------|");
    for (const r of allImprovements) {
      const promptShort = r.prompt.length > 60 ? r.prompt.substring(0, 57) + "..." : r.prompt;
      lines.push(
        "| " + r.id +
        " | " + promptShort +
        " | " + String(r.expected) +
        " | " + String(r.baselineResult) +
        " | " + String(r.candidateResult) + " |",
      );
    }
    lines.push("");
  }

  return lines.join("\n");
}
