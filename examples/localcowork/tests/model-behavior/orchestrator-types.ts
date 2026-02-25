/**
 * Type definitions and comparison utilities for the orchestrator benchmark.
 *
 * These extend the base types from types.ts with orchestrator-specific
 * structures: plan results, step execution results, and baseline comparison.
 */

import * as fs from 'fs';
import * as path from 'path';

// ─── Plan Types (mirrors Rust StepPlan / PlanStep) ─────────────────────────

export interface PlanStep {
  step_number: number;
  description: string;
  expected_server: string | null;
  hint_params: Record<string, unknown> | null;
}

export interface StepPlan {
  needs_tools: boolean;
  direct_response: string | null;
  steps: PlanStep[];
}

// ─── Orchestrator Step Result ──────────────────────────────────────────────

export type OrchestratorFailure =
  | 'plan_error' | 'wrong_tool' | 'no_tool' | 'deflection'
  | 'critical_failure' | 'filter_miss' | 'error';

export interface OrchestratorStepResult {
  stepIndex: number;
  planDescription: string;
  expectedTools: readonly string[];
  actualTools: string[];
  status: 'passed' | 'failed';
  failureReason?: OrchestratorFailure;
  filterHit: boolean;
  filteredTools: string[];
  rawContent: string;
  mockResult: string;
  durationMs: number;
  retries: number;
}

// ─── Orchestrator Chain Result ─────────────────────────────────────────────

export interface OrchestratorChainResult {
  testId: string;
  scenario: string;
  difficulty: 'easy' | 'medium' | 'hard';
  planValid: boolean;
  planStepCount: number;
  planDurationMs: number;
  planRaw: string;
  status: 'passed' | 'failed' | 'plan_error';
  toolsCovered: number;
  totalExpectedTools: number;
  stepsExecuted: number;
  stepResults: OrchestratorStepResult[];
  failureReason?: string;
  totalDurationMs: number;
}

// ─── Metrics ───────────────────────────────────────────────────────────────

export interface DifficultyStats {
  total: number;
  passed: number;
  rate: number;
}

export interface OrchestratorMetrics {
  runId: string;
  timestamp: string;
  plannerEndpoint: string;
  routerEndpoint: string;
  topK: number;
  chainCompletionRate: number;
  stepCompletionRate: number;
  toolCoverageRate: number;
  planSuccessRate: number;
  avgPlanSteps: number;
  byDifficulty: Record<string, DifficultyStats>;
  failureBreakdown: Record<string, number>;
  avgPlanTimeMs: number;
  avgStepTimeMs: number;
  filterHitRate: number;
  totalChains: number;
  passed: number;
  failed: number;
  durationMs: number;
  results: OrchestratorChainResult[];
}

// ─── Baseline Comparison ───────────────────────────────────────────────────

export interface ComparisonResult {
  baselineChainCompletion: number;
  orchestratorChainCompletion: number;
  improvementPP: number;
  baselineStepCompletion: number;
  orchestratorStepCompletion: number;
  byDifficulty: Record<string, {
    baseline: { total: number; passed: number };
    orchestrator: { total: number; passed: number };
  }>;
}

interface BaselineData {
  chainCompletionRate: number;
  stepCompletionRate: number;
  byDifficulty: Record<string, { total: number; passed: number }>;
}

/** Find the most recent multi-step baseline result file. */
export function findLatestBaseline(resultsDir: string): BaselineData | null {
  if (!fs.existsSync(resultsDir)) return null;
  const files = fs.readdirSync(resultsDir)
    .filter((f) => f.startsWith('lfm-multistep-') && f.endsWith('.json'))
    .sort()
    .reverse();
  if (files.length === 0) return null;
  const data = JSON.parse(fs.readFileSync(path.join(resultsDir, files[0]), 'utf-8'));
  return {
    chainCompletionRate: data.chainCompletionRate ?? 0,
    stepCompletionRate: data.stepCompletionRate ?? 0,
    byDifficulty: data.byDifficulty ?? {},
  };
}

/** Build a side-by-side comparison between orchestrator and baseline results. */
export function buildComparison(
  metrics: OrchestratorMetrics,
  baseline: BaselineData,
): ComparisonResult {
  const byDifficulty: ComparisonResult['byDifficulty'] = {};
  for (const [diff, orchStats] of Object.entries(metrics.byDifficulty)) {
    byDifficulty[diff] = {
      baseline: baseline.byDifficulty[diff] ?? { total: orchStats.total, passed: 0 },
      orchestrator: { total: orchStats.total, passed: orchStats.passed },
    };
  }
  return {
    baselineChainCompletion: baseline.chainCompletionRate,
    orchestratorChainCompletion: metrics.chainCompletionRate,
    improvementPP: metrics.chainCompletionRate - baseline.chainCompletionRate,
    baselineStepCompletion: baseline.stepCompletionRate,
    orchestratorStepCompletion: metrics.stepCompletionRate,
    byDifficulty,
  };
}
