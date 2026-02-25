#!/usr/bin/env npx tsx
/**
 * Orchestrator Benchmark — evaluates the dual-model pipeline (ADR-009)
 * against the same 50 multi-step chain scenarios.
 *
 * Planner (Qwen3-30B-A3B via Ollama) decomposes → Router (LFM2-1.2B-Tool)
 * executes each step independently with RAG pre-filtered tools.
 *
 * Usage:
 *   npx tsx tests/model-behavior/benchmark-orchestrator.ts
 *   npx tsx tests/model-behavior/benchmark-orchestrator.ts --difficulty easy
 *   npx tsx tests/model-behavior/benchmark-orchestrator.ts \
 *     --planner-endpoint http://localhost:11434/v1 \
 *     --router-endpoint http://localhost:8082 \
 *     --top-k 15 --step-retries 3
 */

import type { MultiStepTest, MultiStepEntry } from './types';
import type {
  OrchestratorChainResult,
  OrchestratorMetrics,
} from './orchestrator-types';
import { findLatestBaseline, buildComparison } from './orchestrator-types';
import {
  buildToolDefinitions,
  buildToolEmbeddingIndex,
} from './benchmark-shared';
import type { ToolEmbeddingIndex } from './benchmark-shared';
import { callPlanner, validatePlan, mapPlanToExpectedSteps } from './orchestrator-planner';
import { executeAllSteps } from './orchestrator-executor';
import {
  allMultiStepTests,
  simpleChainTests,
  mediumChainTests,
  complexChainTests,
} from './multi-step-chains';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ─── CLI Arguments ──────────────────────────────────────────────────────────

const args = process.argv.slice(2);

function getArg(name: string): string | undefined {
  const idx = args.indexOf(`--${name}`);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : undefined;
}

const PLANNER_ENDPOINT = getArg('planner-endpoint') ?? 'http://localhost:11434';
const PLANNER_MODEL = getArg('planner-model') ?? 'qwen3:30b-a3b';
const ROUTER_ENDPOINT = getArg('router-endpoint') ?? 'http://localhost:8082';
const ROUTER_MODEL = getArg('router-model') ?? '';
const TOP_K = parseInt(getArg('top-k') ?? '15', 10);
const DIFFICULTY = getArg('difficulty') as 'easy' | 'medium' | 'hard' | 'all' | undefined;
const MAX_PLAN_STEPS = parseInt(getArg('max-plan-steps') ?? '10', 10);
const STEP_RETRIES = parseInt(getArg('step-retries') ?? '3', 10);

// ─── Test Selection ─────────────────────────────────────────────────────────

function selectTests(): readonly MultiStepTest[] {
  switch (DIFFICULTY) {
    case 'easy': return simpleChainTests;
    case 'medium': return mediumChainTests;
    case 'hard': return complexChainTests;
    default: return allMultiStepTests;
  }
}

function mapDifficulty(d: 'easy' | 'medium' | 'hard'): 'easy' | 'medium' | 'hard' {
  return d;
}

// ─── Health Check ───────────────────────────────────────────────────────────

async function healthCheck(endpoint: string, label: string): Promise<boolean> {
  try {
    const response = await fetch(`${endpoint}/v1/models`);
    if (!response.ok) {
      console.error(`  ✗ ${label} health check failed: HTTP ${response.status}`);
      return false;
    }
    console.log(`  ✓ ${label} is ready`);
    return true;
  } catch (e) {
    console.error(`  ✗ ${label} health check failed: ${e}`);
    return false;
  }
}

// ─── Bag-of-Tools Evaluation ────────────────────────────────────────────────

/**
 * Evaluate whether a chain passed using bag-of-tools matching.
 *
 * A chain passes if every expected tool from ALL test steps appears
 * somewhere in the actual tools called across all orchestrated steps.
 */
function evaluateChain(
  testSteps: readonly MultiStepEntry[],
  stepResults: Array<{ actualTools: string[] }>,
): { passed: boolean; covered: number; total: number } {
  const allExpected = new Set<string>();
  for (const step of testSteps) {
    for (const tool of step.expectedTools) allExpected.add(tool);
  }

  const allActual = new Set<string>();
  for (const result of stepResults) {
    for (const tool of result.actualTools) allActual.add(tool);
  }

  let covered = 0;
  for (const tool of allExpected) {
    if (allActual.has(tool)) covered++;
  }

  return {
    passed: covered === allExpected.size,
    covered,
    total: allExpected.size,
  };
}

// ─── Metrics Computation ────────────────────────────────────────────────────

function computeMetrics(
  results: OrchestratorChainResult[],
  durationMs: number,
): OrchestratorMetrics {
  const passed = results.filter((r) => r.status === 'passed').length;
  const planErrors = results.filter((r) => r.status === 'plan_error').length;
  const validPlans = results.filter((r) => r.planValid).length;

  // Step-level aggregation
  let totalSteps = 0;
  let passedSteps = 0;
  let totalFilterHits = 0;
  let totalStepCount = 0;
  let totalPlanTime = 0;
  let totalStepTime = 0;
  let totalToolsCovered = 0;
  let totalExpectedTools = 0;

  const failureBreakdown: Record<string, number> = {};
  const diffBuckets: Record<string, { total: number; passed: number }> = {};

  for (const chain of results) {
    totalPlanTime += chain.planDurationMs;
    totalToolsCovered += chain.toolsCovered;
    totalExpectedTools += chain.totalExpectedTools;

    // Per-difficulty bucket
    const diff = chain.difficulty;
    if (!diffBuckets[diff]) diffBuckets[diff] = { total: 0, passed: 0 };
    diffBuckets[diff].total++;
    if (chain.status === 'passed') diffBuckets[diff].passed++;

    // Failure tracking
    if (chain.failureReason) {
      failureBreakdown[chain.failureReason] = (failureBreakdown[chain.failureReason] ?? 0) + 1;
    }

    for (const step of chain.stepResults) {
      totalSteps++;
      totalStepTime += step.durationMs;
      if (step.status === 'passed') passedSteps++;
      if (step.filterHit) totalFilterHits++;
      totalStepCount++;

      if (step.failureReason) {
        failureBreakdown[step.failureReason] = (failureBreakdown[step.failureReason] ?? 0) + 1;
      }
    }
  }

  const byDifficulty: OrchestratorMetrics['byDifficulty'] = {};
  for (const [diff, bucket] of Object.entries(diffBuckets)) {
    byDifficulty[diff] = {
      total: bucket.total,
      passed: bucket.passed,
      rate: bucket.total > 0 ? bucket.passed / bucket.total : 0,
    };
  }

  const avgPlanSteps = validPlans > 0
    ? results.filter((r) => r.planValid).reduce((s, r) => s + r.planStepCount, 0) / validPlans
    : 0;

  return {
    runId: `orchestrator-${Date.now()}`,
    timestamp: new Date().toISOString(),
    plannerEndpoint: PLANNER_ENDPOINT,
    routerEndpoint: ROUTER_ENDPOINT,
    topK: TOP_K,
    chainCompletionRate: results.length > 0 ? passed / results.length : 0,
    stepCompletionRate: totalSteps > 0 ? passedSteps / totalSteps : 0,
    toolCoverageRate: totalExpectedTools > 0 ? totalToolsCovered / totalExpectedTools : 0,
    planSuccessRate: results.length > 0 ? validPlans / results.length : 0,
    avgPlanSteps,
    byDifficulty,
    failureBreakdown,
    avgPlanTimeMs: results.length > 0 ? totalPlanTime / results.length : 0,
    avgStepTimeMs: totalStepCount > 0 ? totalStepTime / totalStepCount : 0,
    filterHitRate: totalStepCount > 0 ? totalFilterHits / totalStepCount : 0,
    totalChains: results.length,
    passed,
    failed: results.length - passed,
    durationMs,
    results,
  };
}

// ─── Output Formatting ──────────────────────────────────────────────────────

function pct(n: number): string { return `${(n * 100).toFixed(1)}%`; }

function printMetrics(metrics: OrchestratorMetrics): void {
  console.log('\n' + '═'.repeat(66));
  console.log('  ORCHESTRATOR BENCHMARK RESULTS');
  console.log('═'.repeat(66));
  console.log(`  Planner: ${metrics.plannerEndpoint}`);
  console.log(`  Router:  ${metrics.routerEndpoint}`);
  console.log(`  Top-K:   ${metrics.topK}   Retries: ${STEP_RETRIES}`);
  console.log('─'.repeat(66));
  console.log(`  Chain Completion:  ${pct(metrics.chainCompletionRate)}  (${metrics.passed}/${metrics.totalChains})`);
  console.log(`  Step Completion:   ${pct(metrics.stepCompletionRate)}`);
  console.log(`  Tool Coverage:     ${pct(metrics.toolCoverageRate)}`);
  console.log(`  Plan Success:      ${pct(metrics.planSuccessRate)}`);
  console.log(`  Avg Plan Steps:    ${metrics.avgPlanSteps.toFixed(1)}`);
  console.log(`  Filter Hit Rate:   ${pct(metrics.filterHitRate)}`);
  console.log('─'.repeat(66));
  console.log('  BY DIFFICULTY:');
  for (const [diff, stats] of Object.entries(metrics.byDifficulty)) {
    console.log(`    ${diff.padEnd(8)} ${pct(stats.rate)}  (${stats.passed}/${stats.total})`);
  }
  console.log('─'.repeat(66));
  console.log('  FAILURE BREAKDOWN:');
  for (const [reason, count] of Object.entries(metrics.failureBreakdown)) {
    console.log(`    ${reason.padEnd(20)} ${count}`);
  }
  console.log('─'.repeat(66));
  console.log(`  Avg Plan Time:  ${metrics.avgPlanTimeMs.toFixed(0)}ms`);
  console.log(`  Avg Step Time:  ${metrics.avgStepTimeMs.toFixed(0)}ms`);
  console.log(`  Total Duration: ${(metrics.durationMs / 1000).toFixed(1)}s`);
  console.log('═'.repeat(66));
}

function printComparison(metrics: OrchestratorMetrics, resultsDir: string): void {
  const baseline = findLatestBaseline(resultsDir);
  if (!baseline) {
    console.log('\n  ⚠ No baseline found — run benchmark-multi-step.ts first for comparison\n');
    return;
  }

  const comparison = buildComparison(metrics, baseline);
  const delta = comparison.improvementPP;
  const sign = delta >= 0 ? '+' : '';

  console.log('\n' + '═'.repeat(66));
  console.log('  ORCHESTRATOR vs. RAW LLM COMPARISON');
  console.log('═'.repeat(66));
  console.log('                      RAW LLM     ORCHESTRATOR    DELTA');
  console.log(`  Chain Completion:   ${pct(comparison.baselineChainCompletion).padEnd(12)}${pct(comparison.orchestratorChainCompletion).padEnd(16)}${sign}${(delta * 100).toFixed(1)}pp`);
  console.log(`  Step Completion:    ${pct(comparison.baselineStepCompletion).padEnd(12)}${pct(comparison.orchestratorStepCompletion)}`);
  console.log('  ' + '─'.repeat(62));
  console.log('  BY DIFFICULTY:');
  for (const [diff, data] of Object.entries(comparison.byDifficulty)) {
    const bRate = data.baseline.total > 0 ? data.baseline.passed / data.baseline.total : 0;
    const oRate = data.orchestrator.total > 0 ? data.orchestrator.passed / data.orchestrator.total : 0;
    const dPP = oRate - bRate;
    const dSign = dPP >= 0 ? '+' : '';
    console.log(`    ${diff.padEnd(8)} ${pct(bRate)} (${data.baseline.passed}/${data.baseline.total})`.padEnd(28)
      + `${pct(oRate)} (${data.orchestrator.passed}/${data.orchestrator.total})`.padEnd(20)
      + `${dSign}${(dPP * 100).toFixed(1)}pp`);
  }
  console.log('═'.repeat(66));
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log('\n🔬 Orchestrator Benchmark — Dual-Model Pipeline (ADR-009)\n');

  // Health checks
  console.log('Checking model endpoints...');
  const [plannerOk, routerOk] = await Promise.all([
    healthCheck(PLANNER_ENDPOINT, 'Planner (Qwen3-30B-A3B)'),
    healthCheck(ROUTER_ENDPOINT, 'Router (LFM2-1.2B-Tool)'),
  ]);

  if (!plannerOk || !routerOk) {
    console.error('\n✗ One or more model endpoints are unavailable. Aborting.');
    process.exit(1);
  }

  // Build tool embedding index for RAG pre-filter
  console.log('\nBuilding tool embedding index...');
  const toolDefs = buildToolDefinitions();
  const toolIndex: ToolEmbeddingIndex = await buildToolEmbeddingIndex(
    ROUTER_ENDPOINT, toolDefs,
  );
  console.log(`  ✓ Indexed ${toolDefs.length} tools\n`);

  // Select tests
  const tests = selectTests();
  const diffLabel = DIFFICULTY ?? 'all';
  console.log(`Running ${tests.length} chain tests (difficulty: ${diffLabel})...\n`);

  const globalStart = Date.now();
  const results: OrchestratorChainResult[] = [];

  for (let i = 0; i < tests.length; i++) {
    const test = tests[i];
    const chainStart = Date.now();
    const prefix = `[${i + 1}/${tests.length}]`;

    // Phase 1: Plan
    const planResult = await callPlanner(PLANNER_ENDPOINT, PLANNER_MODEL, test.scenario);

    if (!planResult.plan) {
      console.log(`${prefix} ✗ PLAN_ERROR  ${test.id} — ${planResult.error ?? 'null plan'}`);
      results.push({
        testId: test.id,
        scenario: test.scenario,
        difficulty: mapDifficulty(test.difficulty),
        planValid: false,
        planStepCount: 0,
        planDurationMs: planResult.durationMs,
        planRaw: planResult.rawResponse.slice(0, 500),
        status: 'plan_error',
        toolsCovered: 0,
        totalExpectedTools: test.steps.reduce((s, st) => s + st.expectedTools.length, 0),
        stepsExecuted: 0,
        stepResults: [],
        failureReason: planResult.error,
        totalDurationMs: Date.now() - chainStart,
      });
      continue;
    }

    const validation = validatePlan(planResult.plan, MAX_PLAN_STEPS);
    if (!validation.valid) {
      console.log(`${prefix} ✗ PLAN_INVALID ${test.id} — ${validation.reason}`);
      results.push({
        testId: test.id,
        scenario: test.scenario,
        difficulty: mapDifficulty(test.difficulty),
        planValid: false,
        planStepCount: planResult.plan.steps?.length ?? 0,
        planDurationMs: planResult.durationMs,
        planRaw: planResult.rawResponse.slice(0, 500),
        status: 'plan_error',
        toolsCovered: 0,
        totalExpectedTools: test.steps.reduce((s, st) => s + st.expectedTools.length, 0),
        stepsExecuted: 0,
        stepResults: [],
        failureReason: `plan_invalid: ${validation.reason}`,
        totalDurationMs: Date.now() - chainStart,
      });
      continue;
    }

    // Phase 2: Map plan to expected steps
    const mapping = mapPlanToExpectedSteps(planResult.plan, test.steps);

    // Phase 3: Execute all steps
    const stepResults = await executeAllSteps(
      planResult.plan, test.steps, mapping,
      ROUTER_ENDPOINT, ROUTER_MODEL, toolIndex, TOP_K, STEP_RETRIES,
    );

    // Phase 4: Evaluate with bag-of-tools
    const evaluation = evaluateChain(test.steps, stepResults);
    const status = evaluation.passed ? 'passed' : 'failed';
    const icon = status === 'passed' ? '✓' : '✗';

    const chainResult: OrchestratorChainResult = {
      testId: test.id,
      scenario: test.scenario,
      difficulty: mapDifficulty(test.difficulty),
      planValid: true,
      planStepCount: planResult.plan.steps.length,
      planDurationMs: planResult.durationMs,
      planRaw: planResult.rawResponse.slice(0, 500),
      status,
      toolsCovered: evaluation.covered,
      totalExpectedTools: evaluation.total,
      stepsExecuted: stepResults.length,
      stepResults,
      totalDurationMs: Date.now() - chainStart,
    };

    if (status === 'failed') {
      const failedStep = stepResults.find((s) => s.status === 'failed');
      chainResult.failureReason = failedStep?.failureReason ?? 'incomplete_coverage';
    }

    results.push(chainResult);

    const coverageStr = `${evaluation.covered}/${evaluation.total} tools`;
    const planStr = `${planResult.plan.steps.length} steps`;
    console.log(`${prefix} ${icon} ${test.id.padEnd(20)} ${status.padEnd(8)} ${coverageStr.padEnd(14)} plan: ${planStr}  (${((Date.now() - chainStart) / 1000).toFixed(1)}s)`);
  }

  const totalDuration = Date.now() - globalStart;
  const metrics = computeMetrics(results, totalDuration);

  // Output
  printMetrics(metrics);

  // Save results
  const resultsDir = path.join(__dirname, '.results');
  if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });
  const filename = `orchestrator-${diffLabel}-${Date.now()}.json`;
  const outputPath = path.join(resultsDir, filename);
  fs.writeFileSync(outputPath, JSON.stringify(metrics, null, 2));
  console.log(`\n  Results saved to: ${outputPath}`);

  // Comparison
  printComparison(metrics, resultsDir);

  // Exit code: non-zero if below threshold
  if (metrics.chainCompletionRate < 0.10) {
    console.log('\n  ⚠ Chain completion below 10% threshold\n');
  }
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
