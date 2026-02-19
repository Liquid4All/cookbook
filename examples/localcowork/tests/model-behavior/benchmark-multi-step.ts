#!/usr/bin/env npx tsx
/**
 * Multi-Step Chain Benchmark for LFM models.
 *
 * Evaluates whether the model can execute multi-turn tool chains by
 * simulating conversation context across steps. For each MultiStepTest,
 * sends step prompts sequentially with conversation history, checks if
 * the model calls the expected tool at each step.
 *
 * Usage:
 *   npx tsx tests/model-behavior/benchmark-multi-step.ts --endpoint http://localhost:8082
 *   npx tsx tests/model-behavior/benchmark-multi-step.ts --endpoint http://localhost:8082 --top-k 15
 *   npx tsx tests/model-behavior/benchmark-multi-step.ts --endpoint http://localhost:8082 --difficulty simple
 */

import type { MultiStepTest, MultiStepResult, StepResult } from './types';
import { allMultiStepTests, simpleChainTests, mediumChainTests, complexChainTests } from './multi-step-chains';
import {
  TOOL_DESCRIPTIONS,
  getMockResult,
  parseLfmToolCalls,
  isDeflection,
} from './benchmark-shared';
import type { ChatMessage } from './benchmark-shared';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ── CLI arguments ─────────────────────────────────────────────────────────

const args = process.argv.slice(2);
function getArg(name: string): string | undefined {
  const idx = args.indexOf(`--${name}`);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : undefined;
}

const ENDPOINT = getArg('endpoint') ?? 'http://localhost:8082';
const TOP_K = getArg('top-k') ? parseInt(getArg('top-k')!, 10) : 0;
const DIFFICULTY = getArg('difficulty') as 'simple' | 'medium' | 'complex' | undefined;

// ── Model query (local — different signature from shared queryModel) ──────

async function queryModelLocal(endpoint: string, messages: ChatMessage[]): Promise<string> {
  const body = {
    messages,
    temperature: 0.1,
    top_p: 0.1,
    max_tokens: 512,
    stream: false,
  };

  const response = await fetch(`${endpoint}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`Model query failed: ${response.status} ${await response.text()}`);
  }

  const data = (await response.json()) as {
    choices: Array<{ message: { content?: string } }>;
  };

  return data.choices?.[0]?.message?.content ?? '';
}

// ── Benchmark runner ──────────────────────────────────────────────────────

async function runChainBenchmark(
  tests: readonly MultiStepTest[],
  endpoint: string,
): Promise<MultiStepResult[]> {
  const results: MultiStepResult[] = [];
  const systemPrompt = `You are LocalCowork, a desktop assistant with full access to local tools for files, documents, tasks, calendar, email, and more.

IMPORTANT RULES:
1. ALWAYS call the appropriate tool. Never say "I can't do that" or "I don't have that capability."
2. Call exactly one tool per response using bracket format: [server.tool_name(param="value")]
3. Use the full dotted tool name (e.g., filesystem.list_dir, NOT list_dir).
4. After receiving a tool result, proceed to the NEXT step immediately. Do NOT ask the user what to do.

Available tools: ${Object.entries(TOOL_DESCRIPTIONS).map(([k, v]) => `${k}: ${v}`).join('\n')}`;

  for (let i = 0; i < tests.length; i++) {
    const test = tests[i];
    const startTime = Date.now();
    const stepResults: StepResult[] = [];
    let chainPassed = true;
    let failedAtStep: number | undefined;
    let failureReason: StepResult['failureReason'] | undefined;

    const messages: ChatMessage[] = [
      { role: 'system', content: systemPrompt },
    ];

    for (let s = 0; s < test.steps.length; s++) {
      const step = test.steps[s];
      messages.push({ role: 'user', content: step.prompt });

      let content: string;
      try {
        content = await queryModelLocal(endpoint, messages);
      } catch (e) {
        stepResults.push({
          stepIndex: s,
          expectedTools: step.expectedTools,
          actualTools: [],
          status: 'failed',
          failureReason: 'error',
          rawContent: String(e),
        });
        chainPassed = false;
        failedAtStep = s;
        failureReason = 'error';
        break;
      }

      const actualTools = parseLfmToolCalls(content);
      const stepPassed = step.expectedTools.some((t) => actualTools.includes(t));
      const deflected = actualTools.length === 0 && isDeflection(content);

      let stepFailureReason: StepResult['failureReason'] | undefined;
      if (!stepPassed) {
        if (deflected) {
          stepFailureReason = 'deflection';
        } else if (actualTools.length === 0) {
          stepFailureReason = 'no_tool';
        } else {
          stepFailureReason = 'wrong_tool';
        }
      }

      stepResults.push({
        stepIndex: s,
        expectedTools: step.expectedTools,
        actualTools,
        status: stepPassed ? 'passed' : 'failed',
        failureReason: stepFailureReason,
        rawContent: content.slice(0, 200),
      });

      if (!stepPassed) {
        chainPassed = false;
        failedAtStep = s;
        failureReason = stepFailureReason;
        break;
      }

      // Add assistant response and mock tool result to conversation history
      messages.push({ role: 'assistant', content });
      messages.push({
        role: 'tool',
        content: getMockResult(actualTools[0]),
      });
    }

    const durationMs = Date.now() - startTime;
    const stepsCompleted = stepResults.filter((r) => r.status === 'passed').length;

    const status = chainPassed ? 'passed' : 'failed';
    const icon = chainPassed ? '✓' : '✗';
    const detail = chainPassed
      ? `${stepsCompleted}/${test.steps.length} steps`
      : `FAILED at step ${(failedAtStep ?? 0) + 1}: ${failureReason}`;

    console.log(
      `[${String(i + 1).padStart(3)}/${tests.length}] ${icon} ${test.id} — ${detail} (${durationMs}ms)`,
    );

    results.push({
      testId: test.id,
      scenario: test.scenario,
      difficulty: test.difficulty,
      status,
      stepsCompleted,
      totalSteps: test.steps.length,
      failedAtStep,
      failureReason,
      stepResults,
      durationMs,
    });
  }

  return results;
}

// ── Main ──────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  // Select test subset
  let tests: readonly MultiStepTest[];
  if (DIFFICULTY === 'simple') {
    tests = simpleChainTests;
  } else if (DIFFICULTY === 'medium') {
    tests = mediumChainTests;
  } else if (DIFFICULTY === 'complex') {
    tests = complexChainTests;
  } else {
    tests = allMultiStepTests;
  }

  // Verify model server
  try {
    const resp = await fetch(`${ENDPOINT}/v1/models`);
    const data = (await resp.json()) as { data?: Array<{ id: string }> };
    const models = data.data?.map((m) => m.id) ?? [];
    console.log(`✓ Model server reachable. Models: ${JSON.stringify(models)}\n`);
  } catch {
    console.error(`✗ Cannot reach model server at ${ENDPOINT}`);
    process.exit(1);
  }

  console.log('══════════════════════════════════════════════════════════════════════');
  console.log('  LFM Multi-Step Chain Benchmark');
  console.log(`  Endpoint: ${ENDPOINT}`);
  console.log(`  Tests: ${tests.length} | Difficulty: ${DIFFICULTY ?? 'all'}`);
  console.log('══════════════════════════════════════════════════════════════════════\n');

  const startTime = Date.now();
  const results = await runChainBenchmark(tests, ENDPOINT);
  const totalDuration = Date.now() - startTime;

  // ── Compute metrics ──────────────────────────────────────────────────

  const passed = results.filter((r) => r.status === 'passed').length;
  const failed = results.filter((r) => r.status === 'failed').length;
  const accuracy = tests.length > 0 ? (passed / tests.length) * 100 : 0;

  const deflections = results.filter((r) => r.failureReason === 'deflection').length;
  const wrongTools = results.filter((r) => r.failureReason === 'wrong_tool').length;
  const noTools = results.filter((r) => r.failureReason === 'no_tool').length;

  const totalSteps = results.reduce((sum, r) => sum + r.totalSteps, 0);
  const completedSteps = results.reduce((sum, r) => sum + r.stepsCompleted, 0);
  const avgStepsCompleted =
    results.length > 0
      ? results.reduce((sum, r) => sum + r.stepsCompleted, 0) / results.length
      : 0;

  // Per-difficulty
  const byDifficulty: Record<string, { total: number; passed: number }> = {};
  for (const r of results) {
    const d = r.difficulty;
    if (!byDifficulty[d]) byDifficulty[d] = { total: 0, passed: 0 };
    byDifficulty[d].total++;
    if (r.status === 'passed') byDifficulty[d].passed++;
  }

  // ── Print results ────────────────────────────────────────────────────

  console.log('\n══════════════════════════════════════════════════════════════════════');
  console.log('  MULTI-STEP BENCHMARK RESULTS');
  console.log('══════════════════════════════════════════════════════════════════════');
  console.log(`  Model:          ${ENDPOINT}`);
  console.log(`  Duration:       ${(totalDuration / 1000).toFixed(1)}s`);
  console.log('──────────────────────────────────────────────────────────────────────');
  console.log(`  Total Chains:   ${tests.length}`);
  console.log(`  Passed:         ${passed}`);
  console.log(`  Failed:         ${failed}`);
  console.log('──────────────────────────────────────────────────────────────────────');
  console.log(`  CHAIN COMPLETION: ${accuracy.toFixed(0)}%`);
  console.log(`  Step Completion:  ${totalSteps > 0 ? ((completedSteps / totalSteps) * 100).toFixed(0) : 0}% (${completedSteps}/${totalSteps} steps)`);
  console.log(`  Avg Steps/Chain:  ${avgStepsCompleted.toFixed(1)}`);
  console.log('──────────────────────────────────────────────────────────────────────');
  console.log(`  FAILURE BREAKDOWN:`);
  console.log(`    Deflection (FM-3): ${deflections} (${tests.length > 0 ? ((deflections / tests.length) * 100).toFixed(0) : 0}%)`);
  console.log(`    Wrong Tool:        ${wrongTools} (${tests.length > 0 ? ((wrongTools / tests.length) * 100).toFixed(0) : 0}%)`);
  console.log(`    No Tool Call:      ${noTools} (${tests.length > 0 ? ((noTools / tests.length) * 100).toFixed(0) : 0}%)`);
  console.log('──────────────────────────────────────────────────────────────────────');
  console.log('  BY DIFFICULTY:');
  for (const [diff, stats] of Object.entries(byDifficulty).sort()) {
    const pct = stats.total > 0 ? ((stats.passed / stats.total) * 100).toFixed(0) : '0';
    console.log(`    ${diff.padEnd(10)} ${pct}% (${stats.passed}/${stats.total})`);
  }
  console.log('══════════════════════════════════════════════════════════════════════\n');

  // ── Save results ─────────────────────────────────────────────────────

  const resultsDir = path.join(__dirname, '.results');
  if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

  const filename = `lfm-multistep-${DIFFICULTY ?? 'all'}-${Date.now()}.json`;
  const outputPath = path.join(resultsDir, filename);

  fs.writeFileSync(
    outputPath,
    JSON.stringify(
      {
        runId: `ms-${Date.now()}`,
        timestamp: new Date().toISOString(),
        endpoint: ENDPOINT,
        topK: TOP_K,
        difficulty: DIFFICULTY ?? 'all',
        totalChains: tests.length,
        passed,
        failed,
        chainCompletionRate: accuracy,
        stepCompletionRate: totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0,
        avgStepsCompleted,
        deflectionRate: tests.length > 0 ? (deflections / tests.length) * 100 : 0,
        byDifficulty,
        results,
        durationMs: totalDuration,
      },
      null,
      2,
    ),
  );

  console.log(`Results saved to: ${outputPath}`);
}

main().catch((e) => {
  console.error('Benchmark failed:', e);
  process.exit(1);
});
