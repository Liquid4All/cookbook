/**
 * Quality Benchmark Runner — measures param extraction, instruction following,
 * and synthesis quality beyond tool selection accuracy.
 *
 * Usage:
 *   # Run all 3 modules:
 *   npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:8080 --greedy
 *
 *   # Run a single module:
 *   npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:8080 --greedy --module params
 *   npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:8080 --greedy --module instructions
 *   npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:8080 --greedy --module synthesis
 *
 *   # Options:
 *   --endpoint URL     Model endpoint (default: http://localhost:8080)
 *   --model NAME       Model name for Ollama
 *   --timeout MS       Per-request timeout (default: 45000)
 *   --greedy           Use greedy sampling (temp=0, top_p=1.0)
 *   --module NAME      Run a single module: params | instructions | synthesis
 *
 * Output:
 *   - Console summary with per-category breakdown
 *   - JSON results in tests/model-behavior/.results/
 */

import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { allParamExtractionTests } from './param-extraction-tests';
import type { ParamExtractionTest } from './param-extraction-tests';
import { allInstructionFollowingTests } from './instruction-following-tests';
import type { InstructionFollowingTest } from './instruction-following-tests';
import { allSynthesisTests } from './synthesis-tests';
import type { SynthesisTest } from './synthesis-tests';
import {
  scoreParamExtraction,
  scoreInstructionFollowing,
  scoreSynthesis,
} from './quality-scoring';
import type { ParamScore, InstructionScore, SynthesisScore } from './quality-scoring';
import { TOOL_DESCRIPTIONS, queryModel } from './benchmark-shared';
import type { ChatMessage } from './benchmark-shared';

// ─── CLI Args ────────────────────────────────────────────────────────────────

const DEFAULT_ENDPOINT = 'http://localhost:8080';
const DEFAULT_TIMEOUT_MS = 45_000;

type ModuleName = 'params' | 'instructions' | 'synthesis';

interface CliArgs {
  readonly endpoint: string;
  readonly timeoutMs: number;
  readonly model: string | undefined;
  readonly greedy: boolean;
  readonly module: ModuleName | 'all';
}

function parseArgs(): CliArgs {
  const args = process.argv.slice(2);
  let endpoint = DEFAULT_ENDPOINT;
  let timeoutMs = DEFAULT_TIMEOUT_MS;
  let model: string | undefined;
  let module_: ModuleName | 'all' = 'all';
  const greedy = args.includes('--greedy');

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--endpoint' && args[i + 1]) {
      endpoint = args[++i];
    } else if (args[i] === '--timeout' && args[i + 1]) {
      timeoutMs = parseInt(args[++i], 10);
    } else if (args[i] === '--model' && args[i + 1]) {
      model = args[++i];
    } else if (args[i] === '--module' && args[i + 1]) {
      const m = args[++i] as ModuleName;
      if (['params', 'instructions', 'synthesis'].includes(m)) {
        module_ = m;
      }
    }
  }

  return { endpoint, timeoutMs, model, greedy, module: module_ };
}

// ─── System Prompt ───────────────────────────────────────────────────────────

function buildToolListMarkdown(): string {
  const lines: string[] = [];
  let currentServer = '';
  for (const [name, desc] of Object.entries(TOOL_DESCRIPTIONS)) {
    const server = name.split('.')[0];
    if (server !== currentServer) {
      currentServer = server;
      lines.push(`\n### ${server}`);
    }
    lines.push(`- **${name}**: ${desc}`);
  }
  return lines.join('\n');
}

const SYSTEM_PROMPT = `You are a local desktop AI assistant with access to the following tools.
When the user asks you to perform an action, respond with a tool call in bracket format:
[tool.name(param1="value1", param2="value2")]

Available tools:
${buildToolListMarkdown()}

Rules:
- Always use the bracket format for tool calls: [server.tool_name(args)]
- Extract all relevant parameters from the user's request
- If a request is ambiguous, ask for clarification instead of guessing
- If the user asks a question that doesn't require a tool, answer directly
- Follow the user's formatting instructions precisely (numbered lists, bullet points, JSON, etc.)
- Respect length constraints when specified
- When given tool results, synthesize them into a natural, useful response — never paste raw JSON
- Today's date is 2026-02-20 (Thursday)`;

// ─── Result Types ────────────────────────────────────────────────────────────

interface ParamTestResult {
  readonly testId: string;
  readonly category: string;
  readonly difficulty: string;
  readonly prompt: string;
  readonly rawContent: string;
  readonly score: ParamScore;
  readonly durationMs: number;
}

interface InstructionTestResult {
  readonly testId: string;
  readonly category: string;
  readonly difficulty: string;
  readonly prompt: string;
  readonly rawContent: string;
  readonly score: InstructionScore;
  readonly durationMs: number;
}

interface SynthesisTestResult {
  readonly testId: string;
  readonly category: string;
  readonly difficulty: string;
  readonly userQuery: string;
  readonly rawContent: string;
  readonly score: SynthesisScore;
  readonly durationMs: number;
}

interface CategorySummary {
  readonly total: number;
  readonly avgScore: number;
  readonly minScore: number;
  readonly maxScore: number;
}

interface ModuleResult {
  readonly module: string;
  readonly totalTests: number;
  readonly avgScore: number;
  readonly categories: Record<string, CategorySummary>;
  readonly results: ReadonlyArray<ParamTestResult | InstructionTestResult | SynthesisTestResult>;
  readonly durationMs: number;
}

interface QualityBenchmarkResult {
  readonly runId: string;
  readonly timestamp: string;
  readonly endpoint: string;
  readonly model: string | undefined;
  readonly sampling: { readonly mode: string; readonly temperature: number; readonly topP: number };
  readonly modules: Record<string, ModuleResult>;
  readonly overallScore: number;
  readonly totalTests: number;
  readonly durationMs: number;
}

// ─── Param Extraction Runner ─────────────────────────────────────────────────

async function runParamExtraction(
  tests: readonly ParamExtractionTest[],
  config: CliArgs,
): Promise<ModuleResult> {
  const results: ParamTestResult[] = [];
  const startTime = Date.now();

  console.log(`\n${'═'.repeat(60)}`);
  console.log('  MODULE: Parameter Extraction Accuracy');
  console.log(`${'═'.repeat(60)}`);

  for (let i = 0; i < tests.length; i++) {
    const test = tests[i];
    const label = `[${i + 1}/${tests.length}] ${test.id}`;

    try {
      const messages: ChatMessage[] = [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: test.prompt },
      ];

      const reqStart = Date.now();
      const response = await queryModel(config.endpoint, messages, {
        temperature: config.greedy ? 0 : 0.1,
        topP: config.greedy ? 1.0 : 0.1,
        maxTokens: 512,
        model: config.model,
      });
      const durationMs = Date.now() - reqStart;

      const score = scoreParamExtraction(response.content, test.expectedTool, test.expectedParams);

      const icon = score.composite >= 0.8 ? '✅' : score.composite >= 0.5 ? '⚠️' : '❌';
      console.log(`  ${icon} ${label}: ${(score.composite * 100).toFixed(0)}% (tool=${score.toolCorrect ? '✓' : '✗'} keys=${(score.keyRecall * 100).toFixed(0)}% vals=${(score.valueAccuracy * 100).toFixed(0)}%) [${durationMs}ms]`);

      results.push({
        testId: test.id,
        category: test.category,
        difficulty: test.difficulty,
        prompt: test.prompt,
        rawContent: response.content,
        score,
        durationMs,
      });
    } catch (err) {
      console.log(`  💥 ${label}: ERROR — ${(err as Error).message}`);
      results.push({
        testId: test.id,
        category: test.category,
        difficulty: test.difficulty,
        prompt: test.prompt,
        rawContent: '',
        score: { toolCorrect: false, keyRecall: 0, valueAccuracy: 0, hallucinatedKeys: 0, composite: 0 },
        durationMs: 0,
      });
    }
  }

  return buildModuleResult('param-extraction', results, startTime);
}

// ─── Instruction Following Runner ────────────────────────────────────────────

async function runInstructionFollowing(
  tests: readonly InstructionFollowingTest[],
  config: CliArgs,
): Promise<ModuleResult> {
  const results: InstructionTestResult[] = [];
  const startTime = Date.now();

  console.log(`\n${'═'.repeat(60)}`);
  console.log('  MODULE: Instruction Following Precision');
  console.log(`${'═'.repeat(60)}`);

  for (let i = 0; i < tests.length; i++) {
    const test = tests[i];
    const label = `[${i + 1}/${tests.length}] ${test.id}`;

    try {
      const messages: ChatMessage[] = [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: test.prompt },
      ];

      const reqStart = Date.now();
      const response = await queryModel(config.endpoint, messages, {
        temperature: config.greedy ? 0 : 0.1,
        topP: config.greedy ? 1.0 : 0.1,
        maxTokens: 1024,
        model: config.model,
      });
      const durationMs = Date.now() - reqStart;

      const score = scoreInstructionFollowing(response.content, test.constraints);

      const icon = score.score >= 0.8 ? '✅' : score.score >= 0.5 ? '⚠️' : '❌';
      const failed = score.results.filter((r) => !r.passed).map((r) => r.type);
      const failInfo = failed.length > 0 ? ` (failed: ${failed.join(', ')})` : '';
      console.log(`  ${icon} ${label}: ${score.constraintsPassed}/${score.totalConstraints} constraints${failInfo} [${durationMs}ms]`);

      results.push({
        testId: test.id,
        category: test.category,
        difficulty: test.difficulty,
        prompt: test.prompt,
        rawContent: response.content,
        score,
        durationMs,
      });
    } catch (err) {
      console.log(`  💥 ${label}: ERROR — ${(err as Error).message}`);
      results.push({
        testId: test.id,
        category: test.category,
        difficulty: test.difficulty,
        prompt: test.prompt,
        rawContent: '',
        score: { constraintsPassed: 0, totalConstraints: test.constraints.length, score: 0, results: [] },
        durationMs: 0,
      });
    }
  }

  return buildModuleResult('instruction-following', results, startTime);
}

// ─── Synthesis Runner ────────────────────────────────────────────────────────

async function runSynthesis(
  tests: readonly SynthesisTest[],
  config: CliArgs,
): Promise<ModuleResult> {
  const results: SynthesisTestResult[] = [];
  const startTime = Date.now();

  console.log(`\n${'═'.repeat(60)}`);
  console.log('  MODULE: Response Synthesis Quality');
  console.log(`${'═'.repeat(60)}`);

  for (let i = 0; i < tests.length; i++) {
    const test = tests[i];
    const label = `[${i + 1}/${tests.length}] ${test.id}`;

    try {
      // Build conversation: user asks → assistant calls tool → tool returns → user sees result
      const messages: ChatMessage[] = [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: test.userQuery },
        { role: 'assistant', content: test.toolCall },
        { role: 'tool', content: test.toolResult },
      ];

      const reqStart = Date.now();
      const response = await queryModel(config.endpoint, messages, {
        temperature: config.greedy ? 0 : 0.1,
        topP: config.greedy ? 1.0 : 0.1,
        maxTokens: 1024,
        model: config.model,
      });
      const durationMs = Date.now() - reqStart;

      const score = scoreSynthesis(response.content, test.constraints);

      const icon = score.score >= 0.8 ? '✅' : score.score >= 0.5 ? '⚠️' : '❌';
      const failed = score.results.filter((r) => !r.passed).map((r) => r.type);
      const failInfo = failed.length > 0 ? ` (failed: ${failed.join(', ')})` : '';
      console.log(`  ${icon} ${label}: ${score.constraintsPassed}/${score.totalConstraints} constraints${failInfo} [${durationMs}ms]`);

      results.push({
        testId: test.id,
        category: test.category,
        difficulty: test.difficulty,
        userQuery: test.userQuery,
        rawContent: response.content,
        score,
        durationMs,
      });
    } catch (err) {
      console.log(`  💥 ${label}: ERROR — ${(err as Error).message}`);
      results.push({
        testId: test.id,
        category: test.category,
        difficulty: test.difficulty,
        userQuery: test.userQuery,
        rawContent: '',
        score: { constraintsPassed: 0, totalConstraints: test.constraints.length, score: 0, results: [] },
        durationMs: 0,
      });
    }
  }

  return buildModuleResult('synthesis', results, startTime);
}

// ─── Result Aggregation ──────────────────────────────────────────────────────

function getScore(result: ParamTestResult | InstructionTestResult | SynthesisTestResult): number {
  if ('score' in result) {
    const s = result.score;
    if ('composite' in s) return (s as ParamScore).composite;
    if ('score' in s) return (s as InstructionScore | SynthesisScore).score;
  }
  return 0;
}

function buildModuleResult(
  module: string,
  results: ReadonlyArray<ParamTestResult | InstructionTestResult | SynthesisTestResult>,
  startTime: number,
): ModuleResult {
  const scores = results.map(getScore);
  const avgScore = scores.length > 0
    ? scores.reduce((a, b) => a + b, 0) / scores.length
    : 0;

  // Group by category
  const categories: Record<string, CategorySummary> = {};
  const byCategory = new Map<string, number[]>();
  for (const r of results) {
    const cat = r.category;
    if (!byCategory.has(cat)) byCategory.set(cat, []);
    byCategory.get(cat)!.push(getScore(r));
  }
  for (const [cat, catScores] of byCategory.entries()) {
    categories[cat] = {
      total: catScores.length,
      avgScore: catScores.reduce((a, b) => a + b, 0) / catScores.length,
      minScore: Math.min(...catScores),
      maxScore: Math.max(...catScores),
    };
  }

  return {
    module,
    totalTests: results.length,
    avgScore,
    categories,
    results,
    durationMs: Date.now() - startTime,
  };
}

// ─── Console Summary ─────────────────────────────────────────────────────────

function printModuleSummary(mod: ModuleResult): void {
  console.log(`\n  📊 ${mod.module.toUpperCase()} — Average Score: ${(mod.avgScore * 100).toFixed(1)}%`);
  console.log(`  ${'─'.repeat(50)}`);
  for (const [cat, summary] of Object.entries(mod.categories)) {
    const bar = '█'.repeat(Math.round(summary.avgScore * 20)) + '░'.repeat(20 - Math.round(summary.avgScore * 20));
    console.log(`  ${bar} ${(summary.avgScore * 100).toFixed(1)}%  ${cat} (${summary.total} tests)`);
  }
  console.log(`  Duration: ${(mod.durationMs / 1000).toFixed(1)}s`);
}

function printOverallSummary(result: QualityBenchmarkResult): void {
  console.log(`\n${'═'.repeat(60)}`);
  console.log('  QUALITY BENCHMARK SUMMARY');
  console.log(`${'═'.repeat(60)}`);
  console.log(`  Endpoint: ${result.endpoint}`);
  console.log(`  Model: ${result.model ?? 'default'}`);
  console.log(`  Sampling: ${result.sampling.mode} (temp=${result.sampling.temperature}, top_p=${result.sampling.topP})`);
  console.log(`  Total tests: ${result.totalTests}`);

  for (const mod of Object.values(result.modules)) {
    printModuleSummary(mod);
  }

  console.log(`\n  ${'─'.repeat(50)}`);
  console.log(`  🏆 OVERALL QUALITY SCORE: ${(result.overallScore * 100).toFixed(1)}%`);
  console.log(`  Total duration: ${(result.durationMs / 1000).toFixed(1)}s`);
  console.log(`${'═'.repeat(60)}\n`);
}

// ─── File Output ─────────────────────────────────────────────────────────────

function saveResults(result: QualityBenchmarkResult): string {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = dirname(__filename);
  const resultsDir = join(__dirname, '.results');
  if (!existsSync(resultsDir)) mkdirSync(resultsDir, { recursive: true });

  const modules = Object.keys(result.modules).join('-');
  const filename = `quality-${modules}-${Date.now()}.json`;
  const filepath = join(resultsDir, filename);
  writeFileSync(filepath, JSON.stringify(result, null, 2));
  return filepath;
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const config = parseArgs();
  const startTime = Date.now();

  console.log('\n🔬 Quality Benchmark Suite');
  console.log(`   Endpoint: ${config.endpoint}`);
  console.log(`   Sampling: ${config.greedy ? 'greedy' : 'near-greedy'}`);
  console.log(`   Module:   ${config.module}`);

  const modules: Record<string, ModuleResult> = {};

  if (config.module === 'all' || config.module === 'params') {
    modules['param-extraction'] = await runParamExtraction(allParamExtractionTests, config);
  }

  if (config.module === 'all' || config.module === 'instructions') {
    modules['instruction-following'] = await runInstructionFollowing(allInstructionFollowingTests, config);
  }

  if (config.module === 'all' || config.module === 'synthesis') {
    modules['synthesis'] = await runSynthesis(allSynthesisTests, config);
  }

  // Calculate overall score
  const allModules = Object.values(modules);
  const totalTests = allModules.reduce((sum, m) => sum + m.totalTests, 0);
  const overallScore = totalTests > 0
    ? allModules.reduce((sum, m) => sum + m.avgScore * m.totalTests, 0) / totalTests
    : 0;

  const result: QualityBenchmarkResult = {
    runId: `quality-${Date.now()}`,
    timestamp: new Date().toISOString(),
    endpoint: config.endpoint,
    model: config.model,
    sampling: {
      mode: config.greedy ? 'greedy' : 'near-greedy',
      temperature: config.greedy ? 0 : 0.1,
      topP: config.greedy ? 1.0 : 0.1,
    },
    modules,
    overallScore,
    totalTests,
    durationMs: Date.now() - startTime,
  };

  printOverallSummary(result);

  const filepath = saveResults(result);
  console.log(`  📁 Results saved to: ${filepath}\n`);
}

main().catch((err: unknown) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
