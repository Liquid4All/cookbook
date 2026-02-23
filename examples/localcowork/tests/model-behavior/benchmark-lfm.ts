/**
 * LFM Tool-Calling Benchmark Runner — with optional RAG pre-filter.
 *
 * Benchmarks an LFM model's tool-calling accuracy against the full 67-tool
 * LocalCowork tool set. Supports an embedding-based pre-filter (Option E)
 * that narrows the tool set to top-K most relevant tools per query.
 *
 * Usage:
 *   # Unfiltered (baseline) — all 67 tools sent to model:
 *   npx tsx tests/model-behavior/benchmark-lfm.ts
 *
 *   # Filtered — top-K tools only (requires --embeddings on llama-server):
 *   npx tsx tests/model-behavior/benchmark-lfm.ts --top-k 10
 *
 *   # Options:
 *   --endpoint URL     Model endpoint (default: http://localhost:8082)
 *   --model NAME       Model name for Ollama (e.g., "mistral-small:24b")
 *   --timeout MS       Per-request timeout (default: 30000)
 *   --top-k N          Enable RAG pre-filter with top-K tools (0 = disabled)
 *   --greedy           Use greedy sampling (temp=0, top_p=1.0, top_k=0)
 *   --servers LIST     Comma-separated server names to restrict tool set and tests
 *                      (e.g., "security,audit,document,ocr")
 *
 * Prerequisites:
 *   - llama-server running on port 8082
 *   - For --top-k mode: llama-server started with --embeddings flag
 *   - Node.js 18+ (native fetch)
 *
 * Output:
 *   - Console summary (per-category accuracy, filter hit rate)
 *   - JSON results file in tests/model-behavior/.results/
 */

import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { allToolSelectionTests } from './tool-selection';
import type { IndividualTestResult, CategoryResult } from './types';
import { VALID_TOOL_NAMES } from './types';
import {
  TOOL_DESCRIPTIONS,
  buildToolDefinitions,
  buildFilteredToolDefinitions,
  buildToolEmbeddingIndex,
  filterToolsByRelevance,
  parseLfmToolCalls,
} from './benchmark-shared';
import type { ToolDef, ToolEmbeddingIndex } from './benchmark-shared';

// ─── Configuration ──────────────────────────────────────────────────────────

const DEFAULT_ENDPOINT = 'http://localhost:8082';
const DEFAULT_TIMEOUT_MS = 30_000;

// ─── CLI Args ───────────────────────────────────────────────────────────────

interface CliArgs {
  endpoint: string;
  timeoutMs: number;
  topK: number;
  model: string | undefined;
  greedy: boolean;
  servers: string[];
}

function parseArgs(): CliArgs {
  const args = process.argv.slice(2);
  let endpoint = DEFAULT_ENDPOINT;
  let timeoutMs = DEFAULT_TIMEOUT_MS;
  let topK = 0;
  let model: string | undefined;
  let servers: string[] = [];
  const greedy = args.includes('--greedy');

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--endpoint' && args[i + 1]) {
      endpoint = args[++i];
    } else if (args[i] === '--timeout' && args[i + 1]) {
      timeoutMs = parseInt(args[++i], 10);
    } else if (args[i] === '--top-k' && args[i + 1]) {
      topK = parseInt(args[++i], 10);
    } else if (args[i] === '--model' && args[i + 1]) {
      model = args[++i];
    } else if (args[i] === '--servers' && args[i + 1]) {
      servers = args[++i].split(',').map((s) => s.trim()).filter(Boolean);
    }
  }

  return { endpoint, timeoutMs, topK, model, greedy, servers };
}

// ─── Server Filtering ──────────────────────────────────────────────────────

/** Filter VALID_TOOL_NAMES to only tools belonging to the given servers. */
function filterToolNamesByServers(servers: string[]): string[] {
  const serverSet = new Set(servers);
  return VALID_TOOL_NAMES.filter((name) => serverSet.has(name.split('.')[0]));
}

/** Filter test cases to only those whose expectedTools include at least one tool from the server set. */
function filterTestsByServers(
  tests: readonly import('./types').ToolSelectionTest[],
  servers: string[],
): import('./types').ToolSelectionTest[] {
  const serverSet = new Set(servers);
  return tests.filter((test) =>
    test.expectedTools.some((tool) => serverSet.has(tool.split('.')[0])),
  );
}

// ─── Model Communication ───────────────────────────────────────────────────

interface ChatCompletionResponse {
  choices: Array<{
    message: {
      content?: string | null;
      tool_calls?: Array<{
        function: { name: string; arguments: string };
      }>;
    };
    finish_reason?: string;
  }>;
}

/** Send a prompt to the model and extract tool selections. */
async function queryModel(
  endpoint: string,
  systemPrompt: string,
  userPrompt: string,
  context: readonly string[],
  timeoutMs: number,
  model?: string,
  greedy?: boolean,
): Promise<{ tools: string[]; rawContent: string; durationMs: number }> {
  const messages: Array<{ role: string; content: string }> = [
    { role: 'system', content: systemPrompt },
  ];

  for (let i = 0; i < context.length; i++) {
    messages.push({
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: context[i],
    });
  }
  messages.push({ role: 'user', content: userPrompt });

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const startTime = Date.now();

  const body: Record<string, unknown> = {
    messages,
    temperature: greedy ? 0 : 0.1,
    top_k: greedy ? 0 : 50,
    top_p: greedy ? 1.0 : 0.1,
    repetition_penalty: greedy ? 1.0 : 1.05,
    max_tokens: 512,
  };
  if (model) {
    body.model = model;
  }

  try {
    const response = await fetch(`${endpoint}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    const durationMs = Date.now() - startTime;

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Model returned HTTP ${response.status}: ${errText}`);
    }

    const data = (await response.json()) as ChatCompletionResponse;
    const content = data.choices[0]?.message?.content ?? '';

    let tools = parseLfmToolCalls(content);

    if (tools.length === 0 && data.choices[0]?.message?.tool_calls) {
      tools = data.choices[0].message.tool_calls.map((tc) => tc.function.name);
    }

    return { tools, rawContent: content, durationMs };
  } finally {
    clearTimeout(timer);
  }
}

// ─── Benchmark Runner ───────────────────────────────────────────────────────

interface BenchmarkResult {
  runId: string;
  timestamp: string;
  model: string;
  endpoint: string;
  mode: 'unfiltered' | 'filtered';
  topK: number;
  totalTests: number;
  passed: number;
  failed: number;
  skipped: number;
  accuracyPercent: number;
  avgLatencyMs: number;
  categories: Record<string, CategoryResult>;
  individual: IndividualTestResult[];
  toolCallRate: number;
  wrongToolRate: number;
  noToolRate: number;
  restraintScore: number;
  durationMs: number;
  sampling: {
    temperature: number;
    top_p: number;
    top_k: number;
    repetition_penalty: number;
    mode: 'greedy' | 'near-greedy';
  };
  // Filter-specific metrics
  filterHitRate?: number;       // % of tests where correct tool was in top-K
  filterHitCount?: number;      // Number of tests where correct tool was in top-K
  avgFilteredToolCount?: number; // Average number of tools sent to model
  // Server-filter metadata
  servers?: string[];
  toolCount?: number;
}

async function runBenchmark(
  endpoint: string,
  timeoutMs: number,
  topK: number,
  toolIndex: ToolEmbeddingIndex | null,
  model?: string,
  greedy?: boolean,
  servers?: string[],
): Promise<BenchmarkResult> {
  const hasServerFilter = servers !== undefined && servers.length > 0;
  const allToolDefs = hasServerFilter
    ? buildFilteredToolDefinitions(filterToolNamesByServers(servers))
    : buildToolDefinitions();
  const tests = hasServerFilter
    ? filterTestsByServers(allToolSelectionTests, servers)
    : allToolSelectionTests;
  const results: IndividualTestResult[] = [];
  const startTime = Date.now();
  let passed = 0;
  let failed = 0;
  let skipped = 0;
  let totalLatency = 0;
  let noToolCount = 0;
  let wrongToolCount = 0;
  let filterHitCount = 0;
  let totalFilteredTools = 0;

  const isFiltered = topK > 0 && toolIndex !== null;
  const mode = isFiltered ? 'filtered' : 'unfiltered';

  const samplingMode = greedy ? 'GREEDY (temp=0)' : 'NEAR-GREEDY (temp=0.1)';

  const toolCount = allToolDefs.length;

  console.log(`\n${'═'.repeat(70)}`);
  console.log(`  LFM Tool Selection Benchmark — ${mode.toUpperCase()}`);
  console.log(`  Endpoint: ${endpoint}`);
  console.log(`  Tests: ${tests.length} | Tools: ${toolCount} (of ${VALID_TOOL_NAMES.length} total)`);
  console.log(`  Sampling: ${samplingMode}`);
  if (hasServerFilter) {
    console.log(`  Servers: ${servers.join(', ')}`);
  }
  if (isFiltered) {
    console.log(`  Pre-filter: top-K=${topK} via /v1/embeddings`);
  }
  console.log(`${'═'.repeat(70)}\n`);

  for (let i = 0; i < tests.length; i++) {
    const test = tests[i];
    const progress = `[${String(i + 1).padStart(3)}/${tests.length}]`;

    try {
      // Determine which tools to send
      let toolsForPrompt: ToolDef[];
      let filteredNames: string[] | null = null;

      if (isFiltered && toolIndex) {
        const { selectedTools, scores } = await filterToolsByRelevance(
          endpoint, test.prompt, toolIndex, topK,
        );
        filteredNames = selectedTools;
        toolsForPrompt = buildFilteredToolDefinitions(selectedTools);
        totalFilteredTools += selectedTools.length;

        // Check if the expected tool(s) are in the filtered set
        const filteredSet = new Set(selectedTools);
        const hasExpected = test.expectedTools.some((t) => filteredSet.has(t));
        if (hasExpected) filterHitCount++;
      } else {
        toolsForPrompt = allToolDefs;
      }

      // Build system prompt with the (possibly filtered) tool set
      const toolsJson = JSON.stringify(toolsForPrompt);
      const systemPrompt = `You are LocalCowork, a desktop assistant with full access to local tools for files, documents, tasks, calendar, email, and more.

IMPORTANT RULES:
1. ALWAYS call the appropriate tool. Never say "I can't do that" or "I don't have that capability" — you have tools for file deletion, file writing, clipboard, encryption, audit logging, and more.
2. Call exactly one tool per response using bracket format: [server.tool_name(param="value")]
3. Use the full dotted tool name (e.g., filesystem.list_dir, NOT list_dir).
4. For file rename operations, use filesystem.move_file with the new name as destination.
5. For extracting text from images or screenshots, use ocr tools. For document files (PDF, DOCX), use document tools.

Available tools: [${toolsJson.slice(1, -1)}]`;

      const { tools: actualTools, rawContent, durationMs } = await queryModel(
        endpoint, systemPrompt, test.prompt, test.context ?? [], timeoutMs, model, greedy,
      );

      totalLatency += durationMs;

      const actualSet = new Set(actualTools);
      const hasCorrectTool = test.expectedTools.some((t) => actualSet.has(t));

      if (actualTools.length === 0) {
        noToolCount++;
        failed++;
        const snippet = rawContent.slice(0, 80).replace(/\n/g, ' ');
        console.log(`${progress} ✗ ${test.id} — NO TOOL CALL (${durationMs}ms)`);
        console.log(`         Expected: ${test.expectedTools.join(', ')}`);
        if (filteredNames) {
          const hasInFilter = test.expectedTools.some((t) => filteredNames!.includes(t));
          console.log(`         Filter: [${filteredNames.slice(0, 5).join(', ')}${filteredNames.length > 5 ? '...' : ''}] ${hasInFilter ? '✓ hit' : '✗ miss'}`);
        }
        console.log(`         Response: "${snippet}..."`);

        results.push({
          testId: test.id, status: 'failed', expectedTools: test.expectedTools,
          actualTools, error: `No tool call. Response: ${rawContent.slice(0, 200)}`, durationMs,
        });
      } else if (hasCorrectTool) {
        passed++;
        console.log(
          `${progress} ✓ ${test.id} — ${actualTools.join(', ')} (${durationMs}ms)`,
        );
        results.push({
          testId: test.id, status: 'passed', expectedTools: test.expectedTools,
          actualTools, durationMs,
        });
      } else {
        wrongToolCount++;
        failed++;
        console.log(`${progress} ✗ ${test.id} — WRONG TOOL (${durationMs}ms)`);
        console.log(`         Expected: ${test.expectedTools.join(', ')}`);
        console.log(`         Got:      ${actualTools.join(', ')}`);
        if (filteredNames) {
          const hasInFilter = test.expectedTools.some((t) => filteredNames!.includes(t));
          console.log(`         Filter: [${filteredNames.slice(0, 5).join(', ')}${filteredNames.length > 5 ? '...' : ''}] ${hasInFilter ? '✓ hit' : '✗ miss'}`);
        }
        results.push({
          testId: test.id, status: 'failed', expectedTools: test.expectedTools,
          actualTools, error: `Wrong tool. Expected: ${test.expectedTools.join(', ')}. Got: ${actualTools.join(', ')}`, durationMs,
        });
      }
    } catch (err) {
      skipped++;
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.log(`${progress} ⊘ ${test.id} — ERROR: ${errorMsg.slice(0, 80)}`);
      results.push({
        testId: test.id, status: 'skipped', expectedTools: test.expectedTools,
        error: errorMsg, durationMs: 0,
      });
    }
  }

  const totalDurationMs = Date.now() - startTime;
  const accuracyPercent = tests.length > 0
    ? Math.round((passed / tests.length) * 10000) / 100 : 0;

  // Category breakdown
  const categories: Record<string, CategoryResult> = {};
  const catMap = new Map<string, { total: number; passed: number; failed: number }>();

  for (let i = 0; i < results.length; i++) {
    const cat = tests[i].category;
    const bucket = catMap.get(cat) ?? { total: 0, passed: 0, failed: 0 };
    bucket.total++;
    if (results[i].status === 'passed') bucket.passed++;
    if (results[i].status === 'failed') bucket.failed++;
    catMap.set(cat, bucket);
  }

  for (const [cat, b] of catMap.entries()) {
    categories[cat] = {
      total: b.total, passed: b.passed, failed: b.failed,
      accuracyPercent: b.total > 0 ? Math.round((b.passed / b.total) * 10000) / 100 : 0,
    };
  }

  const answered = tests.length - skipped;
  const toolCallRate = answered > 0 ? Math.round(((answered - noToolCount) / answered) * 10000) / 100 : 0;
  const wrongToolRate = answered > 0 ? Math.round((wrongToolCount / answered) * 10000) / 100 : 0;
  const noToolRate = answered > 0 ? Math.round((noToolCount / answered) * 10000) / 100 : 0;
  const restraintScore = Math.round((1 - wrongToolRate / 100) * 1000) / 1000;

  const result: BenchmarkResult = {
    runId: `lfm-${mode}-k${topK}-${Date.now()}`,
    timestamp: new Date().toISOString(),
    model: model ?? 'LFM2-1.2B-Tool-F16',
    endpoint,
    mode,
    topK,
    totalTests: tests.length,
    passed, failed, skipped,
    accuracyPercent,
    avgLatencyMs: answered > 0 ? Math.round(totalLatency / answered) : 0,
    categories,
    individual: results,
    toolCallRate, wrongToolRate, noToolRate, restraintScore,
    durationMs: totalDurationMs,
    sampling: {
      temperature: greedy ? 0 : 0.1,
      top_p: greedy ? 1.0 : 0.1,
      top_k: greedy ? 0 : 50,
      repetition_penalty: greedy ? 1.0 : 1.05,
      mode: greedy ? 'greedy' : 'near-greedy',
    },
    servers: hasServerFilter ? servers : undefined,
    toolCount,
  };

  if (isFiltered) {
    result.filterHitRate = tests.length > 0
      ? Math.round((filterHitCount / tests.length) * 10000) / 100 : 0;
    result.filterHitCount = filterHitCount;
    result.avgFilteredToolCount = answered > 0
      ? Math.round((totalFilteredTools / answered) * 10) / 10 : 0;
  }

  return result;
}

// ─── Results Output ─────────────────────────────────────────────────────────

function printSummary(result: BenchmarkResult): void {
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`  BENCHMARK RESULTS — ${result.mode.toUpperCase()}`);
  console.log(`${'═'.repeat(70)}`);
  console.log(`  Model:          ${result.model}`);
  console.log(`  Endpoint:       ${result.endpoint}`);
  console.log(`  Mode:           ${result.mode}${result.topK > 0 ? ` (K=${result.topK})` : ''}`);
  if (result.servers && result.servers.length > 0) {
    console.log(`  Servers:        ${result.servers.join(', ')} (${result.toolCount ?? '?'} tools)`);
  }
  console.log(`  Sampling:       ${result.sampling.mode} (temp=${result.sampling.temperature}, top_p=${result.sampling.top_p})`);
  console.log(`  Duration:       ${(result.durationMs / 1000).toFixed(1)}s`);
  console.log(`  Avg Latency:    ${result.avgLatencyMs}ms per test`);
  console.log(`${'─'.repeat(70)}`);
  console.log(`  Total:          ${result.totalTests}`);
  console.log(`  Passed:         ${result.passed}`);
  console.log(`  Failed:         ${result.failed}`);
  console.log(`  Skipped:        ${result.skipped}`);
  console.log(`${'─'.repeat(70)}`);
  console.log(`  ACCURACY:       ${result.accuracyPercent}%`);
  console.log(`  Tool Call Rate: ${result.toolCallRate}% (made any tool call)`);
  console.log(`  Wrong Tool:     ${result.wrongToolRate}% (called wrong tool)`);
  console.log(`  No Tool:        ${result.noToolRate}% (no tool call at all)`);
  console.log(`  Restraint:      ${result.restraintScore} (1.0 = never picks wrong tool)`);

  if (result.filterHitRate !== undefined) {
    console.log(`${'─'.repeat(70)}`);
    console.log(`  FILTER METRICS:`);
    console.log(`  Filter Hit Rate:    ${result.filterHitRate}% (correct tool in top-K)`);
    console.log(`  Filter Hits:        ${result.filterHitCount}/${result.totalTests}`);
    const totalToolRef = result.toolCount ?? VALID_TOOL_NAMES.length;
    console.log(`  Avg Tools Sent:     ${result.avgFilteredToolCount} (of ${totalToolRef})`);
  }

  console.log(`${'─'.repeat(70)}`);

  // Category breakdown
  console.log(`\n  Per-Category Accuracy:`);
  const cats = Object.entries(result.categories)
    .sort((a, b) => b[1].accuracyPercent - a[1].accuracyPercent);
  for (const [cat, data] of cats) {
    const bar = '█'.repeat(Math.round(data.accuracyPercent / 5))
      + '░'.repeat(20 - Math.round(data.accuracyPercent / 5));
    console.log(
      `    ${cat.padEnd(20)} ${bar} ${String(data.accuracyPercent).padStart(6)}%  (${data.passed}/${data.total})`,
    );
  }

  // Decision gate
  console.log(`\n${'═'.repeat(70)}`);
  if (result.accuracyPercent >= 80) {
    console.log(`  ✅ DECISION: PASS (≥80%) — Proceed to Rust implementation`);
  } else if (result.accuracyPercent >= 70) {
    console.log(`  ⚠️  DECISION: MARGINAL (70-80%) — Proceed with tuning`);
  } else if (result.accuracyPercent >= 60) {
    console.log(`  ⚠️  DECISION: MARGINAL (60-70%) — Consider prompt engineering`);
  } else {
    console.log(`  ❌ DECISION: FAIL (<60%) — Model itself is the bottleneck`);
  }
  console.log(`${'═'.repeat(70)}\n`);
}

function saveResults(result: BenchmarkResult): string {
  const __dirname = dirname(fileURLToPath(import.meta.url));
  const resultsDir = join(__dirname, '.results');
  if (!existsSync(resultsDir)) {
    mkdirSync(resultsDir, { recursive: true });
  }
  const filepath = join(resultsDir, `${result.runId}.json`);
  writeFileSync(filepath, JSON.stringify(result, null, 2), 'utf-8');
  return filepath;
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { endpoint, timeoutMs, topK, model, greedy, servers } = parseArgs();

  // Verify model is reachable
  try {
    const healthResp = await fetch(`${endpoint}/v1/models`);
    if (!healthResp.ok) {
      console.error(`❌ Model endpoint not reachable at ${endpoint}/v1/models`);
      process.exit(1);
    }
    const models = (await healthResp.json()) as { data?: Array<{ id: string }> };
    console.log(`✓ Model server reachable. Models: ${JSON.stringify(models.data?.map((m) => m.id) ?? [])}`);
  } catch (err) {
    console.error(`❌ Cannot connect to model at ${endpoint}: ${err}`);
    process.exit(1);
  }

  // Build tool embedding index if filtering is enabled
  let toolIndex: ToolEmbeddingIndex | null = null;

  if (topK > 0) {
    console.log(`\n  Building tool embedding index (${VALID_TOOL_NAMES.length} tools)...`);
    const indexStart = Date.now();
    try {
      const allDefs = buildToolDefinitions();
      toolIndex = await buildToolEmbeddingIndex(endpoint, allDefs);
      console.log(`  ✓ Embedded ${toolIndex.embeddings.length} tools in ${Date.now() - indexStart}ms`);
      console.log(`  Embedding dim: ${toolIndex.embeddings[0]?.length ?? 0}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`\n❌ Failed to build tool embedding index: ${msg}`);
      console.error(`\n   Make sure llama-server was started with --embeddings flag:`);
      console.error(`   llama-server --model <model>.gguf --port 8082 --ctx-size 32768 --embeddings\n`);
      process.exit(1);
    }
  }

  const result = await runBenchmark(endpoint, timeoutMs, topK, toolIndex, model, greedy, servers);

  printSummary(result);

  const filepath = saveResults(result);
  console.log(`Results saved to: ${filepath}`);
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
