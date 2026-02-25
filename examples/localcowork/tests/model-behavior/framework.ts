/**
 * Model Behavior Test Framework for LocalCowork.
 *
 * Provides validation helpers, result aggregation, and model communication
 * utilities for the model behavior test suite.
 */

import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import type {
  ToolSelectionTest,
  MultiStepTest,
  EdgeCaseTest,
  TestRunResults,
  CategoryResult,
  IndividualTestResult,
} from './types';
import { VALID_TOOL_SET } from './types';

/** Shape of the OpenAI-compatible chat completion response we parse. */
interface ChatCompletionToolCall {
  readonly function: {
    readonly name: string;
    readonly arguments: string;
  };
}

interface ChatCompletionResponse {
  readonly choices: ReadonlyArray<{
    readonly message: {
      readonly tool_calls?: readonly ChatCompletionToolCall[];
      readonly content?: string | null;
    };
  }>;
}

/**
 * Validates that a ToolSelectionTest definition is well-formed.
 * Returns an array of error strings (empty if valid).
 */
export function validateToolSelectionTest(test: ToolSelectionTest): readonly string[] {
  const errors: string[] = [];

  if (!test.id || typeof test.id !== 'string') {
    errors.push(`Test missing or invalid id`);
  }
  if (!test.category || typeof test.category !== 'string') {
    errors.push(`[${test.id}] Missing category`);
  }
  if (!test.prompt || typeof test.prompt !== 'string') {
    errors.push(`[${test.id}] Missing prompt`);
  }
  if (!Array.isArray(test.expectedTools) || test.expectedTools.length === 0) {
    errors.push(`[${test.id}] expectedTools must be a non-empty array`);
  } else {
    for (const tool of test.expectedTools) {
      if (!VALID_TOOL_SET.has(tool)) {
        errors.push(`[${test.id}] Invalid tool name: "${tool}"`);
      }
    }
  }
  if (!['easy', 'medium', 'hard'].includes(test.difficulty)) {
    errors.push(`[${test.id}] Invalid difficulty: "${test.difficulty}"`);
  }
  if (test.expectedParamKeys) {
    for (const [toolName, keys] of Object.entries(test.expectedParamKeys)) {
      if (!VALID_TOOL_SET.has(toolName)) {
        errors.push(`[${test.id}] expectedParamKeys references invalid tool: "${toolName}"`);
      }
      if (!Array.isArray(keys)) {
        errors.push(`[${test.id}] expectedParamKeys["${toolName}"] must be an array`);
      }
    }
  }

  return errors;
}

/** Validates a MultiStepTest definition. Returns error strings (empty if valid). */
export function validateMultiStepTest(test: MultiStepTest): readonly string[] {
  const errors: string[] = [];

  if (!test.id || typeof test.id !== 'string') {
    errors.push(`MultiStep test missing or invalid id`);
  }
  if (!test.category || typeof test.category !== 'string') {
    errors.push(`[${test.id}] Missing category`);
  }
  if (!test.scenario || typeof test.scenario !== 'string') {
    errors.push(`[${test.id}] Missing scenario`);
  }
  if (!Array.isArray(test.steps) || test.steps.length < 3) {
    errors.push(`[${test.id}] Must have at least 3 steps, got ${test.steps?.length ?? 0}`);
  } else {
    for (let i = 0; i < test.steps.length; i++) {
      const step = test.steps[i];
      if (!step.description) {
        errors.push(`[${test.id}] Step ${i} missing description`);
      }
      if (!step.prompt) {
        errors.push(`[${test.id}] Step ${i} missing prompt`);
      }
      if (!Array.isArray(step.expectedTools) || step.expectedTools.length === 0) {
        errors.push(`[${test.id}] Step ${i} must have at least one expected tool`);
      } else {
        for (const tool of step.expectedTools) {
          if (!VALID_TOOL_SET.has(tool)) {
            errors.push(`[${test.id}] Step ${i} invalid tool: "${tool}"`);
          }
        }
      }
    }
  }
  if (!['easy', 'medium', 'hard'].includes(test.difficulty)) {
    errors.push(`[${test.id}] Invalid difficulty: "${test.difficulty}"`);
  }

  return errors;
}

/** Validates an EdgeCaseTest definition. Returns error strings (empty if valid). */
export function validateEdgeCaseTest(test: EdgeCaseTest): readonly string[] {
  const errors: string[] = [];

  if (!test.id || typeof test.id !== 'string') {
    errors.push(`EdgeCase test missing or invalid id`);
  }
  if (!test.category || typeof test.category !== 'string') {
    errors.push(`[${test.id}] Missing category`);
  }
  if (!test.prompt || typeof test.prompt !== 'string') {
    errors.push(`[${test.id}] Missing prompt`);
  }
  const validBehaviors = ['clarify', 'fallback', 'refuse', 'suggest_alternative'];
  if (!validBehaviors.includes(test.expectedBehavior)) {
    errors.push(`[${test.id}] Invalid expectedBehavior: "${test.expectedBehavior}"`);
  }
  if (!test.reason || typeof test.reason !== 'string') {
    errors.push(`[${test.id}] Missing reason`);
  }
  if (test.expectedTools) {
    for (const tool of test.expectedTools) {
      if (!VALID_TOOL_SET.has(tool)) {
        errors.push(`[${test.id}] Invalid tool name: "${tool}"`);
      }
    }
  }

  return errors;
}

/** Checks that all test IDs across all collections are unique. */
export function checkUniqueIds(
  toolTests: readonly ToolSelectionTest[],
  multiTests: readonly MultiStepTest[],
  edgeTests: readonly EdgeCaseTest[],
): readonly string[] {
  const seen = new Map<string, string>();
  const errors: string[] = [];

  const check = (id: string, source: string): void => {
    if (seen.has(id)) {
      errors.push(`Duplicate test ID "${id}" found in ${source} and ${seen.get(id)}`);
    } else {
      seen.set(id, source);
    }
  };

  for (const t of toolTests) check(t.id, 'tool-selection');
  for (const t of multiTests) check(t.id, 'multi-step');
  for (const t of edgeTests) check(t.id, 'edge-cases');

  return errors;
}

/** Writes test run results to the .results/ directory as JSON. */
export function writeResults(results: TestRunResults, resultsDir: string): string {
  if (!existsSync(resultsDir)) {
    mkdirSync(resultsDir, { recursive: true });
  }
  const filename = `run-${results.runId}.json`;
  const filepath = join(resultsDir, filename);
  writeFileSync(filepath, JSON.stringify(results, null, 2), 'utf-8');
  return filepath;
}

/** Builds category-level aggregation from individual results. */
export function aggregateByCategory(
  results: readonly IndividualTestResult[],
  testMap: ReadonlyMap<string, string>,
): Readonly<Record<string, CategoryResult>> {
  const buckets = new Map<string, { total: number; passed: number; failed: number }>();

  for (const r of results) {
    const category = testMap.get(r.testId) ?? 'unknown';
    const bucket = buckets.get(category) ?? { total: 0, passed: 0, failed: 0 };
    bucket.total++;
    if (r.status === 'passed') bucket.passed++;
    if (r.status === 'failed') bucket.failed++;
    buckets.set(category, bucket);
  }

  const output: Record<string, CategoryResult> = {};
  for (const [cat, b] of buckets.entries()) {
    output[cat] = {
      total: b.total,
      passed: b.passed,
      failed: b.failed,
      accuracyPercent: b.total > 0 ? Math.round((b.passed / b.total) * 10000) / 100 : 0,
    };
  }
  return output;
}

/**
 * Sends a prompt to the model endpoint and extracts tool calls from the response.
 * Returns the list of tool names the model selected.
 */
export async function sendPromptToModel(
  endpoint: string,
  prompt: string,
  context: readonly string[],
  timeoutMs: number,
): Promise<readonly string[]> {
  const messages: Array<{ role: string; content: string }> = [];

  for (let i = 0; i < context.length; i++) {
    messages.push({
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: context[i],
    });
  }
  messages.push({ role: 'user', content: prompt });

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${endpoint}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages,
        tools: [],
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`Model returned HTTP ${response.status}: ${await response.text()}`);
    }

    const data = (await response.json()) as ChatCompletionResponse;
    const toolCalls = data.choices[0]?.message?.tool_calls ?? [];
    return toolCalls.map((tc) => tc.function.name);
  } finally {
    clearTimeout(timer);
  }
}
