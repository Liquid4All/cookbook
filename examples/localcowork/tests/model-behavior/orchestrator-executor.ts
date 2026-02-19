/**
 * Orchestrator Executor Module — runs each plan step through the router model
 * (LFM2-1.2B-Tool) with RAG pre-filtered tools.
 *
 * Mirrors the Rust execute_step() in orchestrator.rs.
 */

import type { MultiStepEntry } from './types';
import type { PlanStep, StepPlan } from './orchestrator-types';
import type { OrchestratorStepResult } from './orchestrator-types';
import type { StepMapping } from './orchestrator-planner';
import {
  filterToolsByRelevance,
  buildFilteredToolDefinitions,
  parseLfmToolCalls,
  isDeflection,
  getMockResult,
  TOOL_DESCRIPTIONS,
} from './benchmark-shared';
import type { ToolEmbeddingIndex, ChatMessage } from './benchmark-shared';

// ─── Router System Prompt (identical to orchestrator.rs) ───────────────────

const ROUTER_SYSTEM_PROMPT =
  'You are a tool-calling assistant. Select the most appropriate tool and call it with the correct parameters. ALWAYS call a tool. Never respond with text only.';

// ─── Prior Result Interpolation ────────────────────────────────────────────

/** Replace "step N" references with actual prior results. Mirrors Rust interpolate_prior_results(). */
export function interpolatePriorResults(
  description: string,
  priorResults: OrchestratorStepResult[],
): string {
  let result = description;
  for (const prior of priorResults) {
    if (prior.status !== 'passed') continue;
    const stepRef = `step ${prior.stepIndex + 1}`;
    if (result.toLowerCase().includes(stepRef)) {
      const truncated = prior.mockResult.length > 2000
        ? `${prior.mockResult.slice(0, 2000)}... [truncated]`
        : prior.mockResult;
      result = `${result}\n\n[Context from previous ${stepRef}]:\n${truncated}`;
    }
  }
  return result;
}

// ─── Single Step Execution ─────────────────────────────────────────────────

/** Execute a single plan step using the router model with RAG pre-filtering. */
export async function executeStep(
  step: PlanStep,
  stepIndex: number,
  priorResults: OrchestratorStepResult[],
  routerEndpoint: string,
  toolIndex: ToolEmbeddingIndex,
  topK: number,
  stepRetries: number,
  expectedTools: readonly string[],
): Promise<OrchestratorStepResult> {
  const startTime = Date.now();
  const description = interpolatePriorResults(step.description, priorResults);

  // RAG pre-filter
  let filteredNames: string[];
  try {
    const { selectedTools } = await filterToolsByRelevance(
      routerEndpoint, description, toolIndex, topK,
    );
    filteredNames = selectedTools;
  } catch (e) {
    return {
      stepIndex,
      planDescription: step.description,
      expectedTools,
      actualTools: [],
      status: 'failed',
      failureReason: 'error',
      filterHit: false,
      filteredTools: [],
      rawContent: '',
      mockResult: '',
      durationMs: Date.now() - startTime,
      retries: 0,
    };
  }

  const filterHit = expectedTools.some((t) => filteredNames.includes(t));
  const filteredDefs = buildFilteredToolDefinitions(filteredNames);
  const toolsJson = JSON.stringify(filteredDefs);

  // Build system prompt with filtered tools
  const systemContent = `${ROUTER_SYSTEM_PROMPT}\n\nAvailable tools:\n${filteredDefs.map((t) => `- ${t.name}: ${t.description}`).join('\n')}`;

  // Retry loop
  for (let attempt = 0; attempt < stepRetries; attempt++) {
    const prompt = attempt === 0
      ? description
      : `${description}\n\nYou MUST call a tool. Select from: ${filteredNames.slice(0, 5).join(', ')}`;

    const messages: ChatMessage[] = [
      { role: 'system', content: systemContent },
      { role: 'user', content: prompt },
    ];

    let content: string;
    try {
      const response = await fetch(`${routerEndpoint}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages,
          temperature: 0.1,
          top_p: 0.1,
          max_tokens: 512,
          stream: false,
        }),
      });

      if (!response.ok) continue;

      const data = (await response.json()) as {
        choices: Array<{ message: { content?: string; tool_calls?: Array<{ function: { name: string } }> } }>;
      };
      content = data.choices?.[0]?.message?.content ?? '';

      // Check native tool_calls first, then bracket parser
      let actualTools: string[] = [];
      if (data.choices[0]?.message?.tool_calls?.length) {
        actualTools = data.choices[0].message.tool_calls.map((tc) => tc.function.name);
      }
      if (actualTools.length === 0) {
        actualTools = parseLfmToolCalls(content);
      }

      if (actualTools.length > 0) {
        const toolName = actualTools[0];
        const passed = expectedTools.some((t) => actualTools.includes(t));
        const mockResult = getMockResult(toolName);

        let failureReason: OrchestratorStepResult['failureReason'];
        if (!passed) {
          if (!filterHit) failureReason = 'filter_miss';
          else failureReason = 'wrong_tool';
        }

        return {
          stepIndex,
          planDescription: step.description,
          expectedTools,
          actualTools,
          status: passed ? 'passed' : 'failed',
          failureReason,
          filterHit,
          filteredTools: filteredNames,
          rawContent: content.slice(0, 200),
          mockResult,
          durationMs: Date.now() - startTime,
          retries: attempt,
        };
      }
    } catch {
      continue;
    }
  }

  // All retries exhausted — no tool call produced
  return {
    stepIndex,
    planDescription: step.description,
    expectedTools,
    actualTools: [],
    status: 'failed',
    failureReason: 'no_tool',
    filterHit,
    filteredTools: filteredNames,
    rawContent: '',
    mockResult: '',
    durationMs: Date.now() - startTime,
    retries: stepRetries,
  };
}

// ─── Full Execution Phase ──────────────────────────────────────────────────

/**
 * Execute all plan steps, evaluating each against the test expected tools.
 * Detects critical failures: if a step fails and later steps reference it.
 */
export async function executeAllSteps(
  plan: StepPlan,
  testSteps: readonly MultiStepEntry[],
  mapping: StepMapping,
  routerEndpoint: string,
  toolIndex: ToolEmbeddingIndex,
  topK: number,
  stepRetries: number,
): Promise<OrchestratorStepResult[]> {
  const results: OrchestratorStepResult[] = [];

  for (let planIdx = 0; planIdx < plan.steps.length; planIdx++) {
    const step = plan.steps[planIdx];
    const expectedTools = mapping.planExpected.get(planIdx) ?? [];

    const result = await executeStep(
      step, planIdx, results, routerEndpoint, toolIndex,
      topK, stepRetries, expectedTools,
    );

    // Critical failure detection: if this step failed and later steps reference it
    if (!result.status || result.status === 'failed') {
      const stepRef = `step ${step.step_number}`;
      const isCritical = plan.steps.some((s) =>
        s.step_number > step.step_number
        && s.description.toLowerCase().includes(stepRef),
      );

      if (isCritical) {
        result.failureReason = 'critical_failure';
        results.push(result);
        break;
      }
    }

    results.push(result);
  }

  return results;
}
