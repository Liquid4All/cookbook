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

/**
 * Enhanced prior result interpolation (M3). Mirrors Rust interpolate_prior_results().
 * Three mechanisms:
 * 1. Always forward predecessor (step N-1) result
 * 2. Forward explicitly referenced steps ("step M")
 * 3. Dedup so no step is forwarded twice
 */
export function interpolatePriorResults(
  stepNumber: number,
  description: string,
  priorResults: OrchestratorStepResult[],
): string {
  if (priorResults.length === 0) return description;

  const includedSteps: number[] = [];
  const lines: string[] = [];

  // 1. Always forward predecessor (step N-1) if it succeeded
  const predecessorIdx = stepNumber - 1;
  if (predecessorIdx >= 0 && predecessorIdx < priorResults.length) {
    const pred = priorResults[predecessorIdx];
    if (pred.status === 'passed') {
      const summary = condenseMockResult(pred);
      lines.push(`[Result from step ${predecessorIdx + 1}]: ${summary}`);
      includedSteps.push(predecessorIdx);
    }
  }

  // 2. Explicit "step N" references in description
  const descLower = description.toLowerCase();
  for (const prior of priorResults) {
    if (prior.status !== 'passed') continue;
    if (includedSteps.includes(prior.stepIndex)) continue;
    const stepRef = `step ${prior.stepIndex + 1}`;
    if (descLower.includes(stepRef)) {
      const summary = condenseMockResult(prior);
      lines.push(`[Result from ${stepRef}]: ${summary}`);
      includedSteps.push(prior.stepIndex);
    }
  }

  if (lines.length === 0) return description;
  return `${description}\n\n[Prior step context]:\n${lines.join('\n')}`;
}

/** Condense a step result for forwarding — keep short results intact, truncate long ones. */
function condenseMockResult(result: OrchestratorStepResult): string {
  if (result.mockResult.length <= 200) return result.mockResult;
  return `${result.mockResult.slice(0, 150)}... (${result.mockResult.length} chars total)`;
}

// ─── Single Step Execution ─────────────────────────────────────────────────

/** Execute a single plan step using the router model with RAG pre-filtering. */
export async function executeStep(
  step: PlanStep,
  stepIndex: number,
  priorResults: OrchestratorStepResult[],
  routerEndpoint: string,
  routerModel: string,
  toolIndex: ToolEmbeddingIndex,
  topK: number,
  stepRetries: number,
  expectedTools: readonly string[],
): Promise<OrchestratorStepResult> {
  const startTime = Date.now();
  const description = interpolatePriorResults(stepIndex, step.description, priorResults);

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
      const body: Record<string, unknown> = {
        messages,
        temperature: 0.1,
        top_p: 0.1,
        max_tokens: 512,
        stream: false,
      };
      if (routerModel) body.model = routerModel;

      const response = await fetch(`${routerEndpoint}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
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
  routerModel: string,
  toolIndex: ToolEmbeddingIndex,
  topK: number,
  stepRetries: number,
): Promise<OrchestratorStepResult[]> {
  const results: OrchestratorStepResult[] = [];

  for (let planIdx = 0; planIdx < plan.steps.length; planIdx++) {
    const step = plan.steps[planIdx];
    const expectedTools = mapping.planExpected.get(planIdx) ?? [];

    const result = await executeStep(
      step, planIdx, results, routerEndpoint, routerModel, toolIndex,
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
