/**
 * Orchestrator Planner Module — calls the planner model (Qwen3-30B-A3B)
 * to decompose a scenario into self-contained tool-calling steps.
 *
 * Mirrors the Rust plan_steps() in orchestrator.rs.
 */

import type { MultiStepEntry } from './types';
import type { PlanStep, StepPlan } from './orchestrator-types';
import type { ChatMessage } from './benchmark-shared';

// ─── Planner System Prompt (identical to orchestrator.rs) ──────────────────

const PLANNER_SYSTEM_PROMPT = `You are a task planner for LocalCowork. Given a user request, decompose it into a sequence of tool-calling steps. You do NOT call tools yourself. Output ONLY valid JSON.

Available capability areas (servers):
- filesystem: list, read, write, move, copy, delete, search files
- document: extract text from PDF/DOCX, convert formats, diff, create PDF/DOCX
- ocr: extract text from images/screenshots, extract structured data
- data: CSV/SQLite operations, deduplication, anomaly detection
- knowledge: semantic search across indexed documents, RAG Q&A
- security: PII/secrets scanning, file encryption, duplicate finding
- task: create/update/list tasks, daily briefing
- calendar: list events, create events, find free slots
- email: draft/send emails, search, summarize threads
- meeting: transcribe audio, extract action items, generate minutes
- audit: tool usage logs, session summaries
- clipboard: read/write system clipboard
- system: system info, open apps, take screenshots

Rules:
1. Output ONLY valid JSON. No prose before or after.
2. If the request does NOT require tools, set needs_tools=false and provide direct_response.
3. Each step description must be COMPLETE and self-contained.
4. Include file paths, search terms, and specifics from the user message in each step.
5. For steps needing a prior result, write: "Using the result from step N, ..."
6. Maximum 10 steps.

JSON schema:
{"needs_tools":bool,"direct_response":string|null,"steps":[{"step_number":int,"description":"...","expected_server":"..."|null,"hint_params":{...}|null}]}`;

// ─── Planner Call ──────────────────────────────────────────────────────────

export interface PlanResult {
  plan: StepPlan | null;
  rawResponse: string;
  durationMs: number;
  error?: string;
}

/** Call the planner model to decompose a scenario into steps. */
export async function callPlanner(
  plannerEndpoint: string,
  scenario: string,
): Promise<PlanResult> {
  const messages: ChatMessage[] = [
    { role: 'system', content: PLANNER_SYSTEM_PROMPT },
    { role: 'user', content: scenario },
  ];

  const startTime = Date.now();

  try {
    const response = await fetch(`${plannerEndpoint}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages,
        temperature: 0.1,
        top_p: 0.2,
        max_tokens: 4096,
        stream: false,
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      return { plan: null, rawResponse: text, durationMs: Date.now() - startTime, error: `HTTP ${response.status}: ${text}` };
    }

    const data = (await response.json()) as {
      choices: Array<{ message: { content?: string } }>;
    };
    const content = data.choices?.[0]?.message?.content ?? '';
    const durationMs = Date.now() - startTime;
    const jsonStr = extractJson(content);

    try {
      const plan = JSON.parse(jsonStr) as StepPlan;
      return { plan, rawResponse: content, durationMs };
    } catch (e) {
      return { plan: null, rawResponse: content, durationMs, error: `JSON parse error: ${e}` };
    }
  } catch (e) {
    return { plan: null, rawResponse: '', durationMs: Date.now() - startTime, error: `Network error: ${e}` };
  }
}

// ─── JSON Extraction ───────────────────────────────────────────────────────

/** Extract JSON from text that may be wrapped in markdown code fences. */
export function extractJson(text: string): string {
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start !== -1 && end !== -1 && end > start) {
    return text.slice(start, end + 1);
  }
  return text;
}

// ─── Plan Validation ───────────────────────────────────────────────────────

/** Validate the plan has the expected structure. */
export function validatePlan(
  plan: StepPlan,
  maxSteps: number,
): { valid: boolean; reason?: string } {
  if (plan.needs_tools === undefined) {
    return { valid: false, reason: 'missing needs_tools field' };
  }
  if (!plan.needs_tools) return { valid: true };
  if (!plan.steps || plan.steps.length === 0) {
    return { valid: false, reason: 'needs_tools=true but no steps' };
  }
  if (plan.steps.length > maxSteps) {
    return { valid: false, reason: `too many steps: ${plan.steps.length} > ${maxSteps}` };
  }
  for (const step of plan.steps) {
    if (!step.description || step.description.trim().length === 0) {
      return { valid: false, reason: `step ${step.step_number} has empty description` };
    }
  }
  return { valid: true };
}

// ─── Plan-to-Test Mapping ──────────────────────────────────────────────────

/**
 * Map plan steps to test expected tools using bag-of-tools matching.
 *
 * For each test step, finds the plan step whose description best matches
 * by expected_server hint or keyword overlap with expected tool names.
 * Returns a map: testStepIndex → planStepIndex (or -1 if unmapped).
 *
 * Also returns which expected tools each plan step should be evaluated against.
 */
export interface StepMapping {
  /** testStepIndex → planStepIndex (-1 if unmapped) */
  testToPlan: Map<number, number>;
  /** planStepIndex → expectedTools from the test */
  planExpected: Map<number, readonly string[]>;
}

export function mapPlanToExpectedSteps(
  plan: StepPlan,
  testSteps: readonly MultiStepEntry[],
): StepMapping {
  const testToPlan = new Map<number, number>();
  const planExpected = new Map<number, readonly string[]>();
  const usedPlanSteps = new Set<number>();

  for (let testIdx = 0; testIdx < testSteps.length; testIdx++) {
    const testStep = testSteps[testIdx];
    const expectedServer = testStep.expectedTools[0]?.split('.')[0] ?? '';

    let bestPlanIdx = -1;
    let bestScore = 0;

    for (let planIdx = 0; planIdx < plan.steps.length; planIdx++) {
      if (usedPlanSteps.has(planIdx)) continue;
      const planStep = plan.steps[planIdx];
      let score = 0;

      // Score 1: expected_server hint matches
      if (planStep.expected_server && planStep.expected_server === expectedServer) {
        score += 3;
      }

      // Score 2: description contains the server name
      const descLower = planStep.description.toLowerCase();
      if (descLower.includes(expectedServer)) score += 2;

      // Score 3: description contains any expected tool name
      for (const tool of testStep.expectedTools) {
        const toolParts = tool.split('.');
        if (toolParts.length === 2 && descLower.includes(toolParts[1].replace(/_/g, ' '))) {
          score += 2;
        }
        if (descLower.includes(tool)) score += 3;
      }

      // Score 4: keyword overlap with test step description
      const testWords = testStep.description.toLowerCase().split(/\s+/);
      for (const word of testWords) {
        if (word.length > 3 && descLower.includes(word)) score += 1;
      }

      if (score > bestScore) {
        bestScore = score;
        bestPlanIdx = planIdx;
      }
    }

    testToPlan.set(testIdx, bestPlanIdx);
    if (bestPlanIdx >= 0) {
      usedPlanSteps.add(bestPlanIdx);
      planExpected.set(bestPlanIdx, testStep.expectedTools);
    }
  }

  return { testToPlan, planExpected };
}
