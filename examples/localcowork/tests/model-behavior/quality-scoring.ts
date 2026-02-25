/**
 * Programmatic scoring functions for quality benchmarks.
 *
 * All scoring is deterministic — no LLM judge required.
 * Covers: param extraction accuracy, instruction constraint checking,
 * and synthesis quality verification.
 */

import { VALID_TOOL_SET } from './types';

// ── Param Extraction Scoring ─────────────────────────────────────────────

/** Parse bracket-format args: [tool(key="val", key2=123)] → Record<string, string> */
export function parseBracketArgs(content: string): Record<string, string> {
  const params: Record<string, string> = {};
  // Match [tool.name(args...)]
  const bracketMatch = content.match(/\[([a-z_]+\.[a-z_]+)\(([^)]*)\)\]/);
  if (!bracketMatch) return params;

  const argsStr = bracketMatch[2];
  if (!argsStr.trim()) return params;

  // Parse key="value" or key=value pairs
  const pairPattern = /(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|\[([^\]]*)\]|(\S+))/g;
  let match: RegExpExecArray | null;
  while ((match = pairPattern.exec(argsStr)) !== null) {
    const key = match[1];
    const value = match[2] ?? match[3] ?? match[4] ?? match[5] ?? '';
    params[key] = value;
  }

  return params;
}

/** Extract tool name from bracket-format response. */
export function extractToolName(content: string): string | null {
  const match = content.match(/\[([a-z_]+\.[a-z_]+)\(/);
  if (match && VALID_TOOL_SET.has(match[1])) return match[1];
  return null;
}

/** Normalize a file path for fuzzy comparison. */
export function normalizePath(p: string): string {
  return p
    .replace(/^~\//, '/home/user/')
    .replace(/\/+/g, '/')
    .replace(/\/$/, '')
    .toLowerCase();
}

/** Check if two values match with type-appropriate tolerance. */
export function valuesMatch(
  actual: string,
  expected: string | number | boolean | string[],
  tolerance: 'exact' | 'fuzzy' = 'fuzzy',
): boolean {
  if (typeof expected === 'boolean') {
    return actual.toLowerCase() === String(expected);
  }

  if (typeof expected === 'number') {
    const numActual = parseFloat(actual);
    if (isNaN(numActual)) return false;
    if (tolerance === 'exact') return numActual === expected;
    // Allow ±10% for numeric values
    const diff = Math.abs(numActual - expected);
    return diff <= Math.abs(expected) * 0.1 || diff <= 1;
  }

  if (Array.isArray(expected)) {
    // Check if actual contains all expected items (order-insensitive)
    const actualLower = actual.toLowerCase();
    return expected.every((item) => actualLower.includes(item.toLowerCase()));
  }

  // String comparison
  const actualClean = actual.trim().toLowerCase();
  const expectedClean = String(expected).trim().toLowerCase();

  if (tolerance === 'exact') return actualClean === expectedClean;

  // Fuzzy: path normalization, case-insensitive, partial match
  if (expectedClean.includes('/') || expectedClean.startsWith('~')) {
    return normalizePath(actualClean) === normalizePath(expectedClean)
      || actualClean.includes(expectedClean)
      || expectedClean.includes(actualClean);
  }

  return actualClean === expectedClean
    || actualClean.includes(expectedClean)
    || expectedClean.includes(actualClean);
}

export interface ParamScore {
  readonly toolCorrect: boolean;
  readonly keyRecall: number;      // 0-1: fraction of expected keys present
  readonly valueAccuracy: number;  // 0-1: fraction of present keys with correct values
  readonly hallucinatedKeys: number;
  readonly composite: number;      // 0-1 weighted average
}

export function scoreParamExtraction(
  content: string,
  expectedTool: string,
  expectedParams: Readonly<Record<string, string | number | boolean | string[]>>,
): ParamScore {
  const actualTool = extractToolName(content);
  const toolCorrect = actualTool === expectedTool;

  const actualParams = parseBracketArgs(content);
  const expectedKeys = Object.keys(expectedParams);
  const actualKeys = Object.keys(actualParams);

  if (expectedKeys.length === 0) {
    return { toolCorrect, keyRecall: 1, valueAccuracy: 1, hallucinatedKeys: 0, composite: toolCorrect ? 1 : 0 };
  }

  // Key recall: how many expected keys are present?
  const presentKeys = expectedKeys.filter((k) => actualParams[k] !== undefined);
  const keyRecall = presentKeys.length / expectedKeys.length;

  // Value accuracy: of present keys, how many have correct values?
  let correctValues = 0;
  for (const key of presentKeys) {
    if (valuesMatch(actualParams[key], expectedParams[key])) {
      correctValues++;
    }
  }
  const valueAccuracy = presentKeys.length > 0 ? correctValues / presentKeys.length : 0;

  // Hallucinated keys: keys not in expected set
  const validKeySet = new Set(expectedKeys);
  const hallucinatedKeys = actualKeys.filter((k) => !validKeySet.has(k)).length;

  // Composite: tool 30% + key recall 30% + value accuracy 30% + no hallucination 10%
  const hallucinationPenalty = Math.max(0, 1 - hallucinatedKeys * 0.25);
  const composite = (toolCorrect ? 0.3 : 0) + keyRecall * 0.3 + valueAccuracy * 0.3 + hallucinationPenalty * 0.1;

  return { toolCorrect, keyRecall, valueAccuracy, hallucinatedKeys, composite };
}

// ── Instruction Following Scoring ────────────────────────────────────────

export interface ConstraintDef {
  readonly type: string;
  readonly value: string | number;
  readonly description: string;
}

export interface ConstraintResult {
  readonly type: string;
  readonly description: string;
  readonly passed: boolean;
}

/** Count words in text. */
function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter((w) => w.length > 0).length;
}

/** Count sentences (rough heuristic). */
function sentenceCount(text: string): number {
  return text.split(/[.!?]+/).filter((s) => s.trim().length > 5).length;
}

/** Check if text is valid JSON. */
function isValidJson(text: string): boolean {
  // Extract JSON from response (may have surrounding text)
  const jsonMatch = text.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
  if (!jsonMatch) return false;
  try {
    JSON.parse(jsonMatch[0]);
    return true;
  } catch {
    return false;
  }
}

/** Check if text contains a numbered list. */
function hasNumberedList(text: string): boolean {
  const listPattern = /(?:^|\n)\s*\d+[.)]\s+\S/;
  return listPattern.test(text);
}

/** Check if text contains a bulleted list. */
function hasBulletList(text: string): boolean {
  const bulletPattern = /(?:^|\n)\s*[-*•]\s+\S/;
  return bulletPattern.test(text);
}

export function evaluateConstraint(content: string, constraint: ConstraintDef): ConstraintResult {
  const lower = content.toLowerCase();
  let passed = false;

  switch (constraint.type) {
    case 'contains_keyword':
      passed = lower.includes(String(constraint.value).toLowerCase());
      break;

    case 'excludes_keyword':
      passed = !lower.includes(String(constraint.value).toLowerCase());
      break;

    case 'max_length':
      passed = wordCount(content) <= Number(constraint.value);
      break;

    case 'min_length':
      passed = wordCount(content) >= Number(constraint.value);
      break;

    case 'max_sentences':
      passed = sentenceCount(content) <= Number(constraint.value);
      break;

    case 'format_json':
      passed = isValidJson(content);
      break;

    case 'format_numbered_list':
      passed = hasNumberedList(content);
      break;

    case 'format_bullet_list':
      passed = hasBulletList(content);
      break;

    case 'calls_tool': {
      const toolPattern = /\[([a-z_]+\.[a-z_]+)\(/;
      const match = content.match(toolPattern);
      passed = match !== null && match[1] === String(constraint.value);
      break;
    }

    case 'no_tool_call': {
      const anyTool = /\[([a-z_]+\.[a-z_]+)\(/;
      passed = !anyTool.test(content);
      break;
    }

    case 'correct_count': {
      // Verify the model states the correct number
      const expectedCount = String(constraint.value);
      passed = content.includes(expectedCount);
      break;
    }

    case 'addresses_all_parts': {
      // Value is comma-separated list of keywords that must all appear
      const parts = String(constraint.value).split(',').map((p) => p.trim().toLowerCase());
      passed = parts.every((part) => lower.includes(part));
      break;
    }

    case 'conditional_branch': {
      // Value is the expected branch keyword
      passed = lower.includes(String(constraint.value).toLowerCase());
      break;
    }

    default:
      passed = false;
  }

  return { type: constraint.type, description: constraint.description, passed };
}

export interface InstructionScore {
  readonly constraintsPassed: number;
  readonly totalConstraints: number;
  readonly score: number;           // 0-1
  readonly results: readonly ConstraintResult[];
}

export function scoreInstructionFollowing(
  content: string,
  constraints: readonly ConstraintDef[],
): InstructionScore {
  const results = constraints.map((c) => evaluateConstraint(content, c));
  const constraintsPassed = results.filter((r) => r.passed).length;
  const score = constraints.length > 0 ? constraintsPassed / constraints.length : 0;
  return { constraintsPassed, totalConstraints: constraints.length, score, results };
}

// ── Synthesis Quality Scoring ────────────────────────────────────────────

export interface SynthesisConstraintDef {
  readonly type: string;
  readonly value: string;
  readonly description: string;
}

export function evaluateSynthesisConstraint(
  content: string,
  constraint: SynthesisConstraintDef,
): ConstraintResult {
  const lower = content.toLowerCase();
  let passed = false;

  switch (constraint.type) {
    case 'mentions_key_fact':
      // Value is a fact that must appear (case-insensitive substring)
      passed = lower.includes(constraint.value.toLowerCase());
      break;

    case 'no_hallucination':
      // Value is a keyword that must NOT appear (model inventing data)
      passed = !lower.includes(constraint.value.toLowerCase());
      break;

    case 'acknowledges_limitation':
      // Value is a pattern indicating the model acknowledges missing/empty results
      passed = lower.includes('no result')
        || lower.includes('not found')
        || lower.includes('couldn\'t find')
        || lower.includes('no matching')
        || lower.includes('empty')
        || lower.includes('none')
        || lower.includes('0 result')
        || lower.includes('zero');
      break;

    case 'answers_question':
      // Value is a keyword that should appear in a direct answer
      passed = lower.includes(constraint.value.toLowerCase());
      break;

    case 'correct_count': {
      // Value is the expected number as string
      passed = content.includes(constraint.value);
      break;
    }

    case 'correct_calculation': {
      // Value is the expected numeric result as string
      // Check for the number in various formats (with/without commas, $ signs, etc.)
      const expected = constraint.value;
      const normalizedContent = content.replace(/[,$]/g, '');
      passed = normalizedContent.includes(expected);
      break;
    }

    case 'no_raw_dump': {
      // Verify response doesn't paste raw JSON (long stretches of { } [ ])
      const jsonChunks = content.match(/\{[^}]{50,}\}/g) ?? [];
      const arrayChunks = content.match(/\[[^\]]{50,}\]/g) ?? [];
      passed = jsonChunks.length === 0 && arrayChunks.length === 0;
      break;
    }

    case 'references_source': {
      // Value is a keyword from a specific tool result that must be mentioned
      passed = lower.includes(constraint.value.toLowerCase());
      break;
    }

    default:
      passed = false;
  }

  return { type: constraint.type, description: constraint.description, passed };
}

export interface SynthesisScore {
  readonly constraintsPassed: number;
  readonly totalConstraints: number;
  readonly score: number;
  readonly results: readonly ConstraintResult[];
}

export function scoreSynthesis(
  content: string,
  constraints: readonly SynthesisConstraintDef[],
): SynthesisScore {
  const results = constraints.map((c) => evaluateSynthesisConstraint(content, c));
  const constraintsPassed = results.filter((r) => r.passed).length;
  const score = constraints.length > 0 ? constraintsPassed / constraints.length : 0;
  return { constraintsPassed, totalConstraints: constraints.length, score, results };
}
