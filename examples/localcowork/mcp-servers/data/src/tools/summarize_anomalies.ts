/**
 * data.summarize_anomalies — Detect anomalies in a dataset based on rules.
 *
 * Non-destructive: returns anomaly list, no confirmation needed.
 * Supports statistical (z-score), range, and custom rules.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';

// ─── Types ──────────────────────────────────────────────────────────────────

/** Anomaly detection rule */
interface AnomalyRule {
  field: string;
  type: 'range' | 'z_score' | 'missing' | 'pattern';
  min?: number;
  max?: number;
  z_threshold?: number;
  pattern?: string;
}

/** A detected anomaly */
interface Anomaly {
  row_index: number;
  field: string;
  value: unknown;
  rule: string;
  message: string;
}

// ─── Params Schema ──────────────────────────────────────────────────────────

const ruleSchema = z.object({
  field: z.string().describe('Field to check'),
  type: z
    .enum(['range', 'z_score', 'missing', 'pattern'])
    .describe('Anomaly detection type'),
  min: z.number().optional().describe('Minimum value (for range)'),
  max: z.number().optional().describe('Maximum value (for range)'),
  z_threshold: z
    .number()
    .optional()
    .default(2)
    .describe('Z-score threshold (default 2)'),
  pattern: z.string().optional().describe('Regex pattern (for pattern type)'),
});

const paramsSchema = z.object({
  data: z
    .array(z.record(z.string(), z.unknown()))
    .min(1)
    .describe('Input records'),
  rules: z
    .array(ruleSchema)
    .optional()
    .describe('Custom anomaly rules (auto-detect if omitted)'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Extract numeric values from a dataset column */
function extractNumericValues(
  data: Record<string, unknown>[],
  field: string,
): { values: number[]; indices: number[] } {
  const values: number[] = [];
  const indices: number[] = [];

  for (let i = 0; i < data.length; i++) {
    const v = data[i][field];
    if (typeof v === 'number' && !Number.isNaN(v)) {
      values.push(v);
      indices.push(i);
    }
  }

  return { values, indices };
}

/** Compute mean and standard deviation */
function computeStats(values: number[]): { mean: number; stddev: number } {
  if (values.length === 0) return { mean: 0, stddev: 0 };
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  return { mean, stddev: Math.sqrt(variance) };
}

/** Auto-detect rules for numeric columns */
function autoDetectRules(data: Record<string, unknown>[]): AnomalyRule[] {
  const firstRow = data[0] ?? {};
  const rules: AnomalyRule[] = [];

  for (const field of Object.keys(firstRow)) {
    const { values } = extractNumericValues(data, field);

    // Only generate z-score rules for numeric columns with enough data
    if (values.length >= 5) {
      rules.push({ field, type: 'z_score', z_threshold: 2 });
    }

    // Add missing-value checks for all columns
    rules.push({ field, type: 'missing' });
  }

  return rules;
}

/** Apply a single rule to the dataset and return anomalies */
function applyRule(
  data: Record<string, unknown>[],
  rule: AnomalyRule,
): Anomaly[] {
  const anomalies: Anomaly[] = [];

  switch (rule.type) {
    case 'range': {
      for (let i = 0; i < data.length; i++) {
        const v = data[i][rule.field];
        if (typeof v !== 'number') continue;
        if (rule.min !== undefined && v < rule.min) {
          anomalies.push({
            row_index: i,
            field: rule.field,
            value: v,
            rule: 'range',
            message: `Value ${v} is below minimum ${rule.min}`,
          });
        }
        if (rule.max !== undefined && v > rule.max) {
          anomalies.push({
            row_index: i,
            field: rule.field,
            value: v,
            rule: 'range',
            message: `Value ${v} is above maximum ${rule.max}`,
          });
        }
      }
      break;
    }

    case 'z_score': {
      const { values, indices } = extractNumericValues(data, rule.field);
      const { mean, stddev } = computeStats(values);
      const threshold = rule.z_threshold ?? 2;

      if (stddev === 0) break; // No variation, nothing to flag

      for (let k = 0; k < values.length; k++) {
        const zScore = Math.abs((values[k] - mean) / stddev);
        if (zScore > threshold) {
          anomalies.push({
            row_index: indices[k],
            field: rule.field,
            value: values[k],
            rule: 'z_score',
            message: `Z-score ${zScore.toFixed(2)} exceeds threshold ${threshold} (mean=${mean.toFixed(2)}, stddev=${stddev.toFixed(2)})`,
          });
        }
      }
      break;
    }

    case 'missing': {
      for (let i = 0; i < data.length; i++) {
        const v = data[i][rule.field];
        if (v === null || v === undefined || v === '') {
          anomalies.push({
            row_index: i,
            field: rule.field,
            value: v,
            rule: 'missing',
            message: `Missing value in field "${rule.field}"`,
          });
        }
      }
      break;
    }

    case 'pattern': {
      if (!rule.pattern) break;
      let regex: RegExp;
      try {
        regex = new RegExp(rule.pattern);
      } catch {
        break; // Invalid regex — skip
      }

      for (let i = 0; i < data.length; i++) {
        const v = data[i][rule.field];
        if (typeof v !== 'string') continue;
        if (!regex.test(v)) {
          anomalies.push({
            row_index: i,
            field: rule.field,
            value: v,
            rule: 'pattern',
            message: `Value "${v}" does not match pattern /${rule.pattern}/`,
          });
        }
      }
      break;
    }
  }

  return anomalies;
}

// ─── Tool Definition ────────────────────────────────────────────────────────

export const summarizeAnomalies: MCPTool<Params> = {
  name: 'data.summarize_anomalies',
  description: 'Detect anomalies in a dataset based on rules',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const data = params.data;
      const rules = params.rules ?? autoDetectRules(data);

      const allAnomalies: Anomaly[] = [];

      for (const rule of rules) {
        const found = applyRule(data, rule);
        allAnomalies.push(...found);
      }

      // Sort by row_index for readable output
      allAnomalies.sort((a, b) => a.row_index - b.row_index);

      return {
        success: true,
        data: { anomalies: allAnomalies },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to detect anomalies: ${msg}`);
    }
  },
};
