/**
 * data.deduplicate_records — Find and flag duplicate records in a dataset.
 *
 * Non-destructive: returns unique/duplicate groups, no confirmation needed.
 * Uses Levenshtein similarity on string fields, exact match on numbers/dates.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  data: z
    .array(z.record(z.string(), z.unknown()))
    .min(1)
    .describe('Input records'),
  match_fields: z
    .array(z.string())
    .min(1)
    .describe('Fields to match on'),
  threshold: z
    .number()
    .min(0)
    .max(1)
    .optional()
    .default(0.85)
    .describe('Similarity threshold (0-1)'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Compute Levenshtein distance between two strings */
function levenshteinDistance(a: string, b: string): number {
  const aLen = a.length;
  const bLen = b.length;

  if (aLen === 0) return bLen;
  if (bLen === 0) return aLen;

  // Use single-row optimization for memory efficiency
  let prev = Array.from({ length: bLen + 1 }, (_, i) => i);
  let curr = new Array<number>(bLen + 1);

  for (let i = 1; i <= aLen; i++) {
    curr[0] = i;
    for (let j = 1; j <= bLen; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        prev[j] + 1, // deletion
        curr[j - 1] + 1, // insertion
        prev[j - 1] + cost, // substitution
      );
    }
    [prev, curr] = [curr, prev];
  }

  return prev[bLen];
}

/** Compute similarity ratio (0-1) between two strings */
function stringSimilarity(a: string, b: string): number {
  if (a === b) return 1;
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1;
  return 1 - levenshteinDistance(a, b) / maxLen;
}

/** Compute field-level similarity between two records */
function recordSimilarity(
  a: Record<string, unknown>,
  b: Record<string, unknown>,
  fields: string[],
): number {
  if (fields.length === 0) return 0;

  let totalSimilarity = 0;

  for (const field of fields) {
    const valA = a[field];
    const valB = b[field];

    if (valA === valB) {
      totalSimilarity += 1;
      continue;
    }

    if (valA === null || valA === undefined || valB === null || valB === undefined) {
      totalSimilarity += 0;
      continue;
    }

    // String comparison: Levenshtein similarity
    if (typeof valA === 'string' && typeof valB === 'string') {
      totalSimilarity += stringSimilarity(
        valA.toLowerCase().trim(),
        valB.toLowerCase().trim(),
      );
      continue;
    }

    // Number comparison: exact match
    if (typeof valA === 'number' && typeof valB === 'number') {
      totalSimilarity += valA === valB ? 1 : 0;
      continue;
    }

    // Fallback: string coercion comparison
    totalSimilarity += String(valA) === String(valB) ? 1 : 0;
  }

  return totalSimilarity / fields.length;
}

// ─── Tool Definition ────────────────────────────────────────────────────────

export const deduplicateRecords: MCPTool<Params> = {
  name: 'data.deduplicate_records',
  description: 'Find and flag duplicate records in a dataset',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const data = params.data;
      const matchFields = params.match_fields;
      const threshold = params.threshold ?? 0.85;

      // Validate that match_fields exist in data
      const firstRow = data[0] ?? {};
      for (const field of matchFields) {
        if (!(field in firstRow)) {
          throw new MCPError(
            ErrorCodes.INVALID_PARAMS,
            `Field "${field}" not found in data. Available: ${Object.keys(firstRow).join(', ')}`,
          );
        }
      }

      // Track which records have been matched as duplicates
      const matched = new Set<number>();
      const duplicateGroups: Record<string, unknown>[][] = [];
      const unique: Record<string, unknown>[] = [];

      for (let i = 0; i < data.length; i++) {
        if (matched.has(i)) continue;

        const group: Record<string, unknown>[] = [];

        for (let j = i + 1; j < data.length; j++) {
          if (matched.has(j)) continue;

          const sim = recordSimilarity(data[i], data[j], matchFields);
          if (sim >= threshold) {
            if (group.length === 0) {
              group.push(data[i]);
            }
            group.push(data[j]);
            matched.add(j);
          }
        }

        if (group.length > 0) {
          matched.add(i);
          duplicateGroups.push(group);
        } else {
          unique.push(data[i]);
        }
      }

      return {
        success: true,
        data: { unique, duplicates: duplicateGroups },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to deduplicate records: ${msg}`);
    }
  },
};
