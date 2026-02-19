/**
 * data.write_csv — Write structured data to a CSV file.
 *
 * Mutable: requires user confirmation (writes a file).
 * Creates parent directories if needed.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath } from '../../../_shared/ts/validation';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  data: z
    .array(z.record(z.string(), z.unknown()))
    .min(1)
    .describe('Array of row objects'),
  output_path: z.string().describe('Where to save CSV'),
  headers: z
    .array(z.string())
    .optional()
    .describe('Column headers (auto-detected if omitted)'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Escape a CSV field value per RFC 4180 */
function escapeCsvField(value: unknown): string {
  if (value === null || value === undefined) return '';
  const str = String(value);
  if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

// ─── Tool Definition ────────────────────────────────────────────────────────

export const writeCsv: MCPTool<Params> = {
  name: 'data.write_csv',
  description: 'Write structured data to a CSV file',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    const outputPath = params.output_path;
    assertAbsolutePath(outputPath, 'output_path');
    assertSandboxed(outputPath);

    try {
      const data = params.data;

      // Determine headers: explicit or auto-detected from first row keys
      const headers = params.headers ?? Object.keys(data[0] ?? {});

      if (headers.length === 0) {
        throw new MCPError(ErrorCodes.INVALID_PARAMS, 'No columns detected in data');
      }

      // Build CSV lines
      const lines: string[] = [];

      // Header row
      lines.push(headers.map(escapeCsvField).join(','));

      // Data rows
      for (const row of data) {
        const fields = headers.map((h) => escapeCsvField(row[h]));
        lines.push(fields.join(','));
      }

      const content = lines.join('\n') + '\n';

      // Ensure output directory exists
      const dir = path.dirname(outputPath);
      await fs.mkdir(dir, { recursive: true });

      await fs.writeFile(outputPath, content, 'utf-8');

      return {
        success: true,
        data: { path: outputPath, rows: data.length },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to write CSV: ${msg}`);
    }
  },
};
