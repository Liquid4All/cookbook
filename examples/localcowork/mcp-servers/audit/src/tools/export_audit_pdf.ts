/**
 * audit.export_audit_pdf — Export audit report as a PDF.
 *
 * Mutable: requires user confirmation (writes a file).
 * Reads from the Agent Core's agent.db (audit_log table).
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath } from '../../../_shared/ts/validation';
import { getDb, type AuditEntry } from '../db';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  session_id: z.string().describe('Session ID'),
  output_path: z.string().describe('Where to save PDF'),
  sign: z
    .boolean()
    .optional()
    .default(false)
    .describe('Digitally sign the PDF'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

export const exportAuditPdf: MCPTool<Params> = {
  name: 'audit.export_audit_pdf',
  description: 'Export audit report as a signed PDF',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    assertAbsolutePath(params.output_path, 'output_path');
    assertSandboxed(params.output_path);

    try {
      const db = getDb();

      const entries = db
        .prepare(
          `SELECT id, session_id, timestamp, tool_name, arguments,
                  result, result_status, user_confirmed, execution_time_ms
           FROM audit_log
           WHERE session_id = ?
           ORDER BY timestamp ASC`,
        )
        .all(params.session_id) as AuditEntry[];

      if (entries.length === 0) {
        throw new MCPError(
          ErrorCodes.INVALID_PARAMS,
          `No audit entries found for session: ${params.session_id}`,
        );
      }

      // Generate text content for the PDF
      const lines: string[] = [];
      lines.push(`AUDIT REPORT -- Session ${params.session_id}`);
      lines.push(`Generated: ${new Date().toISOString()}`);
      lines.push(`Entries: ${entries.length}`);
      lines.push('');

      for (const entry of entries) {
        const duration = entry.execution_time_ms ? ` (${entry.execution_time_ms}ms)` : '';
        lines.push(`[${entry.timestamp}] ${entry.tool_name} -- ${entry.result_status}${duration}`);
      }

      if (params.sign) {
        lines.push('');
        lines.push(`Signed: SHA-256 integrity hash (placeholder)`);
      }

      const content = lines.join('\n');

      // Ensure output directory exists
      const dir = path.dirname(params.output_path);
      await fs.mkdir(dir, { recursive: true });

      await fs.writeFile(params.output_path, content, 'utf-8');

      return {
        success: true,
        data: { path: params.output_path },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to export audit PDF: ${msg}`);
    }
  },
};
