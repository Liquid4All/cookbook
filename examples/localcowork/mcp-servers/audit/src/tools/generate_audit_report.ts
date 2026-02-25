/**
 * audit.generate_audit_report — Generate a text audit report for a session.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Reads from the Agent Core's agent.db (audit_log table).
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb, type AuditEntry } from '../db';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  session_id: z.string().describe('Session ID'),
  template: z.string().optional().describe('Report template name'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Report Formatter ───────────────────────────────────────────────────────

function formatDefaultReport(sessionId: string, entries: AuditEntry[]): string {
  const lines: string[] = [];
  // NOTE: Use only ASCII characters in tool output. Multi-byte Unicode
  // (box-drawing, em-dashes) can trigger panics in the Rust string truncation
  // code and wastes tokens the model has to process.
  lines.push('===================================================');
  lines.push(`  AUDIT REPORT -- Session ${sessionId}`);
  lines.push(`  Generated: ${new Date().toISOString()}`);
  lines.push('===================================================');
  lines.push('');

  if (entries.length === 0) {
    lines.push('No audit entries found for this session.');
    return lines.join('\n');
  }

  // Summary
  const toolCounts = new Map<string, number>();
  let userConfirmed = 0;
  let succeeded = 0;
  let failed = 0;

  for (const entry of entries) {
    toolCounts.set(entry.tool_name, (toolCounts.get(entry.tool_name) ?? 0) + 1);
    if (entry.user_confirmed) userConfirmed++;
    if (entry.result_status === 'success') succeeded++;
    if (entry.result_status === 'error') failed++;
  }

  lines.push(`Total tool calls: ${entries.length}`);
  lines.push(`Succeeded: ${succeeded} | Failed: ${failed} | User-confirmed: ${userConfirmed}`);
  lines.push('');

  lines.push('--- Tool Usage ---');
  for (const [tool, count] of toolCounts) {
    lines.push(`  ${tool}: ${count} calls`);
  }
  lines.push('');

  lines.push('--- Timeline ---');
  for (const entry of entries) {
    const time = entry.timestamp.substring(11, 19);
    const duration = entry.execution_time_ms ? ` (${entry.execution_time_ms}ms)` : '';
    lines.push(`  ${time}  ${entry.tool_name} [${entry.result_status}]${duration}`);
  }

  return lines.join('\n');
}

// ─── Tool Definition ────────────────────────────────────────────────────────

export const generateAuditReport: MCPTool<Params> = {
  name: 'audit.generate_audit_report',
  description: 'Generate a text audit report for a session',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
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

      const report = formatDefaultReport(params.session_id, entries);

      return {
        success: true,
        data: { report },
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to generate report: ${msg}`);
    }
  },
};
