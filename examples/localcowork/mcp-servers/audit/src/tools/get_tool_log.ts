/**
 * audit.get_tool_log — Retrieve tool execution log entries.
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
  session_id: z.string().optional().describe('Filter by session'),
  start_time: z.string().optional().describe('Start of time range (ISO 8601)'),
  end_time: z.string().optional().describe('End of time range (ISO 8601)'),
  tool_name: z.string().optional().describe('Filter by tool name'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

export const getToolLog: MCPTool<Params> = {
  name: 'audit.get_tool_log',
  description: 'Retrieve tool execution log entries',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();
      const conditions: string[] = [];
      const values: unknown[] = [];

      if (params.session_id) {
        conditions.push('session_id = ?');
        values.push(params.session_id);
      }

      if (params.start_time) {
        conditions.push('timestamp >= ?');
        values.push(params.start_time);
      }

      if (params.end_time) {
        conditions.push('timestamp <= ?');
        values.push(params.end_time);
      }

      if (params.tool_name) {
        conditions.push('tool_name = ?');
        values.push(params.tool_name);
      }

      const where = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';
      const sql = `SELECT id, session_id, timestamp, tool_name, arguments,
                          result, result_status, user_confirmed, execution_time_ms
                   FROM audit_log ${where}
                   ORDER BY timestamp DESC LIMIT 1000`;
      const rows = db.prepare(sql).all(...values) as AuditEntry[];

      return { success: true, data: rows };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to query audit log: ${msg}`);
    }
  },
};
