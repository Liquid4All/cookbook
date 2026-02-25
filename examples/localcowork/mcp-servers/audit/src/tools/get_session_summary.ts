/**
 * audit.get_session_summary — Aggregate summary for a session.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Reads from the Agent Core's agent.db (audit_log table).
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb } from '../db';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  session_id: z.string().describe('Session ID'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Types ──────────────────────────────────────────────────────────────────

interface ToolSummary {
  tool_name: string;
  call_count: number;
}

// ─── Tool Definition ────────────────────────────────────────────────────────

export const getSessionSummary: MCPTool<Params> = {
  name: 'audit.get_session_summary',
  description: 'Aggregate summary for a session',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();

      // Get tool call summary
      const toolSummaries = db
        .prepare(
          `SELECT tool_name, COUNT(*) as call_count
           FROM audit_log
           WHERE session_id = ?
           GROUP BY tool_name
           ORDER BY call_count DESC`,
        )
        .all(params.session_id) as ToolSummary[];

      // Get success/error counts — agent core writes lowercase via AuditStatus::as_str()
      const succeeded = db
        .prepare(
          `SELECT COUNT(*) as count
           FROM audit_log
           WHERE session_id = ? AND result_status = 'success'`,
        )
        .get(params.session_id) as { count: number };

      const failed = db
        .prepare(
          `SELECT COUNT(*) as count
           FROM audit_log
           WHERE session_id = ? AND result_status = 'error'`,
        )
        .get(params.session_id) as { count: number };

      // Get user-confirmed count (agent core uses user_confirmed integer)
      const userConfirmed = db
        .prepare(
          `SELECT COUNT(*) as count
           FROM audit_log
           WHERE session_id = ? AND user_confirmed = 1`,
        )
        .get(params.session_id) as { count: number };

      // Total duration
      const totalDuration = db
        .prepare(
          `SELECT COALESCE(SUM(execution_time_ms), 0) as total_ms
           FROM audit_log
           WHERE session_id = ?`,
        )
        .get(params.session_id) as { total_ms: number };

      return {
        success: true,
        data: {
          tools_called: toolSummaries,
          succeeded: succeeded.count,
          failed: failed.count,
          user_confirmed: userConfirmed.count,
          total_execution_ms: totalDuration.total_ms,
        },
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to get session summary: ${msg}`);
    }
  },
};
