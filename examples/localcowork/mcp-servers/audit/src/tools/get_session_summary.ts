/**
 * audit.get_session_summary — Aggregate summary for a session.
 *
 * Non-destructive: executes immediately, no confirmation needed.
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

      // Get unique documents touched
      const docs = db
        .prepare(
          `SELECT DISTINCT file_path, file_hash
           FROM audit_log
           WHERE session_id = ? AND file_path IS NOT NULL`,
        )
        .all(params.session_id) as Array<{ file_path: string; file_hash: string | null }>;

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

      // Get confirmation/rejection counts
      const confirmations = db
        .prepare(
          `SELECT COUNT(*) as count
           FROM audit_log
           WHERE session_id = ? AND status = 'confirmed'`,
        )
        .get(params.session_id) as { count: number };

      const rejections = db
        .prepare(
          `SELECT COUNT(*) as count
           FROM audit_log
           WHERE session_id = ? AND status = 'rejected'`,
        )
        .get(params.session_id) as { count: number };

      return {
        success: true,
        data: {
          documents_touched: docs.map((d) => ({
            path: d.file_path,
            hash: d.file_hash,
          })),
          tools_called: toolSummaries,
          confirmations: confirmations.count,
          rejections: rejections.count,
        },
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to get session summary: ${msg}`);
    }
  },
};
