/**
 * task.list_tasks — List tasks with optional filters.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb, type Task } from '../db';

const paramsSchema = z.object({
  status: z
    .enum(['pending', 'completed', 'overdue'])
    .optional()
    .describe('Filter: pending, completed, overdue'),
  priority: z.number().int().min(1).max(5).optional().describe('Filter by priority'),
  limit: z.number().int().min(1).max(500).optional().default(50).describe('Max results'),
});

type Params = z.infer<typeof paramsSchema>;

export const listTasks: MCPTool<Params> = {
  name: 'task.list_tasks',
  description: 'List tasks with optional filters',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();
      const conditions: string[] = [];
      const values: unknown[] = [];

      if (params.status === 'pending') {
        conditions.push('completed_at IS NULL');
      } else if (params.status === 'completed') {
        conditions.push('completed_at IS NOT NULL');
      } else if (params.status === 'overdue') {
        conditions.push("completed_at IS NULL AND due_date < datetime('now')");
      }

      if (params.priority !== undefined) {
        conditions.push('priority = ?');
        values.push(params.priority);
      }

      const where = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';
      const limit = params.limit ?? 50;
      values.push(limit);

      const sql = `SELECT * FROM tasks ${where} ORDER BY priority ASC, due_date ASC NULLS LAST LIMIT ?`;
      const rows = db.prepare(sql).all(...values) as Task[];

      return { success: true, data: rows };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to list tasks: ${msg}`);
    }
  },
};
