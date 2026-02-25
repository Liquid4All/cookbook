/**
 * task.get_overdue — Get all overdue tasks.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb, type Task } from '../db';

const paramsSchema = z.object({});

type Params = z.infer<typeof paramsSchema>;

export const getOverdue: MCPTool<Params> = {
  name: 'task.get_overdue',
  description: 'Get all overdue tasks',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(_params: Params): Promise<MCPResult> {
    try {
      const db = getDb();

      const rows = db
        .prepare(
          `SELECT * FROM tasks
           WHERE completed_at IS NULL AND due_date < datetime('now')
           ORDER BY due_date ASC`,
        )
        .all() as Task[];

      return { success: true, data: rows };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to get overdue tasks: ${msg}`);
    }
  },
};
