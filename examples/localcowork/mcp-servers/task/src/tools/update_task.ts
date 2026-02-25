/**
 * task.update_task — Update an existing task.
 *
 * Mutable: requires confirmation before updating.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb } from '../db';

const paramsSchema = z.object({
  task_id: z.number().int().describe('Task ID'),
  updates: z
    .object({
      title: z.string().min(1).optional(),
      description: z.string().optional(),
      priority: z.number().int().min(1).max(5).optional(),
      due_date: z.string().nullable().optional(),
      completed_at: z.string().nullable().optional(),
    })
    .describe('Fields to update'),
});

type Params = z.infer<typeof paramsSchema>;

export const updateTask: MCPTool<Params> = {
  name: 'task.update_task',
  description: 'Update an existing task',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();

      // Verify the task exists
      const existing = db.prepare('SELECT id FROM tasks WHERE id = ?').get(params.task_id);
      if (!existing) {
        throw new MCPError(ErrorCodes.INVALID_PARAMS, `Task not found: ${params.task_id}`);
      }

      const setClauses: string[] = [];
      const values: unknown[] = [];

      const { updates } = params;

      if (updates.title !== undefined) {
        setClauses.push('title = ?');
        values.push(updates.title);
      }
      if (updates.description !== undefined) {
        setClauses.push('description = ?');
        values.push(updates.description);
      }
      if (updates.priority !== undefined) {
        setClauses.push('priority = ?');
        values.push(updates.priority);
      }
      if (updates.due_date !== undefined) {
        setClauses.push('due_date = ?');
        values.push(updates.due_date);
      }
      if (updates.completed_at !== undefined) {
        setClauses.push('completed_at = ?');
        values.push(updates.completed_at);
      }

      if (setClauses.length === 0) {
        return { success: true, data: { message: 'No updates provided' } };
      }

      values.push(params.task_id);
      const sql = `UPDATE tasks SET ${setClauses.join(', ')} WHERE id = ?`;
      db.prepare(sql).run(...values);

      return { success: true, data: { task_id: params.task_id } };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to update task: ${msg}`);
    }
  },
};
