/**
 * task.create_task — Create a new task.
 *
 * Mutable: requires confirmation before creating.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb } from '../db';

const paramsSchema = z.object({
  title: z.string().min(1).describe('Task title'),
  description: z.string().optional().describe('Task details'),
  source: z.string().optional().describe('Origin: meeting, email, manual, clipboard'),
  priority: z.number().int().min(1).max(5).optional().default(3).describe('1=urgent, 5=someday'),
  due_date: z.string().optional().describe('Due date (ISO 8601)'),
});

type Params = z.infer<typeof paramsSchema>;

export const createTask: MCPTool<Params> = {
  name: 'task.create_task',
  description: 'Create a new task',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();

      const result = db
        .prepare(
          `INSERT INTO tasks (title, description, source, priority, due_date)
           VALUES (?, ?, ?, ?, ?)`,
        )
        .run(
          params.title,
          params.description ?? null,
          params.source ?? null,
          params.priority ?? 3,
          params.due_date ?? null,
        );

      return {
        success: true,
        data: { task_id: result.lastInsertRowid },
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to create task: ${msg}`);
    }
  },
};
