/**
 * task.daily_briefing — Generate a daily briefing of tasks.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Summarizes pending and overdue tasks for a given date.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb, type Task } from '../db';

const paramsSchema = z.object({
  date: z.string().optional().describe('Date for briefing (ISO 8601, default: today)'),
});

type Params = z.infer<typeof paramsSchema>;

/** Format a date string for briefing display. */
function formatDate(iso: string | null): string {
  if (!iso) return 'no date';
  try {
    return new Date(iso).toLocaleDateString();
  } catch {
    return iso;
  }
}

export const dailyBriefing: MCPTool<Params> = {
  name: 'task.daily_briefing',
  description: 'Generate a daily briefing combining tasks and overdue items',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();
      const targetDate = params.date ?? new Date().toISOString().split('T')[0];

      // Pending tasks (not completed)
      const pending = db
        .prepare(
          `SELECT * FROM tasks
           WHERE completed_at IS NULL
           ORDER BY priority ASC, due_date ASC NULLS LAST
           LIMIT 20`,
        )
        .all() as Task[];

      // Overdue tasks
      const overdue = db
        .prepare(
          `SELECT * FROM tasks
           WHERE completed_at IS NULL AND due_date < ?
           ORDER BY due_date ASC`,
        )
        .all(targetDate) as Task[];

      // Tasks due today
      const dueToday = db
        .prepare(
          `SELECT * FROM tasks
           WHERE completed_at IS NULL AND date(due_date) = date(?)
           ORDER BY priority ASC`,
        )
        .all(targetDate) as Task[];

      // Build briefing text
      const lines: string[] = [`Daily Briefing — ${targetDate}`, ''];

      if (overdue.length > 0) {
        lines.push(`⚠️ ${overdue.length} OVERDUE:`);
        for (const t of overdue) {
          lines.push(`  • [P${t.priority}] ${t.title} (due: ${formatDate(t.due_date)})`);
        }
        lines.push('');
      }

      if (dueToday.length > 0) {
        lines.push(`📅 ${dueToday.length} DUE TODAY:`);
        for (const t of dueToday) {
          lines.push(`  • [P${t.priority}] ${t.title}`);
        }
        lines.push('');
      }

      if (pending.length > 0) {
        lines.push(`📋 ${pending.length} PENDING:`);
        for (const t of pending.slice(0, 10)) {
          const due = t.due_date ? ` (due: ${formatDate(t.due_date)})` : '';
          lines.push(`  • [P${t.priority}] ${t.title}${due}`);
        }
        if (pending.length > 10) {
          lines.push(`  ... and ${pending.length - 10} more`);
        }
      }

      return {
        success: true,
        data: {
          briefing: lines.join('\n'),
          tasks: pending,
          overdue,
          due_today: dueToday,
        },
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to generate briefing: ${msg}`);
    }
  },
};
