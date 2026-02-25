/**
 * calendar.list_events -- List calendar events in a date range.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb, type CalendarEvent } from '../db';

const paramsSchema = z.object({
  start_date: z.string().min(1).describe('Start date (ISO 8601 date, e.g. 2026-03-01)'),
  end_date: z.string().min(1).describe('End date (ISO 8601 date, e.g. 2026-03-31)'),
  calendar: z.string().optional().describe('Calendar name filter (default: all calendars)'),
});

type Params = z.infer<typeof paramsSchema>;

export const listEvents: MCPTool<Params> = {
  name: 'calendar.list_events',
  description: 'List calendar events in a date range',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();

      // Normalize dates to ensure we cover full days
      const startBound = params.start_date.includes('T')
        ? params.start_date
        : `${params.start_date}T00:00:00`;
      const endBound = params.end_date.includes('T')
        ? params.end_date
        : `${params.end_date}T23:59:59`;

      const conditions: string[] = ['start_time <= ? AND end_time >= ?'];
      const values: (string | number)[] = [endBound, startBound];

      if (params.calendar !== undefined) {
        conditions.push('calendar_name = ?');
        values.push(params.calendar);
      }

      const where = `WHERE ${conditions.join(' AND ')}`;
      const sql = `SELECT * FROM events ${where} ORDER BY start_time ASC`;
      const rows = db.prepare(sql).all(...values) as CalendarEvent[];

      return { success: true, data: rows };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to list events: ${msg}`);
    }
  },
};
