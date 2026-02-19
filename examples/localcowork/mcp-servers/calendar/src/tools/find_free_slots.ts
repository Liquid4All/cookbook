/**
 * calendar.find_free_slots -- Find available time blocks in a day.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Computes gaps between events during working hours (08:00-18:00).
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb, type CalendarEvent, type TimeSlot } from '../db';

const WORK_START_HOUR = 8;
const WORK_END_HOUR = 18;

const paramsSchema = z.object({
  date: z.string().min(1).describe('Date to check (ISO 8601 date, e.g. 2026-03-15)'),
  min_duration_minutes: z
    .number()
    .int()
    .min(1)
    .optional()
    .default(30)
    .describe('Minimum slot duration in minutes (default: 30)'),
});

type Params = z.infer<typeof paramsSchema>;

/** Parse an ISO datetime string into a Date object. */
function parseDateTime(dt: string): Date {
  return new Date(dt);
}

/** Format a Date object as local ISO 8601 datetime string (no timezone). */
function formatDateTime(d: Date): string {
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  const hours = String(d.getHours()).padStart(2, '0');
  const minutes = String(d.getMinutes()).padStart(2, '0');
  const seconds = String(d.getSeconds()).padStart(2, '0');
  return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}`;
}

/** Compute duration in minutes between two Date objects. */
function durationMinutes(start: Date, end: Date): number {
  return Math.round((end.getTime() - start.getTime()) / 60000);
}

export const findFreeSlots: MCPTool<Params> = {
  name: 'calendar.find_free_slots',
  description: 'Find available time blocks in a day',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();
      const dateStr = params.date.split('T')[0];
      const minDuration = params.min_duration_minutes ?? 30;

      // Define working hours boundaries for the given date
      const workStart = new Date(`${dateStr}T${String(WORK_START_HOUR).padStart(2, '0')}:00:00`);
      const workEnd = new Date(`${dateStr}T${String(WORK_END_HOUR).padStart(2, '0')}:00:00`);

      // Fetch all events that overlap with the working hours on this date
      const dayStart = `${dateStr}T00:00:00`;
      const dayEnd = `${dateStr}T23:59:59`;

      const rows = db
        .prepare(
          `SELECT * FROM events
           WHERE start_time < ? AND end_time > ?
           ORDER BY start_time ASC`,
        )
        .all(dayEnd, dayStart) as CalendarEvent[];

      // Clamp events to working hours and collect busy intervals
      const busyIntervals: Array<{ start: Date; end: Date }> = [];
      for (const event of rows) {
        const evStart = parseDateTime(event.start_time);
        const evEnd = parseDateTime(event.end_time);

        // Clamp to working hours
        const clampedStart = evStart < workStart ? workStart : evStart;
        const clampedEnd = evEnd > workEnd ? workEnd : evEnd;

        if (clampedStart < clampedEnd) {
          busyIntervals.push({ start: clampedStart, end: clampedEnd });
        }
      }

      // Sort by start time (already sorted from query, but be safe after clamping)
      busyIntervals.sort((a, b) => a.start.getTime() - b.start.getTime());

      // Merge overlapping intervals
      const merged: Array<{ start: Date; end: Date }> = [];
      for (const interval of busyIntervals) {
        if (merged.length === 0 || interval.start > merged[merged.length - 1].end) {
          merged.push({ start: interval.start, end: interval.end });
        } else {
          const last = merged[merged.length - 1];
          if (interval.end > last.end) {
            last.end = interval.end;
          }
        }
      }

      // Compute free slots as gaps between merged busy intervals
      const slots: TimeSlot[] = [];
      let cursor = workStart;

      for (const busy of merged) {
        if (cursor < busy.start) {
          const gap = durationMinutes(cursor, busy.start);
          if (gap >= minDuration) {
            slots.push({
              start: formatDateTime(cursor),
              end: formatDateTime(busy.start),
              duration_minutes: gap,
            });
          }
        }
        if (busy.end > cursor) {
          cursor = busy.end;
        }
      }

      // Check for a free slot after the last busy interval
      if (cursor < workEnd) {
        const gap = durationMinutes(cursor, workEnd);
        if (gap >= minDuration) {
          slots.push({
            start: formatDateTime(cursor),
            end: formatDateTime(workEnd),
            duration_minutes: gap,
          });
        }
      }

      return { success: true, data: { slots } };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to find free slots: ${msg}`);
    }
  },
};
