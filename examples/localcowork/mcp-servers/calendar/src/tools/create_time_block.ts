/**
 * calendar.create_time_block -- Create a focused time block on the calendar.
 *
 * Finds available slots for the given date, picks one matching the preferred
 * time of day, and creates an event with the specified duration.
 *
 * Mutable: requires confirmation before creating.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb, type TimeSlot } from '../db';
import { findFreeSlots } from './find_free_slots';

// ─── Time-of-day ranges (hours) ─────────────────────────────────────────────

interface TimeRange {
  readonly startHour: number;
  readonly endHour: number;
}

const TIME_PREFERENCES: Record<string, TimeRange> = {
  morning: { startHour: 8, endHour: 12 },
  afternoon: { startHour: 12, endHour: 17 },
  evening: { startHour: 17, endHour: 20 },
};

// ─── Schema ─────────────────────────────────────────────────────────────────

const paramsSchema = z.object({
  title: z.string().min(1).describe('Time block title'),
  date: z.string().min(1).describe('Date for the time block (ISO 8601 date)'),
  duration_minutes: z
    .number()
    .int()
    .min(1)
    .describe('Duration in minutes'),
  preferred_time: z
    .enum(['morning', 'afternoon', 'evening'])
    .optional()
    .describe('Preferred time of day: morning (8-12), afternoon (12-17), evening (17-20)'),
});

type Params = z.infer<typeof paramsSchema>;

/** Check whether a slot can fit a block of the given duration within a time range. */
function fitBlockInSlot(
  slot: TimeSlot,
  durationMinutes: number,
  range: TimeRange | null,
): { start: Date; end: Date } | null {
  const slotStart = new Date(slot.start);
  const slotEnd = new Date(slot.end);

  if (range) {
    // Compute the intersection of the slot with the preferred time range
    const dateStr = slot.start.split('T')[0];
    const rangeStart = new Date(`${dateStr}T${String(range.startHour).padStart(2, '0')}:00:00`);
    const rangeEnd = new Date(`${dateStr}T${String(range.endHour).padStart(2, '0')}:00:00`);

    const effectiveStart = slotStart > rangeStart ? slotStart : rangeStart;
    const effectiveEnd = slotEnd < rangeEnd ? slotEnd : rangeEnd;

    const availableMinutes = (effectiveEnd.getTime() - effectiveStart.getTime()) / 60000;
    if (availableMinutes >= durationMinutes) {
      const blockEnd = new Date(effectiveStart.getTime() + durationMinutes * 60000);
      return { start: effectiveStart, end: blockEnd };
    }
    return null;
  }

  // No preference -- use the start of the slot
  if (slot.duration_minutes >= durationMinutes) {
    const blockEnd = new Date(slotStart.getTime() + durationMinutes * 60000);
    return { start: slotStart, end: blockEnd };
  }
  return null;
}

/** Format a Date as local ISO 8601 datetime string (no timezone). */
function formatDateTime(d: Date): string {
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  const hours = String(d.getHours()).padStart(2, '0');
  const minutes = String(d.getMinutes()).padStart(2, '0');
  const seconds = String(d.getSeconds()).padStart(2, '0');
  return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}`;
}

export const createTimeBlock: MCPTool<Params> = {
  name: 'calendar.create_time_block',
  description: 'Create a focused time block on the calendar',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      // 1. Find free slots for the given date
      const freeSlotsResult = await findFreeSlots.execute({
        date: params.date,
        min_duration_minutes: params.duration_minutes,
      });

      const slots = (freeSlotsResult.data as { slots: TimeSlot[] }).slots;

      if (slots.length === 0) {
        throw new MCPError(
          ErrorCodes.INVALID_PARAMS,
          `No free slots of ${params.duration_minutes} minutes available on ${params.date}`,
        );
      }

      // 2. Pick a slot matching the preferred time
      const range = params.preferred_time
        ? TIME_PREFERENCES[params.preferred_time] ?? null
        : null;

      let bestFit: { start: Date; end: Date } | null = null;

      // First pass: try to find a slot within the preferred time range
      if (range) {
        for (const slot of slots) {
          const fit = fitBlockInSlot(slot, params.duration_minutes, range);
          if (fit) {
            bestFit = fit;
            break;
          }
        }
      }

      // Fallback: use the first slot that fits (no time preference)
      if (!bestFit) {
        for (const slot of slots) {
          const fit = fitBlockInSlot(slot, params.duration_minutes, null);
          if (fit) {
            bestFit = fit;
            break;
          }
        }
      }

      if (!bestFit) {
        throw new MCPError(
          ErrorCodes.INVALID_PARAMS,
          `No suitable slot found for ${params.duration_minutes} minutes ` +
            `${params.preferred_time ? `in the ${params.preferred_time}` : ''} on ${params.date}`,
        );
      }

      // 3. Create the event
      const db = getDb();
      const result = db
        .prepare(
          `INSERT INTO events (title, description, calendar_name, start_time, end_time)
           VALUES (?, ?, ?, ?, ?)`,
        )
        .run(
          params.title,
          'Time block created by calendar.create_time_block',
          'default',
          formatDateTime(bestFit.start),
          formatDateTime(bestFit.end),
        );

      return {
        success: true,
        data: {
          event_id: String(result.lastInsertRowid),
          scheduled_at: formatDateTime(bestFit.start),
        },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to create time block: ${msg}`);
    }
  },
};
