/**
 * calendar.create_event -- Create a new calendar event.
 *
 * Mutable: requires confirmation before creating.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb } from '../db';

const paramsSchema = z.object({
  title: z.string().min(1).describe('Event title'),
  start: z.string().min(1).describe('Start time (ISO 8601 datetime)'),
  end: z.string().min(1).describe('End time (ISO 8601 datetime)'),
  description: z.string().optional().describe('Event description'),
  calendar: z.string().optional().describe('Calendar name (default: "default")'),
});

type Params = z.infer<typeof paramsSchema>;

export const createEvent: MCPTool<Params> = {
  name: 'calendar.create_event',
  description: 'Create a new calendar event',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();

      // Validate that end is after start
      if (params.end <= params.start) {
        throw new MCPError(
          ErrorCodes.INVALID_PARAMS,
          'End time must be after start time',
        );
      }

      const result = db
        .prepare(
          `INSERT INTO events (title, description, calendar_name, start_time, end_time)
           VALUES (?, ?, ?, ?, ?)`,
        )
        .run(
          params.title,
          params.description ?? null,
          params.calendar ?? 'default',
          params.start,
          params.end,
        );

      return {
        success: true,
        data: { event_id: String(result.lastInsertRowid) },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to create event: ${msg}`);
    }
  },
};
