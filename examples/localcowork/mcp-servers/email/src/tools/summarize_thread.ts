/**
 * email.summarize_thread — Summarize an email thread.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Retrieves all emails in the thread, ordered by received_at,
 * and generates a structured summary without LLM assistance.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb, type EmailRecord } from '../db';

// ─── Params ─────────────────────────────────────────────────────────────────

const paramsSchema = z.object({
  thread_id: z.string().min(1).describe('Thread identifier'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

export const summarizeThread: MCPTool<Params> = {
  name: 'email.summarize_thread',
  description: 'Summarize an email thread',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();

      const rows = db
        .prepare(
          `SELECT * FROM emails
           WHERE thread_id = ?
           ORDER BY received_at ASC`,
        )
        .all(params.thread_id) as EmailRecord[];

      if (rows.length === 0) {
        throw new MCPError(
          ErrorCodes.INVALID_PARAMS,
          `No emails found for thread_id: ${params.thread_id}`,
        );
      }

      // Extract unique participants from from_address
      const participantSet = new Set<string>();
      for (const row of rows) {
        participantSet.add(row.from_address);
      }
      const participants = Array.from(participantSet);

      // Extract key points from each email's subject
      const keyPointSet = new Set<string>();
      for (const row of rows) {
        // Strip common prefixes like Re:, Fwd:
        const cleaned = row.subject.replace(/^(Re:|Fwd:|FW:)\s*/gi, '').trim();
        if (cleaned.length > 0) {
          keyPointSet.add(cleaned);
        }
      }
      const keyPoints = Array.from(keyPointSet);

      // Build a summary from the thread
      const messageCount = rows.length;
      const firstSubject = rows[0].subject;
      const firstDate = rows[0].received_at;
      const lastDate = rows[rows.length - 1].received_at;

      const summary =
        `Thread "${firstSubject}" — ${messageCount} message(s) ` +
        `from ${firstDate} to ${lastDate} ` +
        `involving ${participants.join(', ')}.`;

      return {
        success: true,
        data: { summary, participants, key_points: keyPoints },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to summarize thread: ${msg}`);
    }
  },
};
