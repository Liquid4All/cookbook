/**
 * email.search_emails — Search local email archive.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Searches emails using LIKE queries on subject and body,
 * with optional folder filtering.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb, type EmailRecord } from '../db';

// ─── Params ─────────────────────────────────────────────────────────────────

const paramsSchema = z.object({
  query: z.string().min(1).describe('Search query (matched against subject and body)'),
  folder: z.string().optional().describe('Filter by folder (e.g., inbox, sent, archive)'),
  limit: z
    .number()
    .int()
    .min(1)
    .max(500)
    .optional()
    .default(20)
    .describe('Max results to return'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

export const searchEmails: MCPTool<Params> = {
  name: 'email.search_emails',
  description: 'Search local email archive',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();
      const conditions: string[] = [];
      const values: (string | number)[] = [];

      // LIKE search on subject and body
      const likePattern = `%${params.query}%`;
      conditions.push('(subject LIKE ? OR body LIKE ?)');
      values.push(likePattern, likePattern);

      // Optional folder filter
      if (params.folder !== undefined) {
        conditions.push('folder = ?');
        values.push(params.folder);
      }

      const limit = params.limit ?? 20;
      values.push(limit);

      const where = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';
      const sql = `SELECT * FROM emails ${where} ORDER BY received_at DESC LIMIT ?`;
      const rows = db.prepare(sql).all(...values) as EmailRecord[];

      // Map to summary objects
      const results = rows.map((row) => ({
        id: row.id,
        thread_id: row.thread_id,
        folder: row.folder,
        from: row.from_address,
        to: JSON.parse(row.to_addresses) as string[],
        subject: row.subject,
        received_at: row.received_at,
        is_read: row.is_read === 1,
      }));

      return { success: true, data: results };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to search emails: ${msg}`);
    }
  },
};
