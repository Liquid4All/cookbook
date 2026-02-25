/**
 * email.list_drafts — List saved email drafts.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Returns drafts with status='draft', ordered by created_at DESC.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb, type EmailDraft } from '../db';

// ─── Params ─────────────────────────────────────────────────────────────────

const paramsSchema = z.object({
  limit: z
    .number()
    .int()
    .min(1)
    .max(500)
    .optional()
    .default(20)
    .describe('Max number of drafts to return'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

export const listDrafts: MCPTool<Params> = {
  name: 'email.list_drafts',
  description: 'List saved email drafts',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();
      const limit = params.limit ?? 20;

      const rows = db
        .prepare(
          `SELECT * FROM drafts
           WHERE status = 'draft'
           ORDER BY created_at DESC
           LIMIT ?`,
        )
        .all(limit) as EmailDraft[];

      // Parse JSON address fields for the response
      const drafts = rows.map((row) => ({
        id: row.id,
        to: JSON.parse(row.to_addresses) as string[],
        cc: row.cc_addresses ? (JSON.parse(row.cc_addresses) as string[]) : [],
        subject: row.subject,
        body: row.body,
        created_at: row.created_at,
      }));

      return { success: true, data: drafts };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to list drafts: ${msg}`);
    }
  },
};
