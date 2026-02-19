/**
 * email.send_draft — Send a previously created draft.
 *
 * Mutable: requires confirmation before sending.
 * Marks the draft as 'sent' in the database and generates a message_id.
 * No real SMTP is used — this is a local-first implementation.
 */

import { z } from 'zod';
import crypto from 'crypto';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb, type EmailDraft } from '../db';

// ─── Params ─────────────────────────────────────────────────────────────────

const paramsSchema = z.object({
  draft_id: z.string().min(1).describe('ID of the draft to send'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

export const sendDraft: MCPTool<Params> = {
  name: 'email.send_draft',
  description: 'Send a previously created draft via SMTP',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();

      // Look up the draft
      const draft = db
        .prepare('SELECT * FROM drafts WHERE id = ?')
        .get(params.draft_id) as EmailDraft | undefined;

      if (!draft) {
        throw new MCPError(
          ErrorCodes.INVALID_PARAMS,
          `Draft not found: ${params.draft_id}`,
        );
      }

      if (draft.status !== 'draft') {
        throw new MCPError(
          ErrorCodes.INVALID_PARAMS,
          `Draft "${params.draft_id}" has already been sent`,
        );
      }

      // Mark as sent
      const messageId = `<${crypto.randomUUID()}@localcowork.local>`;
      const sentAt = new Date().toISOString();

      db.prepare(
        `UPDATE drafts
         SET status = 'sent', sent_at = ?, message_id = ?
         WHERE id = ?`,
      ).run(sentAt, messageId, params.draft_id);

      return {
        success: true,
        data: { success: true, message_id: messageId },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to send draft: ${msg}`);
    }
  },
};
