/**
 * email.draft_email — Create an email draft.
 *
 * Mutable: requires confirmation before creating.
 * Stores the draft in the local SQLite database and returns
 * a draft_id with a preview of the body.
 */

import { z } from 'zod';
import crypto from 'crypto';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getDb } from '../db';

// ─── Params ─────────────────────────────────────────────────────────────────

const paramsSchema = z.object({
  to: z.array(z.string().min(1)).min(1).describe('Recipient email addresses'),
  subject: z.string().min(1).describe('Email subject line'),
  body: z.string().min(1).describe('Email body in markdown'),
  cc: z.array(z.string().min(1)).optional().describe('CC email addresses'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

const PREVIEW_MAX_LENGTH = 200;

export const draftEmail: MCPTool<Params> = {
  name: 'email.draft_email',
  description: 'Create an email draft',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    try {
      const db = getDb();
      const draftId = crypto.randomUUID();

      const toJson = JSON.stringify(params.to);
      const ccJson = params.cc ? JSON.stringify(params.cc) : null;
      const preview =
        params.body.length > PREVIEW_MAX_LENGTH
          ? params.body.slice(0, PREVIEW_MAX_LENGTH) + '...'
          : params.body;

      db.prepare(
        `INSERT INTO drafts (id, to_addresses, cc_addresses, subject, body, status)
         VALUES (?, ?, ?, ?, ?, 'draft')`,
      ).run(draftId, toJson, ccJson, params.subject, params.body);

      return {
        success: true,
        data: { draft_id: draftId, preview },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to create draft: ${msg}`);
    }
  },
};
