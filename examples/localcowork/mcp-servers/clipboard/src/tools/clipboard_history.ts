/**
 * clipboard.clipboard_history -- Get recent clipboard entries.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Returns entries in reverse-chronological order (most recent first).
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getHistory, type ClipboardEntry } from '../bridge';

// ── Params ───────────────────────────────────────────────────────────────────

const paramsSchema = z
  .object({
    limit: z
      .number()
      .int()
      .min(1)
      .max(100)
      .optional()
      .default(20)
      .describe('Max entries to return'),
  })
  .describe('Get recent clipboard entries');

type Params = z.infer<typeof paramsSchema>;

// ── Return type ──────────────────────────────────────────────────────────────

interface ClipboardHistoryResult {
  readonly entries: ClipboardEntry[];
}

// ── Tool ─────────────────────────────────────────────────────────────────────

export const clipboardHistory: MCPTool<Params> = {
  name: 'clipboard.clipboard_history',
  description: 'Get recent clipboard entries',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult<ClipboardHistoryResult>> {
    try {
      const limit = params.limit ?? 20;
      const allEntries = getHistory();
      const entries = allEntries.slice(0, limit);
      return {
        success: true,
        data: { entries },
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to get clipboard history: ${msg}`);
    }
  },
};
