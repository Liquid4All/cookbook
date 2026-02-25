/**
 * clipboard.get_clipboard -- Get current clipboard contents.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';

// ── Params ───────────────────────────────────────────────────────────────────

const paramsSchema = z.object({}).describe('Get current clipboard contents (no params)');

type Params = z.infer<typeof paramsSchema>;

// ── Return type ──────────────────────────────────────────────────────────────

interface GetClipboardResult {
  readonly content: string;
  readonly type: string;
}

// ── Tool ─────────────────────────────────────────────────────────────────────

export const getClipboard: MCPTool<Params> = {
  name: 'clipboard.get_clipboard',
  description: 'Get current clipboard contents',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(_params: Params): Promise<MCPResult<GetClipboardResult>> {
    try {
      const bridge = getBridge();
      const result = await bridge.read();
      return {
        success: true,
        data: { content: result.content, type: result.type },
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to read clipboard: ${msg}`);
    }
  },
};
