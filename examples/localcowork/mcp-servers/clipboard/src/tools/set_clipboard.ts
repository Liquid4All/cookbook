/**
 * clipboard.set_clipboard -- Set clipboard contents.
 *
 * Non-destructive per PRD spec: no confirmation required.
 * Adds each write to the clipboard history.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge, addToHistory } from '../bridge';

// ── Params ───────────────────────────────────────────────────────────────────

const paramsSchema = z
  .object({
    content: z.string().min(1).describe('Content to copy to clipboard'),
  })
  .describe('Set clipboard contents');

type Params = z.infer<typeof paramsSchema>;

// ── Return type ──────────────────────────────────────────────────────────────

interface SetClipboardResult {
  readonly success: boolean;
}

// ── Tool ─────────────────────────────────────────────────────────────────────

export const setClipboard: MCPTool<Params> = {
  name: 'clipboard.set_clipboard',
  description: 'Set clipboard contents',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult<SetClipboardResult>> {
    try {
      const bridge = getBridge();
      const ok = await bridge.write(params.content);
      if (ok) {
        addToHistory(params.content, 'text/plain');
      }
      return {
        success: ok,
        data: { success: ok },
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to set clipboard: ${msg}`);
    }
  },
};
