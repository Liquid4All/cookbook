/**
 * system-settings.set_default_browser — Set the default web browser.
 *
 * MUTABLE: requires user confirmation. Undo supported (stores previous value).
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { SetResult } from '../bridge';

const KNOWN_BROWSERS = [
  'Safari', 'Google Chrome', 'Firefox', 'Microsoft Edge',
  'Brave Browser', 'Arc', 'Opera', 'Vivaldi', 'Chromium',
];

const paramsSchema = z
  .object({
    browser: z
      .string()
      .min(1)
      .describe(`Browser name (e.g., ${KNOWN_BROWSERS.slice(0, 4).join(', ')})`),
  })
  .describe('Set default web browser');

type Params = z.infer<typeof paramsSchema>;

export const setDefaultBrowser: MCPTool<Params> = {
  name: 'system-settings.set_default_browser',
  description: 'Set the default web browser for opening HTTP/HTTPS links',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: true,

  async execute(params: Params): Promise<MCPResult<SetResult>> {
    // Validate browser name (case-insensitive match against known list)
    const normalized = params.browser.trim();
    const isKnown = KNOWN_BROWSERS.some(
      (b) => b.toLowerCase() === normalized.toLowerCase(),
    );
    if (!isKnown) {
      throw new MCPError(
        ErrorCodes.INVALID_PARAMS,
        `Unknown browser: "${normalized}". Known browsers: ${KNOWN_BROWSERS.join(', ')}`,
      );
    }

    try {
      const data = await getBridge().setDefaultBrowser(normalized);
      return { success: true, data };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to set default browser: ${msg}`);
    }
  },
};
