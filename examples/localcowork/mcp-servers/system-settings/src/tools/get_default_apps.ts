/**
 * system-settings.get_default_apps — Get default application assignments.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { DefaultApps } from '../bridge';

const paramsSchema = z.object({}).describe('Get default application assignments');
type Params = z.infer<typeof paramsSchema>;

export const getDefaultApps: MCPTool<Params> = {
  name: 'system-settings.get_default_apps',
  description: 'Get default browser, email client, and PDF viewer assignments',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(_params: Params): Promise<MCPResult<DefaultApps>> {
    try {
      const data = await getBridge().getDefaultApps();
      return { success: true, data };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to get default apps: ${msg}`);
    }
  },
};
