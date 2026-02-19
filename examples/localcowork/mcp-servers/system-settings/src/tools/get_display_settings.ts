/**
 * system-settings.get_display_settings — Get display/monitor settings.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { DisplaySettings } from '../bridge';

const paramsSchema = z.object({}).describe('Get display settings');
type Params = z.infer<typeof paramsSchema>;

export const getDisplaySettings: MCPTool<Params> = {
  name: 'system-settings.get_display_settings',
  description: 'Get display sleep timer, brightness, and resolution',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(_params: Params): Promise<MCPResult<DisplaySettings>> {
    try {
      const data = await getBridge().getDisplaySettings();
      return { success: true, data };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to get display settings: ${msg}`);
    }
  },
};
