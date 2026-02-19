/**
 * system-settings.get_power_settings — Get power management settings.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { PowerSettings } from '../bridge';

const paramsSchema = z.object({}).describe('Get power management settings');
type Params = z.infer<typeof paramsSchema>;

export const getPowerSettings: MCPTool<Params> = {
  name: 'system-settings.get_power_settings',
  description: 'Get display sleep, system sleep, disk sleep timers, and wake-on-network status',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(_params: Params): Promise<MCPResult<PowerSettings>> {
    try {
      const data = await getBridge().getPowerSettings();
      return { success: true, data };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to get power settings: ${msg}`);
    }
  },
};
