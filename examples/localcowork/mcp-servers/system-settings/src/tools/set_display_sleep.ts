/**
 * system-settings.set_display_sleep — Set display sleep timer.
 *
 * MUTABLE: requires user confirmation. Undo supported (stores previous value).
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { SetResult } from '../bridge';

const paramsSchema = z
  .object({
    minutes: z
      .number()
      .int()
      .min(0)
      .max(480)
      .describe('Minutes before display sleeps (0 = never)'),
  })
  .describe('Set display sleep timer');

type Params = z.infer<typeof paramsSchema>;

export const setDisplaySleep: MCPTool<Params> = {
  name: 'system-settings.set_display_sleep',
  description: 'Set the display sleep timer in minutes (0 = never sleep)',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: true,

  async execute(params: Params): Promise<MCPResult<SetResult>> {
    try {
      const data = await getBridge().setDisplaySleep(params.minutes);
      return { success: true, data };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to set display sleep: ${msg}`);
    }
  },
};
