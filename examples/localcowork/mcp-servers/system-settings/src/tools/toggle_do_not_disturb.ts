/**
 * system-settings.toggle_do_not_disturb — Toggle Do Not Disturb / Focus mode.
 *
 * MUTABLE: requires user confirmation. Undo supported (stores previous state).
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { SetResult } from '../bridge';

const paramsSchema = z
  .object({
    enable: z
      .boolean()
      .describe('true to enable Do Not Disturb, false to disable'),
  })
  .describe('Toggle Do Not Disturb mode');

type Params = z.infer<typeof paramsSchema>;

export const toggleDoNotDisturb: MCPTool<Params> = {
  name: 'system-settings.toggle_do_not_disturb',
  description: 'Enable or disable Do Not Disturb / Focus mode',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: true,

  async execute(params: Params): Promise<MCPResult<SetResult>> {
    try {
      const data = await getBridge().toggleDoNotDisturb(params.enable);
      return { success: true, data };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to toggle Do Not Disturb: ${msg}`);
    }
  },
};
