/**
 * system.get_system_info — Get system hardware and OS information.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Returns OS, architecture, CPU model, RAM, and optional GPU/NPU info.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { SystemInfo } from '../bridge';

const paramsSchema = z.object({}).describe('Get system hardware and OS information');

type Params = z.infer<typeof paramsSchema>;

export const getSystemInfo: MCPTool<Params> = {
  name: 'system.get_system_info',
  description: 'Get system hardware and OS information',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(_params: Params): Promise<MCPResult<SystemInfo>> {
    try {
      const bridge = getBridge();
      const info = await bridge.getSystemInfo();

      return {
        success: true,
        data: info,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to get system info: ${msg}`);
    }
  },
};
