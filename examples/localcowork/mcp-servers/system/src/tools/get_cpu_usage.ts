/**
 * system.get_cpu_usage — Get CPU utilization.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Returns core count, model, load average, and per-core utilization.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { CpuUsage } from '../bridge';

const paramsSchema = z.object({}).describe('Get CPU utilization');

type Params = z.infer<typeof paramsSchema>;

export const getCpuUsage: MCPTool<Params> = {
  name: 'system.get_cpu_usage',
  description: 'Get CPU core count, model, load average, and per-core utilization percentage',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(_params: Params): Promise<MCPResult<CpuUsage>> {
    try {
      const bridge = getBridge();
      const usage = await bridge.getCpuUsage();

      return {
        success: true,
        data: usage,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to get CPU usage: ${msg}`);
    }
  },
};
