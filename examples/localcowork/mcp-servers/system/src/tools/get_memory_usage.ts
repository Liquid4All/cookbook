/**
 * system.get_memory_usage — Get system memory (RAM) usage.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Returns total, used, free RAM and swap usage in GB.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { MemoryUsage } from '../bridge';

const paramsSchema = z.object({}).describe('Get system memory usage');

type Params = z.infer<typeof paramsSchema>;

export const getMemoryUsage: MCPTool<Params> = {
  name: 'system.get_memory_usage',
  description: 'Get system memory (RAM) and swap usage in GB with percentage',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(_params: Params): Promise<MCPResult<MemoryUsage>> {
    try {
      const bridge = getBridge();
      const usage = await bridge.getMemoryUsage();

      return {
        success: true,
        data: usage,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to get memory usage: ${msg}`);
    }
  },
};
