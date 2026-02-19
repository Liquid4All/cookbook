/**
 * system.get_disk_usage — Get disk volume usage.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Returns per-volume capacity, used, and free space in GB.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { DiskVolume } from '../bridge';

const paramsSchema = z.object({}).describe('Get disk volume usage');

type Params = z.infer<typeof paramsSchema>;

/** Result shape for get_disk_usage */
interface DiskUsageResult {
  readonly volumes: DiskVolume[];
}

export const getDiskUsage: MCPTool<Params> = {
  name: 'system.get_disk_usage',
  description: 'Get disk volume capacity, used space, and free space for all mounted volumes',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(_params: Params): Promise<MCPResult<DiskUsageResult>> {
    try {
      const bridge = getBridge();
      const volumes = await bridge.getDiskUsage();

      return {
        success: true,
        data: { volumes },
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to get disk usage: ${msg}`);
    }
  },
};
