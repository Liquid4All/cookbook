/**
 * system.get_network_info — Get network interface information.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Returns active network interfaces with IP addresses and MAC addresses.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { NetworkInterface } from '../bridge';

const paramsSchema = z.object({}).describe('Get network interface information');

type Params = z.infer<typeof paramsSchema>;

/** Result shape for get_network_info */
interface NetworkInfoResult {
  readonly interfaces: NetworkInterface[];
}

export const getNetworkInfo: MCPTool<Params> = {
  name: 'system.get_network_info',
  description: 'Get active network interfaces with IP addresses, MAC addresses, and status',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(_params: Params): Promise<MCPResult<NetworkInfoResult>> {
    try {
      const bridge = getBridge();
      const interfaces = await bridge.getNetworkInfo();

      return {
        success: true,
        data: { interfaces },
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to get network info: ${msg}`);
    }
  },
};
