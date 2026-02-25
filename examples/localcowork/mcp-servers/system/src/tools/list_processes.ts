/**
 * system.list_processes — List running processes.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Returns a list of running processes with optional name filtering.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { ProcessInfo } from '../bridge';

const paramsSchema = z
  .object({
    filter: z.string().optional().describe('Process name filter'),
  })
  .describe('List running processes');

type Params = z.infer<typeof paramsSchema>;

/** Result shape for list_processes */
interface ListProcessesResult {
  readonly processes: ProcessInfo[];
}

export const listProcesses: MCPTool<Params> = {
  name: 'system.list_processes',
  description: 'List running processes',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult<ListProcessesResult>> {
    try {
      const bridge = getBridge();
      const processes = await bridge.listProcesses(params.filter);

      return {
        success: true,
        data: { processes },
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to list processes: ${msg}`);
    }
  },
};
