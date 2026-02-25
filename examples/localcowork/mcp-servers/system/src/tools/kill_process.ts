/**
 * system.kill_process — Terminate a running process.
 *
 * DESTRUCTIVE: requires user confirmation before execution.
 * Sends a signal (default SIGTERM) to a process by PID.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { KillProcessResult } from '../bridge';

const paramsSchema = z
  .object({
    pid: z.number().int().positive().describe('Process ID to terminate'),
    signal: z
      .string()
      .optional()
      .describe('Signal to send (default: SIGTERM). Options: SIGTERM, SIGKILL, SIGINT'),
  })
  .describe('Terminate a running process by PID');

type Params = z.infer<typeof paramsSchema>;

export const killProcess: MCPTool<Params> = {
  name: 'system.kill_process',
  description: 'Terminate a running process by PID (destructive, requires confirmation)',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult<KillProcessResult>> {
    const validSignals = ['SIGTERM', 'SIGKILL', 'SIGINT', 'SIGHUP'];
    const signal = params.signal ?? 'SIGTERM';

    if (!validSignals.includes(signal)) {
      throw new MCPError(
        ErrorCodes.INVALID_PARAMS,
        `Invalid signal: ${signal}. Valid: ${validSignals.join(', ')}`,
      );
    }

    try {
      const bridge = getBridge();
      const result = await bridge.killProcess(params.pid, signal);

      return {
        success: true,
        data: result,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to kill process: ${msg}`);
    }
  },
};
