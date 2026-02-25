/**
 * system.open_application — Open an application by name.
 *
 * Mutable: requires confirmation before executing.
 * Opens a named application and returns the process ID on success.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { OpenAppResult } from '../bridge';

const paramsSchema = z
  .object({
    app_name: z.string().min(1).describe('Application name or path'),
  })
  .describe('Open an application by name');

type Params = z.infer<typeof paramsSchema>;

export const openApplication: MCPTool<Params> = {
  name: 'system.open_application',
  description: 'Open an application by name',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult<OpenAppResult>> {
    try {
      const bridge = getBridge();
      const result = await bridge.openApplication(params.app_name);

      return {
        success: true,
        data: result,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to open application: ${msg}`);
    }
  },
};
