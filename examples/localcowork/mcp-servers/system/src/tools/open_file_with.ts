/**
 * system.open_file_with — Open a file with a specific application.
 *
 * Mutable: requires confirmation before executing.
 * Opens a file path with the system default or a specified application.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { OpenFileResult } from '../bridge';

const paramsSchema = z
  .object({
    path: z.string().min(1).describe('File path'),
    app: z.string().optional().describe('Application to open with (default: system default)'),
  })
  .describe('Open a file with a specific application');

type Params = z.infer<typeof paramsSchema>;

export const openFileWith: MCPTool<Params> = {
  name: 'system.open_file_with',
  description: 'Open a file with a specific application',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult<OpenFileResult>> {
    try {
      const bridge = getBridge();
      const result = await bridge.openFileWith(params.path, params.app);

      return {
        success: true,
        data: result,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to open file: ${msg}`);
    }
  },
};
