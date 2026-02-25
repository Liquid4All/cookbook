/**
 * filesystem.copy_file — Copy a file to a new location.
 *
 * Mutable: requires user confirmation.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath } from '../../../_shared/ts/validation';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  source: z.string().describe('Absolute path of file to copy'),
  destination: z.string().describe('Absolute path of destination'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

export const copyFile: MCPTool<Params> = {
  name: 'filesystem.copy_file',
  description: 'Copy a file to a new location',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    assertAbsolutePath(params.source, 'source');
    assertAbsolutePath(params.destination, 'destination');
    assertSandboxed(params.source);
    assertSandboxed(params.destination);

    // Verify source exists and is a file
    try {
      const stat = await fs.stat(params.source);
      if (!stat.isFile()) {
        throw new MCPError(ErrorCodes.INVALID_PARAMS, `Not a file: ${params.source}`);
      }
    } catch (err) {
      if (err instanceof MCPError) throw err;
      throw new MCPError(ErrorCodes.FILE_NOT_FOUND, `Source not found: ${params.source}`);
    }

    try {
      // Ensure destination directory exists
      const destDir = path.dirname(params.destination);
      await fs.mkdir(destDir, { recursive: true });

      await fs.copyFile(params.source, params.destination);

      return {
        success: true,
        data: { success: true, path: params.destination },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to copy file: ${msg}`);
    }
  },
};
