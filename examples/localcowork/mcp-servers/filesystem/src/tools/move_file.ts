/**
 * filesystem.move_file — Move a file from source to destination.
 *
 * Mutable + undoable: requires confirmation, pushes to undo stack.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath } from '../../../_shared/ts/validation';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  source: z.string().describe('Absolute path of file to move'),
  destination: z.string().describe('Absolute path of destination'),
  create_dirs: z
    .boolean()
    .optional()
    .default(true)
    .describe('Create parent directories if missing'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

export const moveFile: MCPTool<Params> = {
  name: 'filesystem.move_file',
  description: 'Move a file from source to destination. Requires user confirmation.',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: true,

  async execute(params: Params): Promise<MCPResult> {
    assertAbsolutePath(params.source, 'source');
    assertAbsolutePath(params.destination, 'destination');
    assertSandboxed(params.source);
    assertSandboxed(params.destination);

    // Verify source exists
    try {
      await fs.access(params.source);
    } catch {
      throw new MCPError(ErrorCodes.FILE_NOT_FOUND, `Source not found: ${params.source}`);
    }

    // Ensure destination directory exists
    if (params.create_dirs) {
      const destDir = path.dirname(params.destination);
      await fs.mkdir(destDir, { recursive: true });
    }

    try {
      await fs.rename(params.source, params.destination);

      return {
        success: true,
        data: {
          success: true,
          original_path: params.source,
          new_path: params.destination,
        },
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to move file: ${msg}`);
    }
  },
};
