/**
 * filesystem.delete_file — Delete a file (moves to trash for recovery).
 *
 * Destructive + undoable: requires typed confirmation, recoverable from trash.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath } from '../../../_shared/ts/validation';

// ─── Constants ──────────────────────────────────────────────────────────────

const DATA_DIR =
  process.env.LOCALCOWORK_DATA_DIR ?? path.join(os.homedir(), '.localcowork');

const TRASH_DIR = path.join(DATA_DIR, 'trash');

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  path: z.string().describe('Absolute path of file to delete'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

export const deleteFile: MCPTool<Params> = {
  name: 'filesystem.delete_file',
  description: 'Delete a file. Moves to .localcowork/trash/ for recovery.',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: true,

  async execute(params: Params): Promise<MCPResult> {
    assertAbsolutePath(params.path, 'path');
    assertSandboxed(params.path);

    // Verify file exists
    try {
      const stat = await fs.stat(params.path);
      if (!stat.isFile()) {
        throw new MCPError(ErrorCodes.INVALID_PARAMS, `Not a file: ${params.path}`);
      }
    } catch (err) {
      if (err instanceof MCPError) throw err;
      throw new MCPError(ErrorCodes.FILE_NOT_FOUND, `File not found: ${params.path}`);
    }

    try {
      // Ensure trash directory exists
      await fs.mkdir(TRASH_DIR, { recursive: true });

      // Generate unique trash name to avoid collisions
      const timestamp = Date.now();
      const basename = path.basename(params.path);
      const trashName = `${timestamp}_${basename}`;
      const trashPath = path.join(TRASH_DIR, trashName);

      // Move to trash (rename) instead of permanent delete
      await fs.rename(params.path, trashPath);

      return {
        success: true,
        data: { success: true },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to delete file: ${msg}`);
    }
  },
};
