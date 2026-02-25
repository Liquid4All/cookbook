/**
 * filesystem.get_metadata — Get detailed metadata for a file.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath, getFileCategory } from '../../../_shared/ts/validation';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  path: z.string().describe('Absolute path to file'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Helper ─────────────────────────────────────────────────────────────────

function formatPermissions(mode: number): string {
  const perms = ['---', '--x', '-w-', '-wx', 'r--', 'r-x', 'rw-', 'rwx'];
  const owner = perms[(mode >> 6) & 7];
  const group = perms[(mode >> 3) & 7];
  const others = perms[mode & 7];
  return `${owner}${group}${others}`;
}

// ─── Tool Definition ────────────────────────────────────────────────────────

export const getMetadata: MCPTool<Params> = {
  name: 'filesystem.get_metadata',
  description: 'Get detailed metadata for a file',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    assertAbsolutePath(params.path, 'path');
    assertSandboxed(params.path);

    try {
      const stat = await fs.stat(params.path);

      const fileType = stat.isDirectory()
        ? 'directory'
        : stat.isSymbolicLink()
          ? 'symlink'
          : 'file';

      const category = stat.isFile() ? getFileCategory(params.path) : fileType;
      const extension = path.extname(params.path);

      return {
        success: true,
        data: {
          size: stat.size,
          created: stat.birthtime.toISOString(),
          modified: stat.mtime.toISOString(),
          accessed: stat.atime.toISOString(),
          type: fileType,
          category,
          extension,
          permissions: formatPermissions(stat.mode),
        },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      throw new MCPError(ErrorCodes.FILE_NOT_FOUND, `Cannot read metadata: ${params.path}`);
    }
  },
};
