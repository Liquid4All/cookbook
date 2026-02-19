/**
 * filesystem.list_dir — List contents of a directory.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath } from '../../../_shared/ts/validation';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface FileInfo {
  name: string;
  path: string;
  type: 'file' | 'dir' | 'symlink';
  size: number;
  modified: string;
}

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  path: z.string().describe('Absolute path to directory'),
  recursive: z.boolean().optional().default(false).describe('List recursively'),
  filter: z.string().optional().describe('Glob pattern filter'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Implementation ─────────────────────────────────────────────────────────

async function listEntries(dirPath: string, recursive: boolean): Promise<FileInfo[]> {
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  const results: FileInfo[] = [];

  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    let stat;
    try {
      stat = await fs.stat(fullPath);
    } catch {
      // Skip entries we can't stat (broken symlinks, etc.)
      continue;
    }

    const fileType = entry.isDirectory() ? 'dir' : entry.isSymbolicLink() ? 'symlink' : 'file';

    results.push({
      name: entry.name,
      path: fullPath,
      type: fileType as FileInfo['type'],
      size: stat.size,
      modified: stat.mtime.toISOString(),
    });

    if (recursive && entry.isDirectory()) {
      const nested = await listEntries(fullPath, true);
      results.push(...nested);
    }
  }

  return results;
}

function matchesGlob(name: string, pattern: string): boolean {
  // Simple glob matching: * matches any chars, ? matches single char
  const regex = pattern
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*/g, '.*')
    .replace(/\?/g, '.');
  return new RegExp(`^${regex}$`, 'i').test(name);
}

// ─── Tool Definition ────────────────────────────────────────────────────────

export const listDir: MCPTool<Params> = {
  name: 'filesystem.list_dir',
  description: 'List contents of a directory',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    assertAbsolutePath(params.path, 'path');
    assertSandboxed(params.path);

    try {
      const dirStat = await fs.stat(params.path);
      if (!dirStat.isDirectory()) {
        throw new MCPError(ErrorCodes.INVALID_PARAMS, `Not a directory: ${params.path}`);
      }
    } catch (err) {
      if (err instanceof MCPError) throw err;
      throw new MCPError(ErrorCodes.FILE_NOT_FOUND, `Directory not found: ${params.path}`);
    }

    const entries = await listEntries(params.path, params.recursive);

    const filtered = params.filter
      ? entries.filter((e) => matchesGlob(e.name, params.filter!))
      : entries;

    return { success: true, data: filtered };
  },
};
