/**
 * filesystem.search_files — Search for files matching a pattern.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath } from '../../../_shared/ts/validation';
import type { FileInfo } from './list_dir';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  path: z.string().describe('Root directory to search'),
  pattern: z.string().describe('Search pattern (glob or regex)'),
  type: z
    .enum(['file', 'dir', 'symlink'])
    .optional()
    .describe('Filter by file type'),
  max_results: z
    .number()
    .int()
    .positive()
    .optional()
    .default(100)
    .describe('Maximum results to return'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Implementation ─────────────────────────────────────────────────────────

function toRegex(pattern: string): RegExp {
  // If it looks like a glob (contains * or ?), convert to regex
  if (pattern.includes('*') || pattern.includes('?')) {
    const escaped = pattern
      .replace(/[.+^${}()|[\]\\]/g, '\\$&')
      .replace(/\*/g, '.*')
      .replace(/\?/g, '.');
    return new RegExp(escaped, 'i');
  }
  // Otherwise treat as regex
  try {
    return new RegExp(pattern, 'i');
  } catch {
    // Fall back to literal match if regex is invalid
    return new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
  }
}

async function walkDir(
  dirPath: string,
  regex: RegExp,
  fileType: string | undefined,
  maxResults: number,
  results: FileInfo[],
): Promise<void> {
  if (results.length >= maxResults) return;

  let entries;
  try {
    entries = await fs.readdir(dirPath, { withFileTypes: true });
  } catch {
    return; // Skip directories we can't read
  }

  for (const entry of entries) {
    if (results.length >= maxResults) break;

    const fullPath = path.join(dirPath, entry.name);
    const entryType = entry.isDirectory() ? 'dir' : entry.isSymbolicLink() ? 'symlink' : 'file';

    // Check type filter
    if (fileType && entryType !== fileType) {
      // Still recurse into dirs even if filtering for files
      if (entry.isDirectory()) {
        await walkDir(fullPath, regex, fileType, maxResults, results);
      }
      continue;
    }

    // Check pattern match
    if (regex.test(entry.name)) {
      let stat;
      try {
        stat = await fs.stat(fullPath);
      } catch {
        continue;
      }

      results.push({
        name: entry.name,
        path: fullPath,
        type: entryType as FileInfo['type'],
        size: stat.size,
        modified: stat.mtime.toISOString(),
      });
    }

    // Recurse into subdirectories
    if (entry.isDirectory()) {
      await walkDir(fullPath, regex, fileType, maxResults, results);
    }
  }
}

// ─── Tool Definition ────────────────────────────────────────────────────────

export const searchFiles: MCPTool<Params> = {
  name: 'filesystem.search_files',
  description: 'Search for files matching a pattern',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    assertAbsolutePath(params.path, 'path');
    assertSandboxed(params.path);

    try {
      const stat = await fs.stat(params.path);
      if (!stat.isDirectory()) {
        throw new MCPError(ErrorCodes.INVALID_PARAMS, `Not a directory: ${params.path}`);
      }
    } catch (err) {
      if (err instanceof MCPError) throw err;
      throw new MCPError(ErrorCodes.FILE_NOT_FOUND, `Directory not found: ${params.path}`);
    }

    const regex = toRegex(params.pattern);
    const results: FileInfo[] = [];

    await walkDir(params.path, regex, params.type, params.max_results, results);

    return { success: true, data: results };
  },
};
