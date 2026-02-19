/**
 * filesystem.watch_folder — Register a filesystem watcher on a directory.
 *
 * Mutable: requires first-time confirmation per directory.
 * Uses Node.js native fs.watch (chokidar available as upgrade path).
 */

import * as fs from 'fs';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath } from '../../../_shared/ts/validation';

// ─── Watcher Registry ───────────────────────────────────────────────────────

const activeWatchers = new Map<string, fs.FSWatcher>();

/** Close all active watchers. Called on server shutdown. */
export function closeAllWatchers(): void {
  for (const [id, watcher] of activeWatchers) {
    watcher.close();
    activeWatchers.delete(id);
  }
}

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  path: z.string().describe('Absolute path to watch'),
  patterns: z
    .array(z.string())
    .optional()
    .describe('Glob patterns to filter events'),
  callback_id: z.string().describe('Unique ID for callback routing'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

export const watchFolder: MCPTool<Params> = {
  name: 'filesystem.watch_folder',
  description: 'Register a filesystem watcher on a directory',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    assertAbsolutePath(params.path, 'path');
    assertSandboxed(params.path);

    // Verify directory exists
    try {
      const stat = fs.statSync(params.path);
      if (!stat.isDirectory()) {
        throw new MCPError(ErrorCodes.INVALID_PARAMS, `Not a directory: ${params.path}`);
      }
    } catch (err) {
      if (err instanceof MCPError) throw err;
      throw new MCPError(ErrorCodes.FILE_NOT_FOUND, `Directory not found: ${params.path}`);
    }

    // Close existing watcher with same callback_id if any
    const existing = activeWatchers.get(params.callback_id);
    if (existing) {
      existing.close();
    }

    const watcherId = params.callback_id;

    // Create the watcher
    const watcher = fs.watch(params.path, { recursive: true }, (eventType, filename) => {
      // In a full implementation, this would send a JSON-RPC notification
      // back through stdout to the Agent Core. For now, the watcher is
      // registered and events are captured for future routing.
      if (filename && params.patterns && params.patterns.length > 0) {
        const matches = params.patterns.some((p) => {
          const regex = new RegExp(
            p.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.'),
            'i',
          );
          return regex.test(filename);
        });
        if (!matches) return;
      }
      // Event would be routed via callback_id
    });

    activeWatchers.set(watcherId, watcher);

    return {
      success: true,
      data: { watcher_id: watcherId },
    };
  },
};
