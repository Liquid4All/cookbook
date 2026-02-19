/**
 * filesystem.write_file — Write content to a file.
 *
 * Mutable: requires user confirmation before execution.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath } from '../../../_shared/ts/validation';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  path: z.string().describe('Absolute path to write'),
  content: z.string().describe('Content to write'),
  encoding: z
    .string()
    .optional()
    .default('utf-8')
    .describe('File encoding'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

export const writeFile: MCPTool<Params> = {
  name: 'filesystem.write_file',
  description: 'Write content to a file',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    assertAbsolutePath(params.path, 'path');
    assertSandboxed(params.path);

    try {
      // Ensure parent directory exists
      const dir = path.dirname(params.path);
      await fs.mkdir(dir, { recursive: true });

      await fs.writeFile(params.path, params.content, {
        encoding: params.encoding as BufferEncoding,
      });

      return {
        success: true,
        data: { success: true, path: params.path },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to write file: ${msg}`);
    }
  },
};
