/**
 * filesystem.read_file — Read contents of a file.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 */

import * as fs from 'fs/promises';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath } from '../../../_shared/ts/validation';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  path: z.string().describe('Absolute path to file'),
  encoding: z
    .string()
    .optional()
    .default('utf-8')
    .describe('File encoding'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Tool Definition ────────────────────────────────────────────────────────

export const readFile: MCPTool<Params> = {
  name: 'filesystem.read_file',
  description: 'Read contents of a file',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    assertAbsolutePath(params.path, 'path');
    assertSandboxed(params.path);

    try {
      const stat = await fs.stat(params.path);
      if (!stat.isFile()) {
        throw new MCPError(ErrorCodes.INVALID_PARAMS, `Not a file: ${params.path}`);
      }

      // Normalize encoding name (e.g., 'utf-8' → 'utf8') for Node.js compatibility
      const rawEncoding = params.encoding ?? 'utf-8';
      const encoding = rawEncoding.replace('-', '') as BufferEncoding;
      const content = await fs.readFile(params.path, { encoding });

      return {
        success: true,
        data: { content, size: stat.size },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      throw new MCPError(ErrorCodes.FILE_NOT_FOUND, `Cannot read file: ${params.path}`);
    }
  },
};
