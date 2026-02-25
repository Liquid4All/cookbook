/**
 * system-settings.set_audio_volume — Set system audio volume.
 *
 * MUTABLE: requires user confirmation. Undo supported (stores previous value).
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { SetResult } from '../bridge';

const paramsSchema = z
  .object({
    volume: z
      .number()
      .int()
      .min(0)
      .max(100)
      .describe('Volume level 0-100'),
  })
  .describe('Set audio output volume');

type Params = z.infer<typeof paramsSchema>;

export const setAudioVolume: MCPTool<Params> = {
  name: 'system-settings.set_audio_volume',
  description: 'Set the system audio output volume (0-100)',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: true,

  async execute(params: Params): Promise<MCPResult<SetResult>> {
    try {
      const data = await getBridge().setAudioVolume(params.volume);
      return { success: true, data };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to set audio volume: ${msg}`);
    }
  },
};
