/**
 * system.take_screenshot — Capture a screenshot.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Captures full screen or a specified region.
 * Returns the file path and dimensions.
 */

import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { getBridge } from '../bridge';
import type { ScreenshotResult, ScreenRegion } from '../bridge';

const regionSchema = z
  .object({
    x: z.number().min(0).describe('X coordinate of the region'),
    y: z.number().min(0).describe('Y coordinate of the region'),
    width: z.number().min(1).describe('Width of the region'),
    height: z.number().min(1).describe('Height of the region'),
  })
  .describe('Screen region to capture');

const paramsSchema = z
  .object({
    region: regionSchema.optional().describe('Region { x, y, width, height } (default: full screen)'),
  })
  .describe('Capture a screenshot');

type Params = z.infer<typeof paramsSchema>;

export const takeScreenshot: MCPTool<Params> = {
  name: 'system.take_screenshot',
  description: 'Capture a screenshot',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult<ScreenshotResult>> {
    try {
      const bridge = getBridge();
      const region: ScreenRegion | undefined = params.region;
      const result = await bridge.takeScreenshot(region);

      return {
        success: true,
        data: result,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to take screenshot: ${msg}`);
    }
  },
};
