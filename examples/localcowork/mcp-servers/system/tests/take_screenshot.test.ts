import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { takeScreenshot } from '../src/tools/take_screenshot';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system.take_screenshot', () => {
  beforeAll(() => {
    setupTestBridge();
  });

  afterAll(() => {
    teardownTestBridge();
  });

  it('should capture full screen screenshot when no region specified', async () => {
    const result = await takeScreenshot.execute({});
    expect(result.success).toBe(true);
    expect(result.data.width).toBe(1920);
    expect(result.data.height).toBe(1080);
  });

  it('should capture screenshot with region parameter', async () => {
    const result = await takeScreenshot.execute({
      region: { x: 100, y: 200, width: 800, height: 600 },
    });
    expect(result.success).toBe(true);
    expect(result.data.width).toBe(800);
    expect(result.data.height).toBe(600);
  });

  it('should return path, width, and height', async () => {
    const result = await takeScreenshot.execute({});
    const data = result.data;

    expect(data).toHaveProperty('path');
    expect(data).toHaveProperty('width');
    expect(data).toHaveProperty('height');

    expect(typeof data.path).toBe('string');
    expect(typeof data.width).toBe('number');
    expect(typeof data.height).toBe('number');
  });

  it('should validate region requires positive width and height', () => {
    const zeroWidth = takeScreenshot.paramsSchema.safeParse({
      region: { x: 0, y: 0, width: 0, height: 100 },
    });
    expect(zeroWidth.success).toBe(false);

    const zeroHeight = takeScreenshot.paramsSchema.safeParse({
      region: { x: 0, y: 0, width: 100, height: 0 },
    });
    expect(zeroHeight.success).toBe(false);
  });

  it('should validate region rejects negative coordinates', () => {
    const negX = takeScreenshot.paramsSchema.safeParse({
      region: { x: -1, y: 0, width: 100, height: 100 },
    });
    expect(negX.success).toBe(false);

    const negY = takeScreenshot.paramsSchema.safeParse({
      region: { x: 0, y: -1, width: 100, height: 100 },
    });
    expect(negY.success).toBe(false);
  });

  it('should return a file path string', async () => {
    const result = await takeScreenshot.execute({});
    // MockSystemBridge returns a predictable path
    expect(result.data.path).toBe('/tmp/screenshot-mock.png');
  });

  it('has correct metadata', () => {
    expect(takeScreenshot.name).toBe('system.take_screenshot');
    expect(takeScreenshot.confirmationRequired).toBe(false);
    expect(takeScreenshot.undoSupported).toBe(false);
  });
});
