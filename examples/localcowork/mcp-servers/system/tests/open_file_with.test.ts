import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { openFileWith } from '../src/tools/open_file_with';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system.open_file_with', () => {
  beforeAll(() => {
    setupTestBridge();
  });

  afterAll(() => {
    teardownTestBridge();
  });

  it('should open a file with the default application', async () => {
    const result = await openFileWith.execute({ path: '/Users/user/document.pdf' });
    expect(result.success).toBe(true);
    expect(result.data.success).toBe(true);
  });

  it('should open a file with a specific application', async () => {
    const result = await openFileWith.execute({
      path: '/Users/user/photo.png',
      app: 'Preview',
    });
    expect(result.success).toBe(true);
    expect(result.data.success).toBe(true);
  });

  it('should reject empty path via schema validation', () => {
    const parseResult = openFileWith.paramsSchema.safeParse({ path: '' });
    expect(parseResult.success).toBe(false);
  });

  it('should reject missing path via schema validation', () => {
    const parseResult = openFileWith.paramsSchema.safeParse({});
    expect(parseResult.success).toBe(false);
  });

  it('should accept path without app (app is optional)', () => {
    const parseResult = openFileWith.paramsSchema.safeParse({
      path: '/some/file.txt',
    });
    expect(parseResult.success).toBe(true);
  });

  it('has confirmation metadata set to true', () => {
    expect(openFileWith.confirmationRequired).toBe(true);
    expect(openFileWith.undoSupported).toBe(false);
  });

  it('has correct tool name', () => {
    expect(openFileWith.name).toBe('system.open_file_with');
  });
});
