import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { getClipboard } from '../src/tools/get_clipboard';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('clipboard.get_clipboard', () => {
  beforeEach(() => {
    setupTestBridge();
  });

  afterEach(() => {
    teardownTestBridge();
  });

  it('should return empty string for a fresh clipboard', async () => {
    const result = await getClipboard.execute({});
    expect(result.success).toBe(true);
    expect(result.data.content).toBe('');
    expect(result.data.type).toBe('text/plain');
  });

  it('should return pre-filled content', async () => {
    setupTestBridge({ initialContent: 'Hello clipboard' });

    const result = await getClipboard.execute({});
    expect(result.success).toBe(true);
    expect(result.data.content).toBe('Hello clipboard');
  });

  it('should return updated content after a set operation', async () => {
    const bridge = setupTestBridge();
    await bridge.write('Updated value');

    const result = await getClipboard.execute({});
    expect(result.success).toBe(true);
    expect(result.data.content).toBe('Updated value');
  });

  it('should have the correct result structure', async () => {
    const result = await getClipboard.execute({});
    expect(result).toHaveProperty('success');
    expect(result).toHaveProperty('data');
    expect(result.data).toHaveProperty('content');
    expect(result.data).toHaveProperty('type');
    expect(typeof result.data.content).toBe('string');
    expect(typeof result.data.type).toBe('string');
  });

  it('has correct tool metadata', () => {
    expect(getClipboard.name).toBe('clipboard.get_clipboard');
    expect(getClipboard.description).toBe('Get current clipboard contents');
    expect(getClipboard.confirmationRequired).toBe(false);
    expect(getClipboard.undoSupported).toBe(false);
  });
});
