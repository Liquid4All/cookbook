import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { clipboardHistory } from '../src/tools/clipboard_history';
import { setupTestBridge, teardownTestBridge } from './helpers';
import { addToHistory } from '../src/bridge';

describe('clipboard.clipboard_history', () => {
  beforeEach(() => {
    setupTestBridge();
  });

  afterEach(() => {
    teardownTestBridge();
  });

  it('should return empty array when no history exists', async () => {
    const result = await clipboardHistory.execute({});
    expect(result.success).toBe(true);
    expect(result.data.entries).toEqual([]);
  });

  it('should return seeded history entries (most recent first)', async () => {
    setupTestBridge({ historyCount: 3 });

    const result = await clipboardHistory.execute({});
    expect(result.success).toBe(true);
    expect(result.data.entries.length).toBe(3);
    // Most recent is last seeded entry
    expect(result.data.entries[0].content).toBe('History entry 3');
    expect(result.data.entries[2].content).toBe('History entry 1');
  });

  it('should respect the limit parameter', async () => {
    setupTestBridge({ historyCount: 10 });

    const result = await clipboardHistory.execute({ limit: 3 });
    expect(result.success).toBe(true);
    expect(result.data.entries.length).toBe(3);
  });

  it('should use default limit of 20', async () => {
    setupTestBridge({ historyCount: 25 });

    const result = await clipboardHistory.execute({});
    expect(result.success).toBe(true);
    expect(result.data.entries.length).toBe(20);
  });

  it('should return fewer entries than limit when history is small', async () => {
    setupTestBridge({ historyCount: 2 });

    const result = await clipboardHistory.execute({ limit: 50 });
    expect(result.success).toBe(true);
    expect(result.data.entries.length).toBe(2);
  });

  it('should enforce max limit of 100', () => {
    const parseResult = clipboardHistory.paramsSchema.safeParse({ limit: 200 });
    expect(parseResult.success).toBe(false);
  });

  it('should enforce min limit of 1', () => {
    const parseResult = clipboardHistory.paramsSchema.safeParse({ limit: 0 });
    expect(parseResult.success).toBe(false);
  });

  it('should have correct entry structure', async () => {
    addToHistory('Structured entry', 'text/plain');

    const result = await clipboardHistory.execute({});
    expect(result.data.entries.length).toBe(1);

    const entry = result.data.entries[0];
    expect(entry).toHaveProperty('content');
    expect(entry).toHaveProperty('type');
    expect(entry).toHaveProperty('timestamp');
    expect(typeof entry.content).toBe('string');
    expect(typeof entry.type).toBe('string');
    expect(typeof entry.timestamp).toBe('string');
    // Timestamp should be ISO 8601
    expect(() => new Date(entry.timestamp)).not.toThrow();
    expect(new Date(entry.timestamp).toISOString()).toBe(entry.timestamp);
  });

  it('has correct tool metadata', () => {
    expect(clipboardHistory.name).toBe('clipboard.clipboard_history');
    expect(clipboardHistory.description).toBe('Get recent clipboard entries');
    expect(clipboardHistory.confirmationRequired).toBe(false);
    expect(clipboardHistory.undoSupported).toBe(false);
  });
});
