import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setClipboard } from '../src/tools/set_clipboard';
import { getClipboard } from '../src/tools/get_clipboard';
import { setupTestBridge, teardownTestBridge } from './helpers';
import { getHistory } from '../src/bridge';

describe('clipboard.set_clipboard', () => {
  beforeEach(() => {
    setupTestBridge();
  });

  afterEach(() => {
    teardownTestBridge();
  });

  it('should set text content successfully', async () => {
    const result = await setClipboard.execute({ content: 'Hello world' });
    expect(result.success).toBe(true);
    expect(result.data.success).toBe(true);
  });

  it('should reject empty string (validation error)', () => {
    const parseResult = setClipboard.paramsSchema.safeParse({ content: '' });
    expect(parseResult.success).toBe(false);
  });

  it('should update the clipboard after set', async () => {
    await setClipboard.execute({ content: 'New clipboard value' });

    const readResult = await getClipboard.execute({});
    expect(readResult.data.content).toBe('New clipboard value');
  });

  it('should add an entry to history after set', async () => {
    await setClipboard.execute({ content: 'Tracked content' });

    const history = getHistory();
    expect(history.length).toBe(1);
    expect(history[0].content).toBe('Tracked content');
    expect(history[0].type).toBe('text/plain');
    expect(typeof history[0].timestamp).toBe('string');
  });

  it('should have the correct result structure', async () => {
    const result = await setClipboard.execute({ content: 'Test' });
    expect(result).toHaveProperty('success');
    expect(result).toHaveProperty('data');
    expect(result.data).toHaveProperty('success');
    expect(typeof result.data.success).toBe('boolean');
  });

  it('should accumulate history across multiple sets', async () => {
    await setClipboard.execute({ content: 'First' });
    await setClipboard.execute({ content: 'Second' });
    await setClipboard.execute({ content: 'Third' });

    const history = getHistory();
    expect(history.length).toBe(3);
    // Most recent first
    expect(history[0].content).toBe('Third');
    expect(history[1].content).toBe('Second');
    expect(history[2].content).toBe('First');
  });

  it('has correct tool metadata', () => {
    expect(setClipboard.name).toBe('clipboard.set_clipboard');
    expect(setClipboard.description).toBe('Set clipboard contents');
    expect(setClipboard.confirmationRequired).toBe(false);
    expect(setClipboard.undoSupported).toBe(false);
  });
});
