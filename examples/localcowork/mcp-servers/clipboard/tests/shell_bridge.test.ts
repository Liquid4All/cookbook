/**
 * Integration tests for the ShellBridge (macOS only).
 *
 * These tests interact with the real OS clipboard via pbpaste/pbcopy.
 * Skipped on non-macOS platforms (CI, Linux, Windows).
 */

import { describe, it, expect } from 'vitest';
import { ShellBridge } from '../src/bridge';

const isMac = process.platform === 'darwin';

describe.skipIf(!isMac)('ShellBridge (macOS integration)', () => {
  it('should write and read back from the OS clipboard', async () => {
    const bridge = new ShellBridge();
    const marker = `localcowork-test-${Date.now()}`;

    const writeOk = await bridge.write(marker);
    expect(writeOk).toBe(true);

    const result = await bridge.read();
    expect(result.content).toBe(marker);
    expect(result.type).toBe('text/plain');
  });

  it('should read non-empty clipboard after write', async () => {
    const bridge = new ShellBridge();
    await bridge.write('shell-bridge-read-test');

    const result = await bridge.read();
    expect(result.content.length).toBeGreaterThan(0);
  });

  it('should handle multi-line content', async () => {
    const bridge = new ShellBridge();
    const multiline = 'Line 1\nLine 2\nLine 3';

    await bridge.write(multiline);
    const result = await bridge.read();
    expect(result.content).toBe(multiline);
  });

  it('should handle unicode content', async () => {
    const bridge = new ShellBridge();
    const unicode = 'Hello from LocalCowork \u2014 \u2705\ud83d\ude80';

    await bridge.write(unicode);
    const result = await bridge.read();
    expect(result.content).toBe(unicode);
  });
});
