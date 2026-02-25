/**
 * Test helpers for the clipboard MCP server.
 *
 * Provides mock bridge setup/teardown and optional history seeding.
 */

import {
  MockClipboardBridge,
  setBridge,
  clearHistory,
  addToHistory,
} from '../src/bridge';

/** Seed options for clipboard tests. */
export interface SeedOptions {
  /** Pre-fill the clipboard with this content. */
  readonly initialContent?: string;
  /** Number of history entries to pre-seed. */
  readonly historyCount?: number;
}

/**
 * Set up a fresh MockClipboardBridge with optional seed data.
 * Returns the bridge instance for direct assertions.
 */
export function setupTestBridge(opts?: SeedOptions): MockClipboardBridge {
  const bridge = new MockClipboardBridge();
  setBridge(bridge);
  clearHistory();

  // Pre-fill clipboard content if requested
  if (opts?.initialContent !== undefined) {
    void bridge.write(opts.initialContent);
  }

  // Seed clipboard history if requested
  const historyCount = opts?.historyCount ?? 0;
  for (let i = 0; i < historyCount; i++) {
    addToHistory(`History entry ${i + 1}`, 'text/plain');
  }

  return bridge;
}

/** Tear down the test bridge and clear all history. */
export function teardownTestBridge(): void {
  const fresh = new MockClipboardBridge();
  setBridge(fresh);
  clearHistory();
}
