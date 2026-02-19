/**
 * Test helpers for the system MCP server.
 *
 * Provides MockSystemBridge setup and teardown for unit tests.
 * All tools use the mock bridge during testing for predictable results.
 */

import { MockSystemBridge, setBridge, getBridge } from '../src/bridge';
import type { SystemBridge } from '../src/bridge';

/** The original bridge, saved before tests swap it out */
let originalBridge: SystemBridge | undefined;

/**
 * Set up the MockSystemBridge for testing.
 * Returns the mock bridge instance for assertions.
 */
export function setupTestBridge(): MockSystemBridge {
  originalBridge = getBridge();
  const mock = new MockSystemBridge();
  setBridge(mock);
  return mock;
}

/**
 * Restore the original bridge after tests complete.
 */
export function teardownTestBridge(): void {
  if (originalBridge) {
    setBridge(originalBridge);
    originalBridge = undefined;
  }
}
