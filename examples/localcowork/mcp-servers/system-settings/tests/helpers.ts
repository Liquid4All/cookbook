/**
 * Test helpers for the system-settings MCP server.
 *
 * Provides MockSettingsBridge setup and teardown for unit tests.
 */

import { MockSettingsBridge, setBridge, getBridge } from '../src/bridge';
import type { SettingsBridge } from '../src/bridge';

let originalBridge: SettingsBridge | undefined;

export function setupTestBridge(): MockSettingsBridge {
  originalBridge = getBridge();
  const mock = new MockSettingsBridge();
  setBridge(mock);
  return mock;
}

export function teardownTestBridge(): void {
  if (originalBridge) {
    setBridge(originalBridge);
    originalBridge = undefined;
  }
}
