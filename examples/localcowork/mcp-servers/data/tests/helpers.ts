/**
 * Test helpers for the data MCP server.
 *
 * Provides temp directory setup and sandbox initialization.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { initSandbox } from '../../_shared/ts/validation';

/** Create a temp directory for tests and configure sandbox */
export function setupTestDir(): string {
  const testDir = fs.mkdtempSync(path.join(os.tmpdir(), 'data-test-'));

  // Initialize sandbox with broad temp roots to avoid parallel test conflicts
  initSandbox([os.tmpdir(), '/private/var/folders', '/private/tmp', '/tmp']);

  return testDir;
}

/** Remove the temp directory */
export function teardownTestDir(testDir: string): void {
  fs.rmSync(testDir, { recursive: true, force: true });
}
