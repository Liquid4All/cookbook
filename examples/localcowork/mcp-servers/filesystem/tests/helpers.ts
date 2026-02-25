/**
 * Test helpers for the filesystem MCP server.
 *
 * Provides temporary directory setup/teardown and sandbox initialization.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { initSandbox } from '../../_shared/ts/validation';

/** Create a temporary test directory with optional files. */
export async function setupTestDir(opts?: {
  files?: string[];
  subdirs?: string[];
}): Promise<string> {
  const testDir = await fs.mkdtemp(path.join(os.tmpdir(), 'localcowork-test-'));

  // Initialize sandbox to allow the OS temp directory root.
  // This prevents parallel test files from clobbering each other's sandbox.
  initSandbox([os.tmpdir(), '/private/var/folders', '/private/tmp', '/tmp']);

  if (opts?.subdirs) {
    for (const subdir of opts.subdirs) {
      await fs.mkdir(path.join(testDir, subdir), { recursive: true });
    }
  }

  if (opts?.files) {
    for (const file of opts.files) {
      const filePath = path.join(testDir, file);
      const dir = path.dirname(filePath);
      await fs.mkdir(dir, { recursive: true });
      await fs.writeFile(filePath, `content of ${file}`, 'utf-8');
    }
  }

  return testDir;
}

/** Remove a temporary test directory. */
export async function teardownTestDir(testDir: string): Promise<void> {
  await fs.rm(testDir, { recursive: true, force: true });
}
