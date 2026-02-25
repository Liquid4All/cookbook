/**
 * LocalCowork — Cross-Platform Helpers
 *
 * Utility functions for cross-platform test logic. Provides platform-aware
 * path normalization, temp directory creation, hidden file handling, and
 * filesystem metadata queries.
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { execSync } from 'child_process';

// --- Platform Constants ---

const PLATFORM = os.platform();

/** Maximum path length per platform (conservative estimates) */
const MAX_PATH_LENGTHS: Readonly<Record<string, number>> = {
  darwin: 1024,
  linux: 4096,
  win32: 260,
};

// --- Exported Helpers ---

/** Return the platform-appropriate temp directory. */
export function getPlatformTempDir(): string {
  return os.tmpdir();
}

/**
 * Normalize a file path for the current platform.
 * Resolves . / .. segments and normalizes separators.
 */
export function normalizePath(p: string): string {
  return path.normalize(p);
}

/** Return the expected line ending for the current platform. */
export function getExpectedLineEnding(): string {
  return PLATFORM === 'win32' ? '\r\n' : '\n';
}

/**
 * Detect whether the filesystem at the given path is case-sensitive.
 * Creates a temporary probe file to test.
 */
export async function isCaseSensitiveFS(testDir?: string): Promise<boolean> {
  const dir = testDir ?? getPlatformTempDir();
  const stamp = Date.now();
  const probeLower = path.join(dir, `__case_probe_${stamp}_a`);
  const probeUpper = path.join(dir, `__case_probe_${stamp}_A`);

  try {
    await fs.writeFile(probeLower, 'lower', 'utf-8');
    try {
      await fs.writeFile(probeUpper, 'upper', 'utf-8');
      const content = await fs.readFile(probeLower, 'utf-8');
      return content === 'lower';
    } catch {
      return true;
    }
  } finally {
    await fs.unlink(probeLower).catch(() => {});
    await fs.unlink(probeUpper).catch(() => {});
  }
}

/** Return the maximum path length for the current OS. */
export function getMaxPathLength(): number {
  return MAX_PATH_LENGTHS[PLATFORM] ?? 4096;
}

/**
 * Create a hidden file in the given directory.
 * Unix: prefix with dot. Windows: attrib +H.
 * Returns the full path of the created file.
 */
export async function createHiddenFile(
  dir: string,
  name: string,
  content: string,
): Promise<string> {
  let filePath: string;

  if (PLATFORM === 'win32') {
    filePath = path.join(dir, name);
    await fs.writeFile(filePath, content, 'utf-8');
    try {
      execSync(`attrib +H "${filePath}"`, { timeout: 5000 });
    } catch {
      // attrib may fail in CI; file is still created
    }
  } else {
    const hiddenName = name.startsWith('.') ? name : `.${name}`;
    filePath = path.join(dir, hiddenName);
    await fs.writeFile(filePath, content, 'utf-8');
  }

  return filePath;
}

/** Check whether a file is hidden on the current platform. */
export async function isHiddenFile(filePath: string): Promise<boolean> {
  const basename = path.basename(filePath);

  if (PLATFORM === 'win32') {
    try {
      const output = execSync(`attrib "${filePath}"`, {
        encoding: 'utf-8',
        timeout: 5000,
      });
      return output.includes('H');
    } catch {
      return false;
    }
  }

  return basename.startsWith('.');
}

/** Create a unique temp subdirectory for test isolation. */
export async function createTestTempDir(prefix: string): Promise<string> {
  const tempBase = getPlatformTempDir();
  const rand = Math.random().toString(36).slice(2, 8);
  const dirName = `localcowork-test-${prefix}-${Date.now()}-${rand}`;
  const dirPath = path.join(tempBase, dirName);
  await fs.mkdir(dirPath, { recursive: true });
  return dirPath;
}

/** Recursively remove a directory and all its contents. */
export async function cleanupTestDir(dirPath: string): Promise<void> {
  try {
    await fs.rm(dirPath, { recursive: true, force: true });
  } catch {
    // Best-effort cleanup; some platforms may hold file locks
  }
}
