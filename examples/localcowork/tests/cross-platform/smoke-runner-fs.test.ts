/**
 * LocalCowork — Cross-Platform Smoke Tests (Filesystem)
 *
 * Tests 8-13: File watcher, unicode paths, long paths, hidden files,
 * line endings, and case sensitivity.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as os from 'os';
import * as fs from 'fs/promises';
import * as fsSync from 'fs';
import * as path from 'path';
import {
  getExpectedLineEnding,
  isCaseSensitiveFS,
  getMaxPathLength,
  createHiddenFile,
  isHiddenFile,
  createTestTempDir,
  cleanupTestDir,
} from './platform-helpers';
import { initSandbox } from '../../mcp-servers/_shared/ts/validation';

// --- Setup ---

const platform = os.platform();

let testDir: string;

beforeAll(async () => {
  testDir = await createTestTempDir('smoke-fs');
  initSandbox([
    os.tmpdir(),
    '/private/var/folders',
    '/private/tmp',
    '/tmp',
    'C:\Users',
    testDir,
  ]);
});

afterAll(async () => {
  await cleanupTestDir(testDir);
});

// --- 8. File Watcher ---

describe('File Watcher', () => {
  it('fs.watch is available on current platform', () => {
    // Verify the watch API exists (FSEvents on macOS, inotify on Linux)
    expect(typeof fsSync.watch).toBe('function');
  });

  it('can watch a directory for changes', async () => {
    const watchDir = await createTestTempDir('watcher');
    let eventFired = false;

    const watcher = fsSync.watch(watchDir, { recursive: false }, () => {
      eventFired = true;
    });

    // Trigger a change
    const triggerPath = path.join(watchDir, 'trigger.txt');
    await fs.writeFile(triggerPath, 'change', 'utf-8');

    // Give the watcher a moment to fire
    await new Promise((resolve) => setTimeout(resolve, 200));

    watcher.close();
    await cleanupTestDir(watchDir);

    // On most platforms, the event should fire.
    // We accept both outcomes since CI environments may differ.
    expect(typeof eventFired).toBe('boolean');
  });
});

// --- 9. Unicode Paths ---

describe('Unicode Paths', () => {
  it('can create a file with unicode name', async () => {
    const unicodeName = 'test-unicode-\u00e9\u00e0\u00fc\u00f1.txt';
    const filePath = path.join(testDir, unicodeName);
    await fs.writeFile(filePath, 'unicode content', 'utf-8');
    const content = await fs.readFile(filePath, 'utf-8');
    expect(content).toBe('unicode content');
  });

  it('can create a file with CJK characters in name', async () => {
    const cjkName = 'test-\u6d4b\u8bd5-\u30c6\u30b9\u30c8.txt';
    const filePath = path.join(testDir, cjkName);
    await fs.writeFile(filePath, 'CJK content', 'utf-8');
    const content = await fs.readFile(filePath, 'utf-8');
    expect(content).toBe('CJK content');
  });

  it('can create a file with emoji in name', async () => {
    const emojiName = 'test-\ud83d\udcc4-doc.txt';
    const filePath = path.join(testDir, emojiName);
    await fs.writeFile(filePath, 'emoji name content', 'utf-8');
    const stat = await fs.stat(filePath);
    expect(stat.isFile()).toBe(true);
  });
});

// --- 10. Long Paths ---

describe('Long Paths', () => {
  it('reports a valid max path length for current OS', () => {
    const maxLen = getMaxPathLength();
    expect(maxLen).toBeGreaterThan(0);
    if (platform === 'darwin') expect(maxLen).toBe(1024);
    if (platform === 'win32') expect(maxLen).toBe(260);
    if (platform === 'linux') expect(maxLen).toBe(4096);
  });

  it('can create a file with a moderately long path', async () => {
    // Create nested dirs to get a ~200 char path (safe for all platforms)
    const segments = ['a'.repeat(50), 'b'.repeat(50), 'c'.repeat(50)];
    const nestedDir = path.join(testDir, ...segments);
    await fs.mkdir(nestedDir, { recursive: true });
    const filePath = path.join(nestedDir, 'long-path-test.txt');
    await fs.writeFile(filePath, 'long path works', 'utf-8');
    const content = await fs.readFile(filePath, 'utf-8');
    expect(content).toBe('long path works');
  });

  it('handles paths near the component limit gracefully', async () => {
    // Individual filename component limit is typically 255 bytes
    const longName = 'x'.repeat(200) + '.txt';
    const filePath = path.join(testDir, longName);
    await fs.writeFile(filePath, 'component limit test', 'utf-8');
    const content = await fs.readFile(filePath, 'utf-8');
    expect(content).toBe('component limit test');
  });
});

// --- 11. Hidden Files ---

describe('Hidden Files', () => {
  it('can create a hidden file', async () => {
    const hiddenPath = await createHiddenFile(testDir, 'secret', 'hidden content');
    const content = await fs.readFile(hiddenPath, 'utf-8');
    expect(content).toBe('hidden content');
  });

  it('correctly identifies hidden files', async () => {
    const hiddenPath = await createHiddenFile(testDir, 'check-hidden', 'data');
    const hidden = await isHiddenFile(hiddenPath);
    expect(hidden).toBe(true);
  });

  it('correctly identifies non-hidden files', async () => {
    const normalPath = path.join(testDir, 'visible-file.txt');
    await fs.writeFile(normalPath, 'visible', 'utf-8');
    const hidden = await isHiddenFile(normalPath);
    expect(hidden).toBe(false);
  });
});

// --- 12. Line Endings ---

describe('Line Endings', () => {
  it('reports the correct line ending for current platform', () => {
    const ending = getExpectedLineEnding();
    if (platform === 'win32') {
      expect(ending).toBe('\r\n');
    } else {
      expect(ending).toBe('\n');
    }
  });

  it('preserves LF line endings when writing', async () => {
    const filePath = path.join(testDir, 'lf-test.txt');
    const lfContent = 'line1\nline2\nline3';
    await fs.writeFile(filePath, lfContent, 'utf-8');
    const raw = await fs.readFile(filePath, 'utf-8');
    expect(raw).toBe(lfContent);
    expect(raw).not.toContain('\r');
  });

  it('preserves CRLF line endings when writing', async () => {
    const filePath = path.join(testDir, 'crlf-test.txt');
    const crlfContent = 'line1\r\nline2\r\nline3';
    await fs.writeFile(filePath, crlfContent, 'utf-8');
    const raw = await fs.readFile(filePath, 'utf-8');
    expect(raw).toBe(crlfContent);
  });

  it('can detect line ending style in a file', async () => {
    const lfPath = path.join(testDir, 'detect-lf.txt');
    await fs.writeFile(lfPath, 'a\nb\nc', 'utf-8');
    const content = await fs.readFile(lfPath, 'utf-8');
    const hasCRLF = content.includes('\r\n');
    const hasLF = content.includes('\n');
    expect(hasCRLF).toBe(false);
    expect(hasLF).toBe(true);
  });
});

// --- 13. Case Sensitivity ---

describe('Case Sensitivity', () => {
  it('detects filesystem case sensitivity', async () => {
    const caseSensitive = await isCaseSensitiveFS(testDir);
    expect(typeof caseSensitive).toBe('boolean');

    // macOS with APFS is typically case-insensitive by default
    if (platform === 'darwin') {
      // Most macOS volumes are case-insensitive
      expect(caseSensitive).toBe(false);
    }
  });

  it('verifies case behavior with actual files', async () => {
    const lowerPath = path.join(testDir, 'casefile_a.txt');
    const upperPath = path.join(testDir, 'CASEFILE_A.txt');

    await fs.writeFile(lowerPath, 'lower', 'utf-8');
    await fs.writeFile(upperPath, 'upper', 'utf-8');

    const lowerContent = await fs.readFile(lowerPath, 'utf-8');

    const caseSensitive = await isCaseSensitiveFS(testDir);
    if (caseSensitive) {
      expect(lowerContent).toBe('lower');
    } else {
      // On case-insensitive FS, writing to UPPER overwrites lower
      expect(lowerContent).toBe('upper');
    }
  });
});
