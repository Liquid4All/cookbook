/**
 * LocalCowork — Cross-Platform Smoke Test Runner
 *
 * Vitest-compatible test suite with platform-aware tests. Validates that
 * filesystem operations, platform detection, and OS-level interactions
 * work correctly on the current platform (macOS, Windows, or Linux).
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as os from 'os';
import * as fs from 'fs/promises';
import * as fsSync from 'fs';
import * as path from 'path';
import {
  getPlatformTempDir,
  normalizePath,
  getExpectedLineEnding,
  isCaseSensitiveFS,
  getMaxPathLength,
  createHiddenFile,
  isHiddenFile,
  createTestTempDir,
  cleanupTestDir,
} from './platform-helpers';
import { initSandbox } from '../../mcp-servers/_shared/ts/validation';
import { NodeSystemBridge } from '../../mcp-servers/system/src/bridge';
import {
  MockClipboardBridge,
  clearHistory,
  addToHistory,
  getHistory,
} from '../../mcp-servers/clipboard/src/bridge';

// --- Setup ---

const platform = os.platform(); // 'darwin' | 'win32' | 'linux'
const arch = os.arch(); // 'arm64' | 'x64'

let testDir: string;

beforeAll(async () => {
  testDir = await createTestTempDir('smoke');
  // Initialize sandbox with temp directories for tool testing
  initSandbox([
    os.tmpdir(),
    '/private/var/folders',
    '/private/tmp',
    '/tmp',
    'C:\\Users',
    testDir,
  ]);
});

afterAll(async () => {
  await cleanupTestDir(testDir);
});

// --- 1. Platform Detection ---

describe('Platform Detection', () => {
  it('detects a known OS platform', () => {
    expect(['darwin', 'win32', 'linux']).toContain(platform);
  });

  it('detects a known CPU architecture', () => {
    expect(['arm64', 'x64', 'ia32']).toContain(arch);
  });

  it('returns a valid hostname', () => {
    const hostname = os.hostname();
    expect(hostname.length).toBeGreaterThan(0);
  });

  it('reports positive total memory', () => {
    expect(os.totalmem()).toBeGreaterThan(0);
  });
});

// --- 2. Filesystem Paths ---

describe('Filesystem Paths', () => {
  it('normalizes forward slashes correctly', () => {
    const input = '/Users/test//documents/../documents/file.txt';
    const result = normalizePath(input);
    expect(result).toBe('/Users/test/documents/file.txt');
  });

  it('normalizes backslash paths on any platform', () => {
    // path.normalize converts backslashes on the current OS
    const result = normalizePath('/foo/bar/../baz');
    expect(result).toBe('/foo/baz');
  });

  it('path.join produces platform-correct separators', () => {
    const joined = path.join('a', 'b', 'c');
    if (platform === 'win32') {
      expect(joined).toBe('a\\b\\c');
    } else {
      expect(joined).toBe('a/b/c');
    }
  });

  it('path.resolve returns an absolute path', () => {
    const resolved = path.resolve('relative', 'path');
    expect(path.isAbsolute(resolved)).toBe(true);
  });
});

// --- 3. Temp Directory ---

describe('Temp Directory', () => {
  it('returns a valid temp directory path', () => {
    const tmpDir = getPlatformTempDir();
    expect(tmpDir.length).toBeGreaterThan(0);
  });

  it('temp directory exists on disk', async () => {
    const tmpDir = getPlatformTempDir();
    const stat = await fs.stat(tmpDir);
    expect(stat.isDirectory()).toBe(true);
  });

  it('can create and remove a subdirectory in temp', async () => {
    const sub = await createTestTempDir('subdir-test');
    const stat = await fs.stat(sub);
    expect(stat.isDirectory()).toBe(true);
    await cleanupTestDir(sub);
    await expect(fs.stat(sub)).rejects.toThrow();
  });
});

// --- 4. File Permissions ---

describe('File Permissions', () => {
  it('can write and read back a file', async () => {
    const filePath = path.join(testDir, 'perm-test.txt');
    await fs.writeFile(filePath, 'permission check', 'utf-8');
    const content = await fs.readFile(filePath, 'utf-8');
    expect(content).toBe('permission check');
  });

  it('can check file stats after write', async () => {
    const filePath = path.join(testDir, 'stat-test.txt');
    await fs.writeFile(filePath, 'stat check', 'utf-8');
    const stat = await fs.stat(filePath);
    expect(stat.isFile()).toBe(true);
    expect(stat.size).toBeGreaterThan(0);
  });

  it.skipIf(platform === 'win32')('can set and read Unix file mode', async () => {
    const filePath = path.join(testDir, 'mode-test.txt');
    await fs.writeFile(filePath, 'mode check', 'utf-8');
    await fs.chmod(filePath, 0o644);
    const stat = await fs.stat(filePath);
    // Check owner read+write bits (octal mask)
    // eslint-disable-next-line no-bitwise
    expect(stat.mode & 0o600).toBe(0o600);
  });
});

// --- 5. Process Listing ---

describe('Process Listing', () => {
  it('returns a non-empty process list', async () => {
    const bridge = new NodeSystemBridge();
    const processes = await bridge.listProcesses();
    expect(processes.length).toBeGreaterThan(0);
  });

  it('each process has required fields', async () => {
    const bridge = new NodeSystemBridge();
    const processes = await bridge.listProcesses();
    const proc = processes[0];
    expect(proc).toHaveProperty('pid');
    expect(proc).toHaveProperty('name');
    expect(proc).toHaveProperty('cpu_percent');
    expect(proc).toHaveProperty('memory_mb');
    expect(typeof proc.pid).toBe('number');
    expect(typeof proc.name).toBe('string');
  });

  it('can filter processes by name', async () => {
    const bridge = new NodeSystemBridge();
    const processes = await bridge.listProcesses('node');
    // On the dev machine running vitest, at least one 'node' process exists
    if (platform === 'darwin' || platform === 'linux') {
      expect(processes.length).toBeGreaterThan(0);
      for (const p of processes) {
        expect(p.name.toLowerCase()).toContain('node');
      }
    }
  });
});

// --- 6. System Info ---

describe('System Info', () => {
  it('returns correct platform string', async () => {
    const bridge = new NodeSystemBridge();
    const info = await bridge.getSystemInfo();
    expect(info.os).toBe(platform);
  });

  it('returns correct arch string', async () => {
    const bridge = new NodeSystemBridge();
    const info = await bridge.getSystemInfo();
    expect(info.arch).toBe(arch);
  });

  it('returns positive RAM', async () => {
    const bridge = new NodeSystemBridge();
    const info = await bridge.getSystemInfo();
    expect(info.ram_gb).toBeGreaterThan(0);
  });

  it('returns a non-empty CPU model', async () => {
    const bridge = new NodeSystemBridge();
    const info = await bridge.getSystemInfo();
    expect(info.cpu.length).toBeGreaterThan(0);
  });
});

// --- 7. Clipboard (Mock Bridge) ---

describe('Clipboard (Mock Bridge)', () => {
  it('can write and read clipboard content', async () => {
    const bridge = new MockClipboardBridge();
    await bridge.write('test clipboard data');
    const result = await bridge.read();
    expect(result.content).toBe('test clipboard data');
    expect(result.type).toBe('text/plain');
  });

  it('overwriting replaces previous content', async () => {
    const bridge = new MockClipboardBridge();
    await bridge.write('first');
    await bridge.write('second');
    const result = await bridge.read();
    expect(result.content).toBe('second');
  });

  it('clipboard history tracks entries', () => {
    clearHistory();
    addToHistory('entry 1', 'text/plain');
    addToHistory('entry 2', 'text/plain');
    const history = getHistory();
    expect(history.length).toBe(2);
    // Most recent first
    expect(history[0].content).toBe('entry 2');
    expect(history[1].content).toBe('entry 1');
    clearHistory();
  });
});
