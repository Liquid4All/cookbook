/**
 * Clipboard Bridge Abstraction
 *
 * Provides a swappable bridge layer between the MCP server and the
 * actual clipboard implementation. In standalone/test mode, uses an
 * in-memory mock. When running inside Tauri, the TauriBridge will be
 * wired to the native IPC calls.
 */

// ── Types ────────────────────────────────────────────────────────────────────

/** A single clipboard history entry. */
export interface ClipboardEntry {
  readonly content: string;
  readonly type: string;
  readonly timestamp: string;
}

/** Abstraction over the OS clipboard (or mock). */
export interface ClipboardBridge {
  /** Read the current clipboard contents. */
  read(): Promise<{ content: string; type: string }>;

  /** Write content to the clipboard. Returns true on success. */
  write(content: string): Promise<boolean>;
}

// ── History Store ────────────────────────────────────────────────────────────

const MAX_HISTORY = 100;

let history: ClipboardEntry[] = [];

/** Return a copy of the clipboard history (most-recent first). */
export function getHistory(): ClipboardEntry[] {
  return [...history];
}

/** Add an entry to the clipboard history. */
export function addToHistory(content: string, type: string): void {
  const entry: ClipboardEntry = {
    content,
    type,
    timestamp: new Date().toISOString(),
  };
  history.unshift(entry);
  if (history.length > MAX_HISTORY) {
    history = history.slice(0, MAX_HISTORY);
  }
}

/** Clear the history (used for testing). */
export function clearHistory(): void {
  history = [];
}

// ── Mock Bridge ──────────────────────────────────────────────────────────────

/**
 * In-memory clipboard implementation for testing and standalone mode.
 * Stores a single string value as the "clipboard" contents.
 */
export class MockClipboardBridge implements ClipboardBridge {
  private clipboardContent = '';

  async read(): Promise<{ content: string; type: string }> {
    return { content: this.clipboardContent, type: 'text/plain' };
  }

  async write(content: string): Promise<boolean> {
    this.clipboardContent = content;
    return true;
  }
}

// ── Shell Bridge (macOS) ─────────────────────────────────────────────────────

/**
 * OS clipboard access via shell commands.
 *
 * macOS: uses `pbpaste` (read) and `pbcopy` (write).
 * Falls back to MockClipboardBridge on unsupported platforms.
 */
export class ShellBridge implements ClipboardBridge {
  async read(): Promise<{ content: string; type: string }> {
    const { execSync } = await import('node:child_process');
    try {
      const content = execSync('pbpaste', {
        encoding: 'utf-8',
        timeout: 3000,
        maxBuffer: 1024 * 1024, // 1 MB max clipboard size
      });
      return { content, type: 'text/plain' };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`Failed to read clipboard via pbpaste: ${msg}`);
    }
  }

  async write(content: string): Promise<boolean> {
    const { execSync } = await import('node:child_process');
    try {
      execSync('pbcopy', {
        input: content,
        encoding: 'utf-8',
        timeout: 3000,
      });
      return true;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`Failed to write clipboard via pbcopy: ${msg}`);
    }
  }
}

// ── Tauri Bridge (placeholder) ───────────────────────────────────────────────

/**
 * Placeholder for the real Tauri IPC bridge.
 * Throws when used outside of the Tauri runtime.
 */
export class TauriBridge implements ClipboardBridge {
  async read(): Promise<{ content: string; type: string }> {
    throw new Error('Clipboard bridge not available outside Tauri runtime');
  }

  async write(_content: string): Promise<boolean> {
    throw new Error('Clipboard bridge not available outside Tauri runtime');
  }
}

// ── Bridge Accessor ──────────────────────────────────────────────────────────

/** Create the default bridge based on platform. */
function createDefaultBridge(): ClipboardBridge {
  if (process.platform === 'darwin') {
    return new ShellBridge();
  }
  // Non-macOS: fall back to in-memory mock (cross-platform support is later)
  return new MockClipboardBridge();
}

let currentBridge: ClipboardBridge = createDefaultBridge();

/** Get the active clipboard bridge instance. */
export function getBridge(): ClipboardBridge {
  return currentBridge;
}

/** Set the clipboard bridge (used for testing / DI). */
export function setBridge(bridge: ClipboardBridge): void {
  currentBridge = bridge;
}
