/**
 * Shared Validation Utilities — TypeScript
 *
 * Path sandboxing, input sanitization, and common validators
 * used across all TypeScript MCP servers.
 */

import * as path from 'path';

// ─── Sandbox Validation ─────────────────────────────────────────────────────

/** Allowed base paths — set from user configuration at startup */
let allowedPaths: string[] = [];

/** Initialize sandbox with user-granted directories */
export function initSandbox(paths: string[]): void {
  allowedPaths = paths.map((p) => path.resolve(p));
}

/**
 * Assert that a path is within the sandboxed directories.
 * Throws MCPError with SANDBOX_VIOLATION code if not.
 */
export function assertSandboxed(targetPath: string): void {
  const resolved = path.resolve(targetPath);

  // Check against each allowed path
  const isAllowed = allowedPaths.some((allowed) => {
    return resolved === allowed || resolved.startsWith(allowed + path.sep);
  });

  if (!isAllowed) {
    throw new SandboxViolationError(
      `Path "${resolved}" is outside the sandboxed directories. ` +
        `Allowed: ${allowedPaths.join(', ')}`,
    );
  }
}

export class SandboxViolationError extends Error {
  code = -32001;
  constructor(message: string) {
    super(message);
    this.name = 'SandboxViolationError';
  }
}

// ─── Input Sanitization ─────────────────────────────────────────────────────

/** Sanitize a filename to prevent path traversal */
export function sanitizeFilename(filename: string): string {
  return filename
    .replace(/\.\./g, '')      // Remove path traversal
    .replace(/[<>:"|?*]/g, '') // Remove invalid filename chars (Windows)
    .replace(/\0/g, '')        // Remove null bytes
    .trim();
}

/** Validate that a string is a valid absolute path */
export function isAbsolutePath(p: string): boolean {
  return path.isAbsolute(p);
}

/** Assert a path is absolute, throw if not */
export function assertAbsolutePath(p: string, paramName: string): void {
  if (!isAbsolutePath(p)) {
    throw new Error(`Parameter "${paramName}" must be an absolute path. Got: "${p}"`);
  }
}

// ─── Type Guards ────────────────────────────────────────────────────────────

/** Check if a value is a non-empty string */
export function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

/** Check if a value is a positive integer */
export function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0;
}

// ─── File Type Helpers ──────────────────────────────────────────────────────

/** Common file type categories */
export const FILE_CATEGORIES = {
  document: ['.pdf', '.docx', '.doc', '.txt', '.md', '.rtf', '.odt', '.html'],
  spreadsheet: ['.xlsx', '.xls', '.csv', '.tsv', '.ods'],
  image: ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'],
  audio: ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma'],
  video: ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
  archive: ['.zip', '.tar', '.gz', '.rar', '.7z', '.bz2'],
  code: ['.ts', '.js', '.py', '.rs', '.go', '.java', '.c', '.cpp', '.h'],
} as const;

/** Get the category of a file by its extension */
export function getFileCategory(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  for (const [category, extensions] of Object.entries(FILE_CATEGORIES)) {
    if ((extensions as readonly string[]).includes(ext)) {
      return category;
    }
  }
  return 'other';
}
