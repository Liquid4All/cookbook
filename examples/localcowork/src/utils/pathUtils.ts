/**
 * Cross-platform path utilities for the frontend.
 *
 * Handles both Unix (/) and Windows (\) path separators,
 * home directory detection on both platforms, and
 * platform-correct path joining.
 */

/** Regex matching the home directory on macOS/Linux or Windows. */
const HOME_DIR_PATTERN = /^(?:\/Users\/[^/]+|\/home\/[^/]+|[A-Z]:\\Users\\[^\\]+)/;

/** Split a path into components, handling both / and \ separators. */
export function splitPathComponents(p: string): string[] {
  return p.split(/[/\\]/).filter((s) => s.length > 0);
}

/** Detect the home directory prefix from an absolute path. */
export function detectHomeDir(p: string): string | null {
  const match = p.match(HOME_DIR_PATTERN);
  return match?.[0] ?? null;
}

/** Get the parent path, handling both separators. */
export function getParentPath(p: string): string {
  // Find the last separator (either / or \)
  const lastSlash = p.lastIndexOf("/");
  const lastBackslash = p.lastIndexOf("\\");
  const lastSep = Math.max(lastSlash, lastBackslash);

  if (lastSep <= 0) {
    return p;
  }

  return p.slice(0, lastSep);
}

/** Detect which separator a path uses. */
export function detectSeparator(p: string): string {
  // If the path contains backslashes, it's a Windows path
  if (p.includes("\\")) {
    return "\\";
  }
  return "/";
}

/** Join two path segments using the separator from the base path. */
export function joinPath(base: string, segment: string): string {
  const sep = detectSeparator(base);
  const trimmedBase = base.replace(/[/\\]$/, "");
  return `${trimmedBase}${sep}${segment}`;
}
