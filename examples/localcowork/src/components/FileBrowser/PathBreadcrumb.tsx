/**
 * PathBreadcrumb — clickable breadcrumb showing the current directory path.
 *
 * Splits the full path into segments. Clicking a segment navigates
 * to that directory. Supports both Unix (/) and Windows (\) paths.
 */

import { useCallback, useMemo } from "react";

import {
  splitPathComponents,
  detectHomeDir,
  detectSeparator,
} from "../../utils/pathUtils";

interface PathBreadcrumbProps {
  readonly path: string;
  readonly onNavigate: (path: string) => void;
}

interface PathSegment {
  readonly label: string;
  readonly fullPath: string;
}

/** Split a path into navigable segments (cross-platform). */
function splitPath(path: string): readonly PathSegment[] {
  const sep = detectSeparator(path);
  const homePlaceholder = "~";
  let displayPath = path;
  const homeDir = detectHomeDir(path);
  if (homeDir != null) {
    displayPath = path.replace(homeDir, homePlaceholder);
  }

  const parts = splitPathComponents(displayPath);
  const segments: PathSegment[] = [];

  for (let i = 0; i < parts.length; i++) {
    const label = parts[i];
    // Rebuild the full actual path for navigation
    if (label === homePlaceholder && homeDir != null) {
      segments.push({ label: "~", fullPath: homeDir });
    } else {
      const prevPath = i > 0 ? segments[i - 1].fullPath : "";
      segments.push({
        label,
        fullPath: `${prevPath}${sep}${label}`,
      });
    }
  }

  return segments;
}

export function PathBreadcrumb({
  path,
  onNavigate,
}: PathBreadcrumbProps): React.JSX.Element {
  const segments = useMemo(() => splitPath(path), [path]);
  const sep = useMemo(() => detectSeparator(path), [path]);

  const handleClick = useCallback(
    (fullPath: string) => () => {
      onNavigate(fullPath);
    },
    [onNavigate],
  );

  return (
    <div className="path-breadcrumb">
      {segments.map((seg, index) => (
        <span key={seg.fullPath} className="breadcrumb-segment">
          {index > 0 && <span className="breadcrumb-separator">{sep}</span>}
          <button
            className="breadcrumb-btn"
            onClick={handleClick(seg.fullPath)}
            type="button"
            title={seg.fullPath}
          >
            {seg.label}
          </button>
        </span>
      ))}
    </div>
  );
}
