/**
 * PermissionsTab — manage persistent "Always Allow" permission grants.
 *
 * Lists all tools with permanent permission grants and allows
 * the user to revoke individual grants.
 */

import { useCallback } from "react";

import type { PermissionGrant } from "../../types";

interface PermissionsTabProps {
  readonly grants: readonly PermissionGrant[];
  readonly onRevoke: (toolName: string) => void;
}

/** Human-friendly tool name from a fully-qualified name. */
function formatToolName(name: string): string {
  const parts = name.split(".");
  if (parts.length === 2) {
    const [server, tool] = parts;
    return `${tool} (${server})`;
  }
  return name;
}

/** Format an ISO date string to a human-readable form. */
function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return iso;
  }
}

export function PermissionsTab({
  grants,
  onRevoke,
}: PermissionsTabProps): React.JSX.Element {
  const handleRevoke = useCallback(
    (toolName: string) => {
      onRevoke(toolName);
    },
    [onRevoke],
  );

  if (grants.length === 0) {
    return (
      <div className="permissions-empty">
        <p className="permissions-empty-text">
          No permanent permissions granted.
        </p>
        <p className="permissions-empty-hint">
          When you choose &ldquo;Always allow&rdquo; on a confirmation dialog,
          the tool will appear here.
        </p>
      </div>
    );
  }

  return (
    <div className="permissions-tab">
      <p className="permissions-description">
        Tools with permanent &ldquo;Always Allow&rdquo; grants skip the
        confirmation dialog. Revoke to require confirmation again.
      </p>
      <ul className="permissions-list">
        {grants.map((grant) => (
          <li key={grant.toolName} className="permissions-item">
            <div className="permissions-item-info">
              <span className="permissions-item-name">
                {formatToolName(grant.toolName)}
              </span>
              <span className="permissions-item-date">
                Granted {formatDate(grant.grantedAt)}
              </span>
            </div>
            <button
              className="permissions-revoke-btn"
              onClick={() => handleRevoke(grant.toolName)}
              type="button"
            >
              Revoke
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
