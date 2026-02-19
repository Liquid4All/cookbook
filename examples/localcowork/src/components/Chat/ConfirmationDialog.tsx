/**
 * ConfirmationDialog — modal for user confirmation of mutable/destructive actions.
 *
 * Three-tier permission model:
 * - Cancel (reject)
 * - Allow Once (default)
 * - Allow for Session (skip confirmation for this tool until session ends)
 * - Always allow (small text link — persistent, discourage overuse)
 */

import type { ConfirmationRequest, ConfirmationResponse } from "../../types";

interface ConfirmationDialogProps {
  readonly request: ConfirmationRequest;
  readonly onRespond: (response: ConfirmationResponse) => void;
}

export function ConfirmationDialog({
  request,
  onRespond,
}: ConfirmationDialogProps): React.JSX.Element {
  const toolShortName = request.toolName.split(".").pop() ?? request.toolName;

  return (
    <div className="confirmation-overlay">
      <div className="confirmation-dialog">
        <div className="confirmation-header">
          {request.isDestructive ? (
            <span className="confirmation-icon destructive">&#9888;</span>
          ) : (
            <span className="confirmation-icon mutable">&#128736;</span>
          )}
          <h3 className="confirmation-title">
            {request.isDestructive ? "Destructive Action" : "Confirm Action"}
          </h3>
        </div>

        <div className="confirmation-body">
          <div className="confirmation-tool">
            <span className="confirmation-label">Tool:</span>
            <code className="confirmation-tool-name">{toolShortName}</code>
          </div>

          <div className="confirmation-preview">
            <span className="confirmation-label">Action:</span>
            <p className="confirmation-preview-text">{request.preview}</p>
          </div>

          {request.undoSupported && (
            <p className="confirmation-undo-hint">
              This action can be undone.
            </p>
          )}
        </div>

        <div className="confirmation-actions">
          <button
            className="confirm-btn cancel"
            onClick={() => onRespond({ type: "rejected" })}
            type="button"
          >
            Cancel
          </button>
          <button
            className={`confirm-btn ${request.isDestructive ? "destructive" : "confirm"}`}
            onClick={() => onRespond({ type: "confirmed" })}
            type="button"
          >
            {request.isDestructive ? "Delete" : "Allow Once"}
          </button>
          {!request.isDestructive && (
            <button
              className="confirm-btn session"
              onClick={() => onRespond({ type: "confirmedForSession" })}
              type="button"
            >
              Allow for Session
            </button>
          )}
        </div>

        {!request.isDestructive && (
          <div className="confirmation-always">
            <button
              className="confirm-link-btn"
              onClick={() => onRespond({ type: "confirmedAlways" })}
              type="button"
            >
              Always allow this tool
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
