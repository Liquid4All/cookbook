/**
 * ContextIndicator — shows remaining context window budget.
 *
 * Renders a compact progress bar with token counts. Color changes
 * from green → yellow → red as the budget fills up.
 */

import type { ContextBudget } from "../../types";

interface ContextIndicatorProps {
  readonly budget: ContextBudget | null;
}

/** Percentage thresholds for color changes. */
const YELLOW_THRESHOLD = 60;
const RED_THRESHOLD = 85;

function getBarColor(usedPercent: number): string {
  if (usedPercent >= RED_THRESHOLD) return "context-bar-red";
  if (usedPercent >= YELLOW_THRESHOLD) return "context-bar-yellow";
  return "context-bar-green";
}

function formatTokens(count: number): string {
  if (count >= 1000) {
    return `${(count / 1000).toFixed(1)}k`;
  }
  return String(count);
}

export function ContextIndicator({
  budget,
}: ContextIndicatorProps): React.JSX.Element | null {
  if (!budget) return null;

  const used = budget.total - budget.remaining;
  const usedPercent = Math.min(100, (used / budget.total) * 100);
  const barClass = getBarColor(usedPercent);

  return (
    <div className="context-indicator" title="Context window usage">
      <div className="context-indicator-bar">
        <div
          className={`context-indicator-fill ${barClass}`}
          style={{ width: `${usedPercent}%` }}
        />
      </div>
      <span className="context-indicator-label">
        {formatTokens(used)} / {formatTokens(budget.total)} tokens
      </span>
    </div>
  );
}
