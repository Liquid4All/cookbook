/**
 * ToolTrace — real-time tool execution visualizer.
 *
 * Displays tool calls and their results in an expandable tree view.
 * Correlates ToolCall IDs with ToolResult messages to show execution
 * status: pending, running, complete, or error.
 */

import { useCallback, useMemo, useState } from "react";

import type { ChatMessage, ToolCall } from "../../types";

/** Status of a single tool execution step. */
type TraceStepStatus = "pending" | "executing" | "complete" | "error";

/** A single step in the tool execution trace. */
interface TraceStep {
  readonly id: string;
  readonly toolName: string;
  readonly arguments: Record<string, unknown>;
  readonly status: TraceStepStatus;
  readonly result?: unknown;
  readonly error?: string;
  readonly executionTimeMs?: number;
}

interface ToolTraceProps {
  /** Tool calls from the assistant message. */
  readonly toolCalls: readonly ToolCall[];
  /** All messages in the conversation (for result correlation). */
  readonly allMessages: readonly ChatMessage[];
  /** Whether the assistant is still generating (tools may be pending). */
  readonly isExecuting: boolean;
}

/** Format a duration in milliseconds for display. */
function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
  return `${(ms / 1000).toFixed(1)}s`;
}

/** Truncate a string to a max length, appending ellipsis if needed. */
function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength)}...`;
}

/** Abbreviate tool arguments for the collapsed summary. */
function abbreviateArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args);
  if (entries.length === 0) {
    return "";
  }
  const parts = entries.slice(0, 2).map(([key, val]) => {
    const strVal = typeof val === "string" ? val : JSON.stringify(val);
    return `${key}: ${truncate(strVal, 30)}`;
  });
  const suffix = entries.length > 2 ? ", ..." : "";
  return `(${parts.join(", ")}${suffix})`;
}

/** Format a tool result for preview display. */
function formatResultPreview(result: unknown): string {
  if (result == null) {
    return "No output";
  }
  if (typeof result === "string") {
    return truncate(result, 200);
  }
  const json = JSON.stringify(result, null, 2);
  return truncate(json, 200);
}

/** Correlate tool calls with their results from subsequent messages. */
function buildTraceSteps(
  toolCalls: readonly ToolCall[],
  allMessages: readonly ChatMessage[],
  isExecuting: boolean,
): readonly TraceStep[] {
  // Build a map of toolCallId → tool result message
  const resultMap = new Map<
    string,
    { result: unknown; error?: string; executionTimeMs?: number }
  >();

  for (const msg of allMessages) {
    if (msg.role === "tool" && msg.toolCallId != null) {
      // Parse tool result to extract success/error and timing
      const toolResult = msg.toolResult as
        | {
            success?: boolean;
            result?: unknown;
            error?: string;
            executionTimeMs?: number;
          }
        | string
        | undefined;

      if (typeof toolResult === "object" && toolResult != null) {
        resultMap.set(msg.toolCallId, {
          result: toolResult.result ?? toolResult,
          error: toolResult.error,
          executionTimeMs: toolResult.executionTimeMs,
        });
      } else {
        resultMap.set(msg.toolCallId, {
          result: toolResult,
        });
      }
    }
  }

  return toolCalls.map((tc) => {
    const resultEntry = resultMap.get(tc.id);
    if (resultEntry != null) {
      const hasError =
        resultEntry.error != null && resultEntry.error.length > 0;
      return {
        id: tc.id,
        toolName: tc.name,
        arguments: tc.arguments,
        status: hasError ? ("error" as const) : ("complete" as const),
        result: resultEntry.result,
        error: resultEntry.error,
        executionTimeMs: resultEntry.executionTimeMs,
      };
    }
    // No result yet — either pending or executing
    return {
      id: tc.id,
      toolName: tc.name,
      arguments: tc.arguments,
      status: isExecuting ? ("executing" as const) : ("pending" as const),
    };
  });
}

/** Status icon for a trace step. */
function StatusIcon({
  status,
}: {
  readonly status: TraceStepStatus;
}): React.JSX.Element {
  switch (status) {
    case "pending":
      return <span className="trace-status trace-status-pending">&#9711;</span>;
    case "executing":
      return (
        <span className="trace-status trace-status-executing">&#8635;</span>
      );
    case "complete":
      return (
        <span className="trace-status trace-status-complete">&#10003;</span>
      );
    case "error":
      return <span className="trace-status trace-status-error">&#10007;</span>;
  }
}

/** A single step row in the trace tree. */
function TraceStepRow({
  step,
  isLast,
}: {
  readonly step: TraceStep;
  readonly isLast: boolean;
}): React.JSX.Element {
  const [showFullResult, setShowFullResult] = useState(false);

  const resultPreview =
    step.result != null ? formatResultPreview(step.result) : null;
  const fullResult =
    step.result != null
      ? typeof step.result === "string"
        ? step.result
        : JSON.stringify(step.result, null, 2)
      : null;
  const isResultTruncated =
    fullResult != null && resultPreview != null && fullResult !== resultPreview;

  const toggleResult = useCallback(() => {
    setShowFullResult((prev) => !prev);
  }, []);

  return (
    <div className="trace-step">
      <div className="trace-step-connector">
        <span className="trace-connector-line">
          {isLast ? "\u2514" : "\u251C"}
        </span>
      </div>
      <div className="trace-step-content">
        <div className="trace-step-header">
          <StatusIcon status={step.status} />
          <span className="trace-tool-name">{step.toolName}</span>
          <span className="trace-tool-args">
            {abbreviateArgs(step.arguments)}
          </span>
          {step.executionTimeMs != null && (
            <span className="trace-time">
              {formatDuration(step.executionTimeMs)}
            </span>
          )}
        </div>

        {/* Result preview */}
        {step.status === "complete" && resultPreview != null && (
          <div className="trace-result">
            <span className="trace-result-arrow">&rarr;</span>
            <pre className="trace-result-text">
              {showFullResult ? fullResult : resultPreview}
            </pre>
            {isResultTruncated && (
              <button
                className="trace-show-more"
                onClick={toggleResult}
                type="button"
              >
                {showFullResult ? "Show less" : "Show more"}
              </button>
            )}
          </div>
        )}

        {/* Error display */}
        {step.status === "error" && step.error != null && (
          <div className="trace-error">
            <span className="trace-error-icon">&#9888;</span>
            <span className="trace-error-text">{step.error}</span>
          </div>
        )}

        {/* Executing indicator */}
        {step.status === "executing" && (
          <div className="trace-executing">
            <span className="trace-executing-text">Executing...</span>
          </div>
        )}
      </div>
    </div>
  );
}

export function ToolTrace({
  toolCalls,
  allMessages,
  isExecuting,
}: ToolTraceProps): React.JSX.Element {
  const [isExpanded, setIsExpanded] = useState(false);

  const steps = useMemo(
    () => buildTraceSteps(toolCalls, allMessages, isExecuting),
    [toolCalls, allMessages, isExecuting],
  );

  const completedCount = steps.filter(
    (s) => s.status === "complete" || s.status === "error",
  ).length;
  const totalCount = steps.length;
  const totalTimeMs = steps.reduce(
    (sum, s) => sum + (s.executionTimeMs ?? 0),
    0,
  );
  const hasErrors = steps.some((s) => s.status === "error");

  const toggleExpand = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  return (
    <div className="tool-trace-container">
      {/* Header / collapsed summary */}
      <button
        className="tool-trace-header"
        onClick={toggleExpand}
        type="button"
        aria-expanded={isExpanded}
      >
        <span className="tool-trace-toggle">
          {isExpanded ? "\u25BC" : "\u25B6"}
        </span>
        <span className="tool-trace-icon">&#9881;</span>
        <span className="tool-trace-summary">
          {completedCount === totalCount
            ? `${totalCount} tool${totalCount !== 1 ? "s" : ""} executed`
            : `${completedCount}/${totalCount} tools completed`}
          {totalTimeMs > 0 && ` in ${formatDuration(totalTimeMs)}`}
        </span>
        {hasErrors && (
          <span className="tool-trace-error-badge">errors</span>
        )}
      </button>

      {/* Expanded tree */}
      {isExpanded && (
        <div className="tool-trace-tree" role="list">
          {steps.map((step, index) => (
            <TraceStepRow
              key={step.id}
              step={step}
              isLast={index === steps.length - 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}
