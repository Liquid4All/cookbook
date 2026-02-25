"""
Heuristic action suggestion engine for the screenshot-to-action pipeline.

Classifies extracted text and UI elements into actionable suggestions
using keyword/regex matching. Each rule maps a detected pattern to
a specific MCP tool chain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline_types import ActionSuggestion, UIElement


# ---- Pattern Rule Definitions ------------------------------------------------


@dataclass(frozen=True)
class PatternRule:
    """A single heuristic rule mapping a regex pattern to an action suggestion."""

    name: str
    pattern: re.Pattern[str]
    action: str
    description: str
    confidence: float
    tool_chain: list[str]


# Email addresses: user@domain.tld
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# File paths: /absolute/path or ~/home-relative or C:\windows\path
_FILE_PATH_PATTERN = re.compile(
    r"(?:(?:/[A-Za-z0-9._-]+){2,}|~/[A-Za-z0-9._/-]+|[A-Z]:\\[A-Za-z0-9._\\-]+)"
)

# Dates and times: YYYY-MM-DD, MM/DD/YYYY, HH:MM, "Jan 15", "March 3rd"
_DATE_TIME_PATTERN = re.compile(
    r"\b(?:"
    r"\d{4}-\d{2}-\d{2}"             # ISO date
    r"|\d{1,2}/\d{1,2}/\d{2,4}"     # US date
    r"|\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?"  # Time
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}"  # Month Day
    r")\b"
)

# TODO/task patterns: "TODO", "[ ]", "[x]", "Action item:", "FIXME"
_TODO_PATTERN = re.compile(
    r"(?i)\b(?:TODO|FIXME|HACK|ACTION\s+ITEM|TASK)\b"
    r"|(?:\[[ xX]\])"
)

# URLs: http(s)://... or www....
_URL_PATTERN = re.compile(
    r"\bhttps?://[^\s<>\"']+|www\.[^\s<>\"']+"
)

# Table-like data: lines with multiple delimiter-separated columns
_TABLE_PATTERN = re.compile(
    r"(?:^|\n)(?:[^\n|,\t]*[|,\t]){2,}[^\n]*(?:\n|$)"
)


# Ordered list of rules — checked in priority order
PATTERN_RULES: list[PatternRule] = [
    PatternRule(
        name="email",
        pattern=_EMAIL_PATTERN,
        action="Draft reply email",
        description="Email addresses detected — compose a reply or new email",
        confidence=0.85,
        tool_chain=["email.draft_email"],
    ),
    PatternRule(
        name="file_path",
        pattern=_FILE_PATH_PATTERN,
        action="Open file",
        description="File paths detected — open the referenced file",
        confidence=0.80,
        tool_chain=["system.open_file_with"],
    ),
    PatternRule(
        name="date_time",
        pattern=_DATE_TIME_PATTERN,
        action="Create calendar event",
        description="Dates or times detected — create a calendar event",
        confidence=0.75,
        tool_chain=["calendar.create_event"],
    ),
    PatternRule(
        name="todo",
        pattern=_TODO_PATTERN,
        action="Create task",
        description="Task or TODO items detected — add to task list",
        confidence=0.80,
        tool_chain=["task.create_task"],
    ),
    PatternRule(
        name="url",
        pattern=_URL_PATTERN,
        action="Open URL",
        description="URLs detected — open in default browser",
        confidence=0.85,
        tool_chain=["system.open_application"],
    ),
    PatternRule(
        name="table",
        pattern=_TABLE_PATTERN,
        action="Extract to spreadsheet",
        description="Tabular data detected — export to CSV spreadsheet",
        confidence=0.70,
        tool_chain=["data.write_csv"],
    ),
]


# ---- Classification Functions ------------------------------------------------


def classify_text(text: str) -> list[ActionSuggestion]:
    """
    Analyze extracted text and return action suggestions.

    Applies each pattern rule against the text. If a pattern matches,
    the corresponding action is added to the suggestions list.
    Suggestions are returned sorted by confidence (descending).

    Args:
        text: The extracted text from a screenshot or OCR result.

    Returns:
        List of ActionSuggestion objects sorted by confidence.
    """
    if not text or not text.strip():
        return []

    suggestions: list[ActionSuggestion] = []
    seen_actions: set[str] = set()

    for rule in PATTERN_RULES:
        if rule.pattern.search(text) and rule.action not in seen_actions:
            suggestions.append(
                ActionSuggestion(
                    action=rule.action,
                    description=rule.description,
                    confidence=rule.confidence,
                    tool_chain=rule.tool_chain,
                )
            )
            seen_actions.add(rule.action)

    # Sort by confidence descending
    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    return suggestions


def classify_with_elements(
    text: str,
    elements: list[UIElement] | None,
) -> list[ActionSuggestion]:
    """
    Analyze text and optional UI elements, returning action suggestions.

    Combines text-based classification with element-based heuristics.
    If UI elements contain interactive elements (buttons, text fields),
    additional suggestions may be generated.

    Args:
        text: The extracted text from a screenshot.
        elements: Optional list of detected UI elements.

    Returns:
        List of ActionSuggestion objects sorted by confidence.
    """
    suggestions = classify_text(text)
    seen_actions = {s.action for s in suggestions}

    if elements:
        # Gather all element text for secondary pattern analysis
        element_text = " ".join(el.text for el in elements if el.text)
        for rule in PATTERN_RULES:
            if rule.action not in seen_actions and rule.pattern.search(element_text):
                suggestions.append(
                    ActionSuggestion(
                        action=rule.action,
                        description=rule.description,
                        confidence=rule.confidence * 0.9,  # slightly lower for element-only
                        tool_chain=rule.tool_chain,
                    )
                )
                seen_actions.add(rule.action)

    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    return suggestions
