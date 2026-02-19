"""
Heuristic-based extraction engine for meeting transcripts.

Provides regex/pattern-based extraction of action items, commitments,
decisions, and open questions from transcript text. Designed to be
swapped for LLM-based extraction once the inference layer is integrated.

Functions:
    extract_action_items_from_text  — find action items with assignees and deadlines
    extract_commitments_from_text   — find commitments, decisions, open questions
    generate_minutes_text           — produce formatted meeting minutes markdown
"""

from __future__ import annotations

import re
from typing import Sequence

from meeting_types import ActionItem, Commitment, Decision


# ─── Patterns ─────────────────────────────────────────────────────────────────

# Action item markers (explicit labels in transcript)
_ACTION_MARKERS: re.Pattern[str] = re.compile(
    r"(?:ACTION|TODO|Action item|AI|\[AI\])\s*[:]\s*(.+)",
    re.IGNORECASE,
)

# Phrases that signal an action assignment
_ACTION_PHRASES: re.Pattern[str] = re.compile(
    r"(?:will\s+(?:do|handle|take care of|finish|complete|prepare|send|review|"
    r"create|write|update|fix|set up|follow up|schedule|draft|submit|implement|"
    r"check|investigate|look into|work on|coordinate))"
    r"|(?:needs?\s+to\b)"
    r"|(?:should\b)"
    r"|(?:has\s+to\b)"
    r"|(?:is\s+responsible\s+for\b)"
    r"|(?:take\s+care\s+of\b)",
    re.IGNORECASE,
)

# @person pattern (e.g., "@John will review the proposal")
_AT_PERSON: re.Pattern[str] = re.compile(r"@(\w+)")

# Speaker label pattern (e.g., "John:", "Speaker_1:")
_SPEAKER_LABEL: re.Pattern[str] = re.compile(r"^([A-Z][A-Za-z0-9_]+)\s*:")

# Deadline patterns
_DEADLINE: re.Pattern[str] = re.compile(
    r"(?:by|before|due|until|deadline)\s+([A-Za-z0-9, ]+?)(?:\.|$|;|\n)",
    re.IGNORECASE,
)

# Priority indicators
_HIGH_PRIORITY: re.Pattern[str] = re.compile(
    r"\b(?:urgent(?:ly)?|ASAP|critical(?:ly)?|immediately|top priority|high priority)\b",
    re.IGNORECASE,
)

_LOW_PRIORITY: re.Pattern[str] = re.compile(
    r"\b(?:low priority|when possible|nice to have|eventually|if time permits)\b",
    re.IGNORECASE,
)

# Commitment patterns ("I will...", "I'll...", "I commit to...")
_COMMITMENT: re.Pattern[str] = re.compile(
    r"(I\s+will|I'll|I\s+commit\s+to|I\s+promise\s+to|I\s+guarantee)\s+(.+?)(?:\.|$)",
    re.IGNORECASE,
)

# Decision patterns ("We decided...", "Decision:...", "Agreed:...")
_DECISION_MARKERS: re.Pattern[str] = re.compile(
    r"(?:We\s+decided|Decision\s*:|Agreed\s*:|The\s+decision\s+is|"
    r"Let's\s+go\s+with|We(?:'re| are)\s+going\s+(?:to|with))\s*(.+?)(?:\.|$)",
    re.IGNORECASE,
)

# Open question patterns
_QUESTION_MARKERS: re.Pattern[str] = re.compile(
    r"(?:TBD|to be determined|open question|need to figure out|"
    r"still need to decide|remains to be seen|unclear|"
    r"we need to discuss|pending decision)\b",
    re.IGNORECASE,
)


# ─── Action Item Extraction ──────────────────────────────────────────────────


def extract_action_items_from_text(transcript: str) -> list[ActionItem]:
    """
    Extract action items from a meeting transcript using heuristic patterns.

    Looks for explicit markers (ACTION:, TODO:) and implicit phrases
    ("will do", "needs to", etc.). Extracts assignee from speaker labels
    or @mentions, and deadlines from "by/before/due" phrases.

    Args:
        transcript: The full meeting transcript text.

    Returns:
        A list of ActionItem objects found in the transcript.
    """
    if not transcript.strip():
        return []

    items: list[ActionItem] = []
    lines = transcript.splitlines()
    current_speaker = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Track current speaker from labels like "John:"
        speaker_match = _SPEAKER_LABEL.match(stripped)
        if speaker_match:
            current_speaker = speaker_match.group(1)

        # Check for explicit action markers
        marker_match = _ACTION_MARKERS.search(stripped)
        if marker_match:
            task_text = marker_match.group(1).strip()
            assignee = _extract_assignee(task_text, current_speaker)
            deadline = _extract_deadline(task_text)
            priority = _detect_priority(task_text)
            context = _build_context(lines, i)
            items.append(ActionItem(
                assignee=assignee,
                task=task_text,
                deadline=deadline,
                context=context,
                priority=priority,
            ))
            continue

        # Check for action phrases ("will do", "needs to", etc.)
        if _ACTION_PHRASES.search(stripped):
            assignee = _extract_assignee(stripped, current_speaker)
            task_text = stripped
            # Remove speaker label prefix from task text
            if speaker_match:
                task_text = stripped[speaker_match.end():].strip()
            deadline = _extract_deadline(stripped)
            priority = _detect_priority(stripped)
            context = _build_context(lines, i)
            items.append(ActionItem(
                assignee=assignee,
                task=task_text,
                deadline=deadline,
                context=context,
                priority=priority,
            ))

    return items


# ─── Commitments / Decisions / Questions Extraction ───────────────────────────


def extract_commitments_from_text(
    transcript: str,
) -> tuple[list[Commitment], list[Decision], list[str]]:
    """
    Extract commitments, decisions, and open questions from a transcript.

    Commitments: "I will...", "I'll...", "I commit to..."
    Decisions: "We decided...", "Decision:...", "Agreed:..."
    Open questions: lines containing "?" or markers like "TBD"

    Args:
        transcript: The full meeting transcript text.

    Returns:
        A tuple of (commitments, decisions, open_questions).
    """
    if not transcript.strip():
        return [], [], []

    commitments: list[Commitment] = []
    decisions: list[Decision] = []
    open_questions: list[str] = []

    lines = transcript.splitlines()
    current_speaker = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Track current speaker
        speaker_match = _SPEAKER_LABEL.match(stripped)
        if speaker_match:
            current_speaker = speaker_match.group(1)

        # Check for commitments ("I will...", "I'll...")
        commitment_match = _COMMITMENT.search(stripped)
        if commitment_match:
            commitment_text = commitment_match.group(2).strip()
            deadline = _extract_deadline(stripped)
            context = _build_context(lines, i)
            commitments.append(Commitment(
                person=current_speaker or "Unknown",
                commitment=commitment_text,
                deadline=deadline,
                context=context,
            ))

        # Check for decisions ("We decided...", "Agreed:...")
        decision_match = _DECISION_MARKERS.search(stripped)
        if decision_match:
            decision_text = decision_match.group(1).strip()
            context = _build_context(lines, i)
            decisions.append(Decision(
                decision=decision_text,
                made_by=current_speaker or "Group",
                context=context,
            ))

        # Check for open questions
        if "?" in stripped or _QUESTION_MARKERS.search(stripped):
            # Remove speaker label for cleaner question text
            question_text = stripped
            if speaker_match:
                question_text = stripped[speaker_match.end():].strip()
            if question_text and question_text not in open_questions:
                open_questions.append(question_text)

    return commitments, decisions, open_questions


# ─── Minutes Generation ──────────────────────────────────────────────────────


def generate_minutes_text(transcript: str, template: str | None = None) -> str:
    """
    Generate formatted meeting minutes markdown from a transcript.

    Extracts attendees from speaker labels, splits transcript into
    discussion sections, and includes action items and decisions.

    Args:
        transcript: The full meeting transcript text.
        template: Optional template name (reserved for future use).

    Returns:
        A markdown-formatted meeting minutes string.
    """
    if not transcript.strip():
        return "## Meeting Minutes\n\nNo transcript content provided.\n"

    attendees = _extract_attendees(transcript)
    action_items = extract_action_items_from_text(transcript)
    commitments, decisions, open_questions = extract_commitments_from_text(transcript)
    discussion_sections = _split_into_sections(transcript)

    parts: list[str] = []
    parts.append("## Meeting Minutes\n")

    # Attendees
    parts.append("### Attendees\n")
    if attendees:
        for attendee in sorted(attendees):
            parts.append(f"- {attendee}")
    else:
        parts.append("- (no speakers identified)")
    parts.append("")

    # Discussion
    parts.append("### Discussion\n")
    for idx, section in enumerate(discussion_sections, start=1):
        parts.append(f"**Topic {idx}**\n")
        parts.append(section.strip())
        parts.append("")

    # Action Items
    parts.append("### Action Items\n")
    if action_items:
        for item in action_items:
            deadline_str = f" (due: {item.deadline})" if item.deadline else ""
            priority_str = f" [{item.priority}]" if item.priority != "medium" else ""
            parts.append(
                f"- **{item.assignee}**: {item.task}{deadline_str}{priority_str}"
            )
    else:
        parts.append("- (none identified)")
    parts.append("")

    # Decisions
    parts.append("### Decisions\n")
    if decisions:
        for dec in decisions:
            by_str = f" ({dec.made_by})" if dec.made_by else ""
            parts.append(f"- {dec.decision}{by_str}")
    else:
        parts.append("- (none identified)")
    parts.append("")

    # Open Questions
    if open_questions:
        parts.append("### Open Questions\n")
        for q in open_questions:
            parts.append(f"- {q}")
        parts.append("")

    return "\n".join(parts)


# ─── Helper Functions ─────────────────────────────────────────────────────────


def _extract_assignee(text: str, current_speaker: str) -> str:
    """Extract the assignee from @mentions or fall back to current speaker."""
    at_match = _AT_PERSON.search(text)
    if at_match:
        return at_match.group(1)
    return current_speaker or "Unassigned"


def _extract_deadline(text: str) -> str:
    """Extract a deadline phrase from the text."""
    match = _DEADLINE.search(text)
    if match:
        return match.group(1).strip()
    return ""


def _detect_priority(text: str) -> str:
    """Detect priority level from keywords in the text."""
    if _HIGH_PRIORITY.search(text):
        return "high"
    if _LOW_PRIORITY.search(text):
        return "low"
    return "medium"


def _build_context(lines: Sequence[str], index: int) -> str:
    """Build a context string from surrounding lines (1 line before/after)."""
    start = max(0, index - 1)
    end = min(len(lines), index + 2)
    return " ".join(line.strip() for line in lines[start:end] if line.strip())


def _extract_attendees(transcript: str) -> list[str]:
    """Extract unique speaker names from speaker labels in the transcript."""
    attendees: list[str] = []
    for line in transcript.splitlines():
        match = _SPEAKER_LABEL.match(line.strip())
        if match:
            name = match.group(1)
            if name not in attendees:
                attendees.append(name)
    return attendees


def _split_into_sections(transcript: str) -> list[str]:
    """
    Split transcript into discussion sections.

    Splits on double newlines (blank lines). If no blank-line breaks
    exist, groups every 5 lines into a section.
    """
    # Try splitting on blank lines first
    sections = re.split(r"\n\s*\n", transcript.strip())
    sections = [s.strip() for s in sections if s.strip()]

    if len(sections) > 1:
        return sections

    # Fallback: group every 5 lines
    lines = [ln for ln in transcript.splitlines() if ln.strip()]
    if not lines:
        return [transcript.strip()]

    grouped: list[str] = []
    for i in range(0, len(lines), 5):
        chunk = "\n".join(lines[i : i + 5])
        grouped.append(chunk)
    return grouped
