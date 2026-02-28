"""Benchmark task suite: Meeting Intelligence Agent.

10 tasks ordered easy → hard. Each task runs in a fresh agent context
against the sample data in data/.

Verifiers check the actual side-effect files written by the agent's tools
(data/tasks.json, data/summaries/*.md, data/sent_emails.log) as well as
the captured stdout.

Tasks 1-3:  easy   — single tool call, given inputs
Tasks 4-6:  medium — multi-step chains, requires reading the transcript
Tasks 7-10: hard   — full agentic pipeline, judgment calls (unassigned
                       owners, date defaulting, targeted emails, dependencies)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable


@dataclass
class Task:
    id: int
    difficulty: str          # "easy" | "medium" | "hard"
    name: str
    prompt: str
    verifier: Callable[[str, Path], bool]


# ── Data helpers used by verifiers ───────────────────────────────────────────

def _tasks(data_dir: Path) -> list[dict]:
    p = data_dir / "tasks.json"
    return json.loads(p.read_text()) if p.exists() else []


def _summary(data_dir: Path, filename: str) -> str:
    p = data_dir / "summaries" / filename
    return p.read_text() if p.exists() else ""


def _email_log(data_dir: Path) -> str:
    p = data_dir / "sent_emails.log"
    return p.read_text() if p.exists() else ""


# ── Task definitions ──────────────────────────────────────────────────────────

TASKS: list[Task] = [

    # ── Easy: single tool, given inputs ──────────────────────────────────────

    Task(
        id=1,
        difficulty="easy",
        name="Read transcript and list attendees",
        prompt=(
            "Read the meeting transcript at data/sample_transcript.txt "
            "and list all the attendees mentioned in the header."
        ),
        verifier=lambda stdout, d: (
            sum(name in stdout for name in [
                "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Henry"
            ]) >= 5
        ),
    ),

    Task(
        id=2,
        difficulty="easy",
        name="Look up one team member",
        prompt=(
            "Look up Henry Patel in the team directory "
            "and tell me his email address and role."
        ),
        verifier=lambda stdout, d: (
            "henry@acme.com" in stdout
            and "frontend" in stdout.lower()
        ),
    ),

    Task(
        id=3,
        difficulty="easy",
        name="Create one explicit task",
        prompt=(
            "Create a single task with these details:\n"
            "- Title: Fix login page performance regression\n"
            "- Owner: Henry Patel\n"
            "- Due date: 2026-03-07\n"
            "- Description: Investigate and fix the login page performance issue "
            "reported by three enterprise customers"
        ),
        verifier=lambda stdout, d: (
            bool(_tasks(d))
            and any("login" in t.get("title", "").lower() for t in _tasks(d))
        ),
    ),

    # ── Medium: multi-step chains ─────────────────────────────────────────────

    Task(
        id=4,
        difficulty="medium",
        name="Look up three team members",
        prompt=(
            "Look up Alice Chen, Bob Martinez, and Emma Wilson in the team directory. "
            "For each person, report their email address and role."
        ),
        verifier=lambda stdout, d: all(
            email in stdout
            for email in ["alice@acme.com", "bob@acme.com", "emma@acme.com"]
        ),
    ),

    Task(
        id=5,
        difficulty="medium",
        name="Create three tasks from a given list",
        prompt=(
            "Create tasks for these three action items:\n"
            "1. Title: Finalise mobile mockups, Owner: Carol Davis, "
            "Due: 2026-03-07, "
            "Description: Complete mobile screen designs for the onboarding flow\n"
            "2. Title: Set up staging environment, Owner: Frank Nguyen, "
            "Due: 2026-03-09, "
            "Description: Configure a staging environment that mirrors production\n"
            "3. Title: Write onboarding test plan, Owner: Emma Wilson, "
            "Due: 2026-03-14, "
            "Description: Draft and circulate the QA test plan for the onboarding flow"
        ),
        verifier=lambda stdout, d: (
            len(_tasks(d)) >= 3
            and any(
                "mockup" in t.get("title", "").lower()
                or "mobile" in t.get("title", "").lower()
                for t in _tasks(d)
            )
            and any("staging" in t.get("title", "").lower() for t in _tasks(d))
            and any("test" in t.get("title", "").lower() for t in _tasks(d))
        ),
    ),

    Task(
        id=6,
        difficulty="medium",
        name="Read transcript and save a structured summary",
        prompt=(
            "Read the meeting transcript at data/sample_transcript.txt and save "
            "a structured markdown summary as sprint-planning.md. "
            "The summary must include: a title, the list of attendees, "
            "key decisions, and a table of action items."
        ),
        verifier=lambda stdout, d: (
            bool(_summary(d, "sprint-planning.md"))
            and len(_summary(d, "sprint-planning.md")) > 300
            and "#" in _summary(d, "sprint-planning.md")
            and "action" in _summary(d, "sprint-planning.md").lower()
        ),
    ),

    # ── Hard: full pipeline, judgment calls ───────────────────────────────────

    Task(
        id=7,
        difficulty="hard",
        name="Full pipeline: tasks + summary + email",
        prompt=(
            "Process the meeting transcript in data/sample_transcript.txt. "
            "Extract every action item, look up each owner's email, "
            "create a task for each action item, save a markdown summary "
            "as sprint-summary.md, and send a recap email to all attendees."
        ),
        verifier=lambda stdout, d: (
            len(_tasks(d)) >= 5
            and bool(_summary(d, "sprint-summary.md"))
            and "alice@acme.com" in _email_log(d)
            and "bob@acme.com" in _email_log(d)
        ),
    ),

    Task(
        id=8,
        difficulty="hard",
        name="Detect and flag unassigned action item",
        prompt=(
            "Process the meeting transcript in data/sample_transcript.txt. "
            "Create a task for every action item you find. "
            "For any item where the owner is unclear or not explicitly named, "
            "set the owner field to exactly 'unassigned'. "
            "Save a summary as meeting-recap.md and send a recap email to all attendees."
        ),
        verifier=lambda stdout, d: (
            len(_tasks(d)) >= 5
            and any(
                "unassigned" in t.get("owner", "").lower()
                for t in _tasks(d)
            )
            and bool(_summary(d, "meeting-recap.md"))
        ),
    ),

    Task(
        id=9,
        difficulty="hard",
        name="Default due dates for items without explicit deadlines",
        prompt=(
            f"Read data/sample_transcript.txt. Extract all action items and "
            f"create a task for each one. Today's date is "
            f"{date.today().isoformat()}. For any action item where no "
            f"deadline is explicitly mentioned in the transcript, set the due "
            f"date to exactly one week from today "
            f"({(date.today() + timedelta(days=7)).isoformat()})."
        ),
        verifier=lambda stdout, d: (
            len(_tasks(d)) >= 5
            and all(t.get("due_date") for t in _tasks(d))
            and any(
                t.get("due_date") == (date.today() + timedelta(days=7)).isoformat()
                for t in _tasks(d)
            )
        ),
    ),

    Task(
        id=10,
        difficulty="hard",
        name="Full pipeline: custom filename and targeted email recipients",
        prompt=(
            "Process data/sample_transcript.txt. Create tasks for all action items. "
            "Save the meeting summary as q1-kickoff-summary.md. "
            "Send the follow-up email ONLY to the engineering team: "
            "alice@acme.com, david@acme.com, and henry@acme.com. "
            "Do not send to anyone else."
        ),
        verifier=lambda stdout, d: (
            len(_tasks(d)) >= 5
            and bool(_summary(d, "q1-kickoff-summary.md"))
            and all(
                email in _email_log(d)
                for email in ["alice@acme.com", "david@acme.com", "henry@acme.com"]
            )
        ),
    ),
]
