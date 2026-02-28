import json
from datetime import datetime
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def read_transcript(path: str) -> str:
    """Read a meeting transcript file from disk."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"[error] File not found: {path}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


def lookup_team_member(name: str) -> str:
    """Look up a team member by name from the local directory."""
    try:
        directory_path = _DATA_DIR / "team_directory.json"
        members = json.loads(directory_path.read_text())
        query = name.lower()
        matches = [m for m in members if query in m["name"].lower()]
        if not matches:
            return f'[not found] No team member matching "{name}"'
        return json.dumps(matches[0], indent=2)
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


def create_task(title: str, owner: str, due_date: str, description: str) -> str:
    """Create a task record in the local project tracker."""
    tasks_path = _DATA_DIR / "tasks.json"
    tasks = json.loads(tasks_path.read_text()) if tasks_path.exists() else []
    task = {
        "id": len(tasks) + 1,
        "title": title,
        "owner": owner,
        "due_date": due_date,
        "description": description,
        "status": "open",
    }
    tasks.append(task)
    tasks_path.write_text(json.dumps(tasks, indent=2))
    return f'Task #{task["id"]} created: "{title}" → {owner} by {due_date}'


def save_summary(filename: str, content: str) -> str:
    """Save a meeting summary as a markdown file."""
    summaries_dir = _DATA_DIR / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    output_path = summaries_dir / filename
    output_path.write_text(content, encoding="utf-8")
    return f"Summary saved to {output_path}"


def send_email(to: str, subject: str, body: str) -> str:
    """Send a follow-up email (mock: appended to a local log file)."""
    log_path = _DATA_DIR / "sent_emails.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n{'='*60}\n[{timestamp}]\nTo: {to}\nSubject: {subject}\n\n{body}\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    return f'Email sent to {to} — Subject: "{subject}"'


# JSON Schema definitions sent to the model
TOOLS: list[dict] = [
    {
        "name": "read_transcript",
        "description": "Read a meeting transcript file from disk and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the transcript file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "lookup_team_member",
        "description": "Look up a person's email address and role from the local team directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full or partial name of the team member to look up"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "create_task",
        "description": "Create a task record in the local project tracker (data/tasks.json).",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short title for the task"},
                "owner": {"type": "string", "description": "Name of the person responsible, or 'unassigned'"},
                "due_date": {"type": "string", "description": "Due date in YYYY-MM-DD format"},
                "description": {"type": "string", "description": "Detailed description of what needs to be done"},
            },
            "required": ["title", "owner", "due_date", "description"],
        },
    },
    {
        "name": "save_summary",
        "description": "Save the meeting summary as a markdown file under data/summaries/.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename for the summary (e.g. 'sprint-42.md')"},
                "content": {"type": "string", "description": "Full markdown content of the summary"},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "send_email",
        "description": "Send a follow-up email to meeting participants (mock: appends to data/sent_emails.log).",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address(es), comma-separated"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body text"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]

TOOL_FUNCTIONS: dict[str, object] = {
    "read_transcript": read_transcript,
    "lookup_team_member": lookup_team_member,
    "create_task": create_task,
    "save_summary": save_summary,
    "send_email": send_email,
}


def execute_tool(name: str, inputs: dict) -> str:
    """Dispatch a tool call by name and return the string result."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"[error] Unknown tool: {name}"
    try:
        return fn(**inputs)  # type: ignore[operator]
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"
