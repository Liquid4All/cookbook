# Meeting Intelligence Agent — Project Brief

## Webinar context

This project is the demo for the **"How to build your own AI assistant — Step by step"** webinar.

The key narrative is:

> Every company has too many meetings. Decisions get made verbally, action items live in people's heads, and follow-ups fall through the cracks. A meeting intelligence agent reads the transcript after the call, extracts every action item, assigns it to the right person, opens a task in the project tracker, and drafts the follow-up email — all running locally, so confidential discussions never leave the building.

The model used is **LFM2-24B-A2B** (LiquidAI), which was chosen because:
- It runs on a single consumer laptop (32 GB RAM, no GPU required)
- It has native function-calling support
- It is explicitly designed for agentic tool use and privacy-preserving workflows
- It runs at 112 tok/s on CPU — fast enough for a live demo

---

## What the agent does

The user pastes or points to a meeting transcript. The agent then:

1. Reads the transcript
2. Identifies every action item mentioned
3. Looks up each owner in the local team directory to get their email
4. Creates a task record for each action item (written to a local JSON file)
5. Drafts and "sends" a follow-up email to all participants (appended to a log file)
6. Saves a structured markdown summary to disk

All side effects are local files — no external APIs, no cloud services.

---

## Architecture

Reuse the **exact same architecture** as `examples/local-coding-assistant`. The only things that change are:

- The project name and CLI entry point
- The `SYSTEM_PROMPT` in `context.py`
- The tools in `tools.py` (replace coding tools with meeting intelligence tools)
- The sample data files under `data/`

Everything else — the agentic loop, dual-backend LLM abstraction, context compaction, CLI flags — is identical. Copy it verbatim and adapt.

### File tree

```
examples/meeting-intelligence-agent/
├── pyproject.toml
├── README.md
├── .env.example
├── start_model.sh                     # llama.cpp server launch helper
├── data/
│   ├── team_directory.json            # mock team contact info (checked in)
│   ├── sample_transcript.txt          # example transcript for the demo (checked in)
│   ├── tasks.json                     # created by the agent at runtime
│   └── summaries/                     # meeting summary .md files saved by the agent
├── src/
│   └── meeting_intelligence/
│       ├── __init__.py
│       ├── main.py                    # CLI (click), same pattern as lca
│       ├── agent.py                   # agentic loop — copy verbatim from lca
│       ├── config.py                  # Config dataclass — copy verbatim, rename env prefix to MIA_
│       ├── context.py                 # ContextManager (copy) + new SYSTEM_PROMPT
│       ├── tools.py                   # 5 meeting intelligence tools (new)
│       └── llm/
│           ├── __init__.py            # get_llm_client factory — copy verbatim
│           ├── base.py                # LLMResponse + LLMClient protocol — copy verbatim
│           ├── anthropic_client.py    # copy verbatim from lca
│           └── llama_client.py        # copy verbatim from lca
```

---

## Step-by-step implementation plan

### Step 1 — Scaffold the project

Create the directory and initialise with uv:

```bash
mkdir -p examples/meeting-intelligence-agent
cd examples/meeting-intelligence-agent
uv init --no-workspace
uv add anthropic openai python-dotenv click
mkdir -p data/summaries
mkdir -p src/meeting_intelligence/llm
```

`pyproject.toml` — same structure as lca, change name and entry point:

```toml
[project]
name = "meeting-intelligence-agent"
version = "0.1.0"
description = "A local AI agent that turns meeting transcripts into action items"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "anthropic>=0.40.0",
    "openai>=1.50.0",
    "python-dotenv>=1.0.0",
    "click>=8.0.0",
]

[project.scripts]
mia = "meeting_intelligence.main:main"

[build-system]
requires = ["uv_build>=0.9.18,<0.10.0"]
build-backend = "uv_build"
```

`.env.example`:

```
ANTHROPIC_API_KEY=sk-ant-...

# Set to "local" to use a local llama.cpp server instead of Anthropic
MIA_BACKEND=anthropic
MIA_ANTHROPIC_MODEL=claude-sonnet-4-6

# Local server settings (only used when MIA_BACKEND=local)
MIA_LOCAL_BASE_URL=http://localhost:8080/v1
MIA_LOCAL_MODEL=local
MIA_LOCAL_CTX_SIZE=32768
MIA_LOCAL_GPU_LAYERS=99
```

---

### Step 2 — Copy the LLM layer verbatim

Copy these four files directly from `examples/local-coding-assistant/src/local_coding_assistant/`:

- `llm/base.py` → `src/meeting_intelligence/llm/base.py`
- `llm/anthropic_client.py` → `src/meeting_intelligence/llm/anthropic_client.py`
- `llm/llama_client.py` → `src/meeting_intelligence/llm/llama_client.py`
- `llm/__init__.py` → `src/meeting_intelligence/llm/__init__.py`

No changes needed.

---

### Step 3 — Copy config.py, rename env prefix

Copy `config.py` from lca. Replace every `LCA_` prefix with `MIA_` in the `os.getenv` calls. The dataclass fields and defaults remain the same.

---

### Step 4 — Copy agent.py verbatim

`agent.py` has zero domain logic — it is a pure agentic loop. Copy it verbatim from lca and update the import path from `local_coding_assistant` to `meeting_intelligence`.

---

### Step 5 — Write the new context.py

Keep `ContextManager` exactly as in lca (copy verbatim). Replace only `SYSTEM_PROMPT`:

```python
SYSTEM_PROMPT = """\
You are a Meeting Intelligence Agent running entirely on local hardware.
Your job is to process meeting transcripts and turn them into structured outputs.

You have access to these tools:
- read_transcript: read a meeting transcript file from disk
- lookup_team_member: look up a person's email and role from the team directory
- create_task: create a task record in the local project tracker
- save_summary: save the meeting summary as a markdown file
- send_email: send a follow-up email (runs locally, appended to a log)

Workflow for every transcript you receive:
1. Read the transcript with read_transcript
2. Identify all action items: what was decided, who owns it, when it is due
3. For each owner, call lookup_team_member to get their email address
4. Call create_task once per action item
5. Call save_summary to save a structured markdown summary
6. Call send_email to send a recap to all participants
7. Report back what you did

Be concise. Show your work through tool calls, not long explanations.
If a due date is not mentioned, default to one week from today.
If an owner is not mentioned, leave it as "unassigned".
"""
```

---

### Step 6 — Write tools.py

This is the only file with meaningful new logic. There are five tools.

#### `read_transcript(path: str) -> str`

Read a transcript file from disk. Same implementation as `read_file` in lca — use `Path(path).read_text(encoding="utf-8")`. Return the content, or a `[error]` string on failure.

#### `lookup_team_member(name: str) -> str`

Search `data/team_directory.json` for a team member whose name contains the query (case-insensitive). Return a JSON string with their record, or `"[not found]"` if nobody matches.

The `team_directory.json` format (see Step 7) is a list of objects:
```json
{"name": "Alice Chen", "email": "alice@acme.com", "role": "Engineering Lead", "team": "Platform"}
```

Implementation sketch:
```python
import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent.parent / "data"

def lookup_team_member(name: str) -> str:
    directory_path = _DATA_DIR / "team_directory.json"
    members = json.loads(directory_path.read_text())
    query = name.lower()
    matches = [m for m in members if query in m["name"].lower()]
    if not matches:
        return f'[not found] No team member matching "{name}"'
    return json.dumps(matches[0], indent=2)
```

#### `create_task(title: str, owner: str, due_date: str, description: str) -> str`

Append a task to `data/tasks.json`. Load existing list (or start with `[]`), append a new task dict with an auto-incremented `id`, `status: "open"`, and the provided fields. Write back to disk.

```python
def create_task(title: str, owner: str, due_date: str, description: str) -> str:
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
```

#### `save_summary(filename: str, content: str) -> str`

Write a markdown file to `data/summaries/{filename}`. Create the directory if it doesn't exist.

```python
def save_summary(filename: str, content: str) -> str:
    summaries_dir = _DATA_DIR / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    output_path = summaries_dir / filename
    output_path.write_text(content, encoding="utf-8")
    return f"Summary saved to {output_path}"
```

#### `send_email(to: str, subject: str, body: str) -> str`

Mock email sender. Append to `data/sent_emails.log` with a timestamp and separator. Return a confirmation string.

```python
from datetime import datetime

def send_email(to: str, subject: str, body: str) -> str:
    log_path = _DATA_DIR / "sent_emails.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n{'='*60}\n[{timestamp}]\nTo: {to}\nSubject: {subject}\n\n{body}\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    return f'Email sent to {to} — Subject: "{subject}"'
```

#### TOOLS list and execute_tool dispatcher

Define `TOOLS` as a list of JSON Schema dicts (same pattern as lca). One entry per tool, with `name`, `description`, and `parameters`. Map them in `TOOL_FUNCTIONS` dict and dispatch via `execute_tool(name, inputs)`.

---

### Step 7 — Create the sample data files

#### `data/team_directory.json`

```json
[
  {"name": "Alice Chen",    "email": "alice@acme.com",   "role": "Engineering Lead",  "team": "Platform"},
  {"name": "Bob Martinez",  "email": "bob@acme.com",     "role": "Product Manager",   "team": "Product"},
  {"name": "Carol Davis",   "email": "carol@acme.com",   "role": "Designer",          "team": "Design"},
  {"name": "David Kim",     "email": "david@acme.com",   "role": "Backend Engineer",  "team": "Platform"},
  {"name": "Emma Wilson",   "email": "emma@acme.com",    "role": "QA Engineer",       "team": "Quality"},
  {"name": "Frank Nguyen",  "email": "frank@acme.com",   "role": "DevOps Engineer",   "team": "Infrastructure"},
  {"name": "Grace Lee",     "email": "grace@acme.com",   "role": "Data Analyst",      "team": "Analytics"},
  {"name": "Henry Patel",   "email": "henry@acme.com",   "role": "Frontend Engineer", "team": "Platform"}
]
```

#### `data/sample_transcript.txt`

Write a realistic 300–400 word transcript of a 30-minute sprint planning meeting at a fictional SaaS company (Acme Corp). Include:
- 4–5 attendees (use names from the team directory above)
- 5–7 concrete action items with clear owners
- At least 2 action items with explicit deadlines ("by Friday", "end of next week")
- Some items without explicit deadlines (agent defaults to +7 days)
- One item with an unclear owner (agent should leave as "unassigned")
- Natural conversation filler so it reads like a real transcript

Example opening:
```
Sprint Planning — Acme Corp Platform Team
Date: [leave blank, agent will use today's date]
Attendees: Alice Chen, Bob Martinez, Carol Davis, David Kim, Emma Wilson

[00:00] Alice: Okay let's get started. Bob, do you want to kick us off with the priorities?

[00:15] Bob: Sure. So the big thing this sprint is getting the new onboarding flow shipped.
Carol, the mockups you shared yesterday looked great — can you finalise the mobile
screens by end of week so David can start implementation?

[00:45] Carol: Yes, I'll have them done by Friday.
...
```

Make sure the transcript contains these action items (mix of explicit and implicit):
1. Carol finalises mobile mockups → Carol Davis → Friday
2. David implements onboarding backend endpoints → David Kim → end of next week
3. Emma writes test plan for onboarding flow → Emma Wilson → no date mentioned
4. Frank sets up staging environment → Frank Nguyen → Monday
5. Henry fixes the login page performance regression → Henry Patel → this sprint (no exact date)
6. Someone should update the API docs (no clear owner)
7. Bob schedules a demo with the CEO → Bob Martinez → next Thursday

---

### Step 8 — Write main.py

Same structure as lca's `main.py`. Changes:
- Update banner text to "Meeting Intelligence Agent"
- Change env prefix references to `MIA_`
- Change `server_proc` logic (copy verbatim — it references `config.backend == "local"`)
- Entry point is `mia` not `lca`

The `-p` / `--prompt` non-interactive flag is valuable for the webinar demo — keep it.

---

### Step 9 — Write `start_model.sh`

Copy from lca. This script downloads and starts the llama.cpp server with LFM2-24B-A2B:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/LFM2-24B-A2B.gguf}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Model not found at $MODEL_PATH"
  echo "Download with:"
  echo "  huggingface-cli download LiquidAI/LFM2-24B-A2B-GGUF --include 'LFM2-24B-A2B-Q4_K_M.gguf' --local-dir models/"
  exit 1
fi

llama-server \
  --model "$MODEL_PATH" \
  --ctx-size 32768 \
  --n-gpu-layers 0 \
  --host 0.0.0.0 \
  --port 8080
```

---

### Step 10 — Write `__init__.py` files

All `__init__.py` files are empty. Create them for:
- `src/meeting_intelligence/__init__.py`
- `src/meeting_intelligence/llm/__init__.py` (copy the `get_llm_client` factory from lca)

---

## Demo script for the webinar

### Setup (before the webinar)
```bash
cd examples/meeting-intelligence-agent
uv sync
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env (for the Anthropic backend demo)
```

### Live demo sequence

**Step 1 — Show the transcript**
```bash
cat data/sample_transcript.txt
```
_Talk through: "This is a 30-minute meeting. 7 people, lots of decisions. Who remembers what they need to do by Monday?"_

**Step 2 — Run the agent (Anthropic backend, interactive)**
```bash
uv run mia
> Process the meeting transcript in data/sample_transcript.txt
```
Watch the agent call tools in sequence and narrate each one.

**Step 3 — Show the outputs**
```bash
cat data/tasks.json
cat data/summaries/*.md
cat data/sent_emails.log
```

**Step 4 — Switch to local model**
```bash
# In a second terminal, if the model is running:
uv run mia --backend local --model models/LFM2-24B-A2B-Q4_K_M.gguf
> Process the meeting transcript in data/sample_transcript.txt
```
_Talk through: "Same agent, same tools, same output — but the transcript never leaves this laptop."_

**Step 5 — Non-interactive mode (automation story)**
```bash
uv run mia --backend local -p "Process data/sample_transcript.txt and save the summary as sprint-42.md"
```
_Talk through: "This is how you'd plug it into a cron job or a Zoom webhook."_

---

## Key teaching moments

| Moment | What to explain |
|---|---|
| Agent calls `read_transcript` first | "The model decides to read the file before doing anything — we didn't tell it to, that's the system prompt guiding it" |
| Agent calls `lookup_team_member` for each owner | "It's enriching the data — connecting names to real email addresses from your internal directory" |
| Agent calls `create_task` 7 times in a row | "This is the inner loop — the model loops until there's nothing left to do" |
| Tool result feeds back into next turn | "Tool output is just another message in the conversation — that's the whole trick" |
| Switch to local model, same output | "Privacy by default: swap one environment variable, the model runs on your hardware" |
| `-p` flag for non-interactive | "This is what connects to your Zoom webhook, your email parser, your cron job" |

---

## Extension ideas (if time permits or for follow-up)

- **RAG over past summaries**: add a `search_past_meetings(query)` tool that does keyword search over the `summaries/` directory, so the agent can reference decisions made in previous meetings
- **Calendar integration**: add a `schedule_followup(attendees, date, agenda)` tool that writes to a local `.ics` file
- **Slack mock**: replace `send_email` with `post_to_slack_channel(channel, message)` that writes to a log — same implementation, more relatable for tech audiences
- **Multi-meeting batch mode**: process a whole folder of transcripts and produce a weekly digest
