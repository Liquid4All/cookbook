# Meeting Intelligence Agent

A local AI agent that reads meeting transcripts and turns them into structured outputs: action items, a markdown summary, and a follow-up email — all running on your hardware.

## What it does

Given a meeting transcript, the agent:

1. Reads the transcript
2. Identifies every action item (owner, due date, description)
3. Looks up each owner in the local team directory
4. Creates a task record for each action item (`data/tasks.json`)
5. Saves a structured markdown summary (`data/summaries/`)
6. Drafts and "sends" a follow-up email (`data/sent_emails.log`)

No external APIs are called. All side effects are local files.

## Setup

```bash
cd examples/meeting-intelligence-agent
uv sync
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Usage

**Interactive mode (Anthropic backend):**
```bash
uv run mia
> Process the meeting transcript in data/sample_transcript.txt
```

**Non-interactive mode:**
```bash
uv run mia -p "Process data/sample_transcript.txt and save the summary as sprint-42.md"
```

**Local model (llama.cpp):**
```bash
# First, download the model and start the server:
bash start_model.sh models/LFM2-24B-A2B-Q4_K_M.gguf

# Then run with the local backend:
uv run mia --backend local
> Process the meeting transcript in data/sample_transcript.txt
```

## Configuration

| Environment variable    | Default                      | Description                        |
|-------------------------|------------------------------|------------------------------------|
| `MIA_BACKEND`           | `anthropic`                  | `anthropic` or `local`             |
| `MIA_ANTHROPIC_MODEL`   | `claude-sonnet-4-6`          | Anthropic model ID                 |
| `MIA_LOCAL_BASE_URL`    | `http://localhost:8080/v1`   | llama.cpp server URL               |
| `MIA_LOCAL_MODEL`       | `local`                      | Model name or HuggingFace path     |
| `MIA_LOCAL_CTX_SIZE`    | `32768`                      | Context window size                |
| `MIA_LOCAL_GPU_LAYERS`  | `99`                         | GPU layers to offload (0 = CPU)    |

## Demo outputs

After running the agent on `data/sample_transcript.txt`:

```bash
cat data/tasks.json          # structured task records
cat data/summaries/*.md      # markdown meeting summary
cat data/sent_emails.log     # follow-up email log
```
