# agent.md - Open The Demo Locally

This is the human-facing shortcut. The canonical agent runbook is
[AGENTS.md](AGENTS.md).

## Quick Path

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

hf auth login
./scripts/download_models.sh
```

Terminal 1:

```bash
source .venv/bin/activate
python scripts/serve_local_methane.py --preload fine
```

Terminal 2:

```bash
source .venv/bin/activate
METHANE_INFER_URL=http://127.0.0.1:8791/infer python scripts/serve_demo.py
```

Open:

```text
http://127.0.0.1:8787/demo
```

Select an image, choose Base or Fine-tuned, then click Run.

## What Should Happen

The browser sends one rendered methane observation plus the fixed schema to the
local model server. The selected model returns one flat JSON object. Built-in
samples show an assessment against ground truth; uploaded images show live JSON
without ground-truth scoring.
