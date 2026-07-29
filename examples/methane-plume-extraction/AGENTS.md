# AGENTS.md - Methane Plume Demo Runner

## Mission

Open and validate the local methane plume structured-output demo. Do not train,
relabel, or regenerate the evaluation set from this repo. The expected behavior
is:

```text
one rendered methane observation image + fixed schema -> one flat JSON object
```

## Operating Rules

- Run commands from `examples/methane-plume-extraction/`.
- Do not commit model weights, caches, `.env`, tokens, or generated `models/`
  directories.
- Keep the fine-tuned model private on Hugging Face unless the owner explicitly
  changes that policy.
- Use live inference for demos. Built-in samples include ground truth for
  assessment, but model outputs should come from the local inference server.
- The schema is intentionally fixed for the demo. Do not edit `schema.yaml`
  during a customer demo. Treat key renames, new fields, changed allowed values,
  or nested JSON as a new schema that needs re-evaluation before use.

## Required Access

The operator needs:

- Read access to `LiquidAI/LFM2.5-VL-1.6B`.
- Read access to the private fine-tuned model repo:

```text
felipeliquid/LFM2.5-1.6B-VL-Extract-Plume-Demo
```

Authenticate with:

```bash
hf auth login
hf auth whoami
```

Prefer `HF_TOKEN` or `hf auth login`; do not paste tokens into scripts.

## First-Time Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Download models:

```bash
./scripts/download_models.sh
```

This creates:

```text
models/base
models/fine
```

## Run The Demo

Terminal 1:

```bash
source .venv/bin/activate
python scripts/serve_local_methane.py --preload fine
```

Terminal 2:

```bash
source .venv/bin/activate
METHANE_INFER_URL=http://127.0.0.1:8791/infer \
  python scripts/serve_demo.py
```

Open:

```text
http://127.0.0.1:8787/demo
```

## Smoke Test

1. Select `Core absent`.
2. Keep `Fine-tuned` selected.
3. Click `Run`.
4. Confirm the output tab shows valid JSON and latency.
5. Confirm the assessment panel shows most fields verified.
6. Switch to `Base`, click `Run`, and confirm the base model is visibly worse
   at following the methane schema.

The exact free-text evidence sentence may differ. Treat it as correct when the
intent matches the ground truth and it does not add wrong visual information.

## Expected Architecture

```text
browser -> scripts/serve_demo.py:/api/infer -> scripts/serve_local_methane.py:/infer -> local HF model
```

The visible endpoint control is hidden under the page's Advanced section because
normal demo users should only choose a model and click Run.

## If Something Fails

- `401` or `403` during download: request Hugging Face access to the private
  fine-tuned repo or use a token with the right account.
- Browser output says endpoint unavailable: restart both servers and check
  `METHANE_INFER_URL`.
- Fine-tuned model emits captions instead of JSON: verify the fine model was
  downloaded into `models/fine`, not accidentally pointed at the base model.
- MLX output differs from Transformers: use the default Transformers backend for
  the customer demo and investigate MLX parity separately.

## Deployment Boundary

This package is a demo runner. Production customer work should go back through
the structured-output fine-tune process: customer schema, held-out eval set, L0
baseline, fine-tune only if needed, LLM and VLM judge, then export/deploy.
