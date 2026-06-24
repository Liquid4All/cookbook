# Model Access

The demo does not store model weights in GitHub. Operators download the base and
fine-tuned models from Hugging Face.

## Base Model

```text
LiquidAI/LFM2.5-VL-1.6B
```

If Hugging Face requires acceptance of model terms, accept them from the model
page before running `scripts/download_models.sh`.

## Fine-Tuned Model

The fine-tuned model should be a private Hugging Face model repo named:

```text
LFM2.5-1.6B-VL-Extract-Plume-Demo
```

Full repo id:

```text
felipeliquid/LFM2.5-1.6B-VL-Extract-Plume-Demo
```

Grant demo users read access to that private repo, then have them authenticate:

```bash
hf auth login
hf auth whoami
```

They can then set:

```bash
export FINE_MODEL_ID=felipeliquid/LFM2.5-1.6B-VL-Extract-Plume-Demo
```

## Upload Reminder For The Owner

If the private fine-tuned repo has not been uploaded yet:

```bash
hf repos create felipeliquid/LFM2.5-1.6B-VL-Extract-Plume-Demo \
  --type model \
  --private \
  --exist-ok

hf upload-large-folder \
  felipeliquid/LFM2.5-1.6B-VL-Extract-Plume-Demo \
  /path/to/LFM2.5-1.6B-VL-Extract-Plume-Demo \
  --type model \
  --private
```

Use `HF_TOKEN` or `hf auth login`; do not hard-code tokens.
