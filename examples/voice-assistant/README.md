# Voice Assistant powered by a local LFM2.5-Audio

This project builds a local voice assistant that maps spoken Home Assistant
commands directly to function calls, using a fine-tuned `LFM2.5-Audio-1.5B`
running entirely on-device via llama.cpp. No cloud, no STT pipeline, no
intermediate transcription step: audio in, function call out.

In this tutorial you will:

1. [Establish the floor](#step-1-establish-the-floor): evaluate the unmodified
   `LFM2.5-Audio-1.5B` and show that without fine-tuning it can only transcribe,
   not call functions.
2. [Preprocess and fine-tune](#step-2-preprocess-and-fine-tune) on
   [OHF-Voice](https://huggingface.co/datasets/Paulescu/OHF-Voice-audio-20260504),
   55,302 (audio, function call) pairs across 41 Home Assistant operations.
3. [Evaluate the fine-tuned model](#step-3-evaluate-the-fine-tuned-model) and
   measure the gap.
4. [Quantize and deploy](#step-4-quantize-and-deploy) the result as a GGUF pair
   that runs inside `llama-liquid-audio-server`.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) for the Python
  toolchain.
- An HF account + `HF_TOKEN` with write access to your username, for pushing
  the dataset/model artifacts you produce.
- A [Modal](https://modal.com) account for the fine-tuning step (an
  A100-80GB run takes a few hours).
- For local GGUF eval / deployment: nothing extra. `scripts/eval.py` downloads
  the prebuilt `llama-liquid-audio-server` binary for your platform.
- For quantization: `git`, `cmake`, and a C++ compiler (on macOS:
  `xcode-select --install` and `brew install cmake`).

## Quick start: run the published fine-tuned model

If you just want to see the fine-tuned model in action and you don't care
about reproducing the training, run the published-checkpoint eval directly:

```bash
git clone https://github.com/Liquid4All/cookbook
cd cookbook/examples/voice-assistant
uv sync
HF_TOKEN=hf_... uv run python scripts/eval.py --config configs/finetuned.yaml
```

This downloads `Paulescu/LFM2.5-Audio-1.5B-OHF-Voice-GGUF` (the four-file
audio model + the platform runner) and evaluates it on 397 stratified samples
from the held-out test split.

## The dataset

[`Paulescu/OHF-Voice-audio-20260504`](https://huggingface.co/datasets/Paulescu/OHF-Voice-audio-20260504)
is a fork of
[`LiquidAI/OHF-Voice-audio-20260504`](https://huggingface.co/datasets/LiquidAI/OHF-Voice-audio-20260504)
with a deterministic 95/5 train/test split, stratified by function name. The
upstream dataset ships only a `train` split; the fork carves a disjoint test
set once, so train and eval can never accidentally overlap.

| split | samples |
|---|---|
| train | 52,536 |
| test  | 2,766  |
| total | 55,302 |

Each row is an (audio, function call) pair. The audio is the spoken command
(WAV, 24 kHz); the function call is the model's target output. Format:

```
HassStartTimer|$minutes=5|$name=oven
HassLightSet|$area=bedroom|$brightness=70
HassGetCurrentTime
```

41 distinct function names. The four most common (Timer-related) cover ~28% of
all samples; the rarest (`HassRespond`) appears 94 times. See
[`CONTEXT.md`](./CONTEXT.md) for the full vocabulary.

If you want to reproduce the split locally instead of consuming the published
artifact, run:

```bash
HF_TOKEN=hf_... uv run python scripts/prepare_raw_data.py --dry-run
HF_TOKEN=hf_... uv run python scripts/prepare_raw_data.py
```

The first invocation prints per-function counts; the second pushes the split
to your HF namespace.

## The model

[`LiquidAI/LFM2.5-Audio-1.5B`](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)
is a 1.5B-parameter multimodal model that consumes audio (and/or text) and
emits audio (and/or text). For this project we only use the audio-to-text
direction: spoken command in, function call text out.

Inference on-device runs through
[`llama-liquid-audio-server`](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF),
a custom build of llama.cpp from
[PR #18641](https://github.com/ggml-org/llama.cpp/pull/18641) that adds audio
multimodal support. Prebuilt binaries for macos-arm64, ubuntu-x64, ubuntu-arm64,
and android-arm64 ship inside the GGUF repo under `runners/`. `scripts/eval.py`
pulls the right one for your platform.

## Step 1: Establish the floor

Before fine-tuning, run the eval against the *unmodified* model to see what
the floor looks like.

```bash
HF_TOKEN=hf_... uv run python scripts/eval.py --config configs/baseline.yaml
```

[`configs/baseline.yaml`](./configs/baseline.yaml) points at
`LiquidAI/LFM2.5-Audio-1.5B-GGUF` with `system_prompt: "Perform ASR."` (the
only viable audio-to-text behavioral switch the runtime exposes; see
[ADR-0001](./docs/adr/0001-eval-methodology.md) for why we don't inject
function specs into the prompt). The eval downloads ~3 GB the first time and
runs ~400 samples sequentially.

Expected result: 0% across all three layered metrics. The unmodified model is
doing plain ASR, faithfully transcribing what was said:

| metric                  | baseline result |
|---|---|
| Format compliance       | 0/397   (0.0%)  |
| Function-name accuracy  | 0/397   (0.0%)  |
| Argument accuracy       | 0/397   (0.0%)  |

```
gt:   HassBroadcast|$message=dinner is ready
pred: "Dinner is ready."
```

This is exactly what we want from the floor: no amount of prompting on this
runtime can get function-call output out of the unmodified weights. Fine-tuning
isn't an optimization, it's the only path.

## Step 2: Preprocess and fine-tune

Two stages: preprocess the train split into the on-disk tensor format the
trainer expects, then run the fine-tune on Modal.

### Preprocess

```bash
HF_TOKEN=hf_... uv run --group finetune python scripts/preprocess_ohf_voice.py --modal
```

This runs `scripts/preprocess_ohf_voice.py` on a Modal A100. It reads
`Paulescu/OHF-Voice-audio-20260504` (train split only), passes each row through
`LFM2AudioProcessor`, and writes the resulting tensor dataset to the Modal
volume `ohf-voice-data` at `/output/train`. Skips any sample whose tokenised
length exceeds 512 tokens.

### Fine-tune

```bash
HF_TOKEN=hf_... uv run --group finetune python scripts/train.py \
    --modal \
    --log-with wandb \
    --wandb-run-name "ohf-voice-run-1"
```

This runs `scripts/train.py` on a Modal A100-80GB. Defaults baked in:

| knob | value |
|---|---|
| `--context-length` | 512 |
| `--batch-size`     | 32 |
| `--max-steps`      | 10,000 |
| `--warmup-steps`   | 250 |
| `--lr`             | 5e-5 |
| `--val-split-ratio`| 0.05 (carved from train, **not** the test split) |
| `--seed`           | 42 |

Checkpoints land in the Modal volume `lfm2-training-output`. The validation
slice is carved from the train split, never from the held-out test set, so the
final eval against `Paulescu/OHF-Voice-audio-20260504` `test` remains clean.

Both `preprocess_ohf_voice.py` and `train.py` are vendored from
`liquid-audio-staging` (`examples/audio-to-function-calling` branch, commit
`376b06a`). See [ADR-0002](./docs/adr/0002-vendoring-strategy.md) for why we
vendor instead of upstreaming.

When training finishes, push the resulting safetensors checkpoint to HF:

```
Paulescu/LFM2.5-Audio-1.5B-OHF-Voice
```

(The exact push mechanism depends on how you choose to drain the Modal volume.
The simplest option is `modal volume get` then `huggingface-cli upload-large-folder`.)

## Step 3: Evaluate the fine-tuned model

Update [`configs/finetuned.yaml`](./configs/finetuned.yaml) to point at your
fine-tuned GGUF repo (default: `Paulescu/LFM2.5-Audio-1.5B-OHF-Voice-GGUF`),
then:

```bash
HF_TOKEN=hf_... uv run python scripts/eval.py --config configs/finetuned.yaml
```

Same 397-sample stratified test subset as the baseline. Expected outcome (numbers
filled in once the canonical run lands):

| metric                  | baseline    | fine-tuned |
|---|---|---|
| Format compliance       | 0%          | _TBD_      |
| Function-name accuracy  | 0%          | _TBD_      |
| Argument accuracy       | 0%          | _TBD_      |

Per-function breakdown lands at
`evals/finetuned_<timestamp>/report.md` after the run.

## Step 4: Quantize and deploy

The fine-tuned checkpoint is a safetensors HF model. To run it inside
`llama-liquid-audio-server` (the thing `scripts/eval.py` uses) we have to convert
to GGUF.

```bash
HF_TOKEN=hf_... uv run --group finetune python scripts/quantize.py \
    --source-repo Paulescu/LFM2.5-Audio-1.5B-OHF-Voice \
    --target-repo Paulescu/LFM2.5-Audio-1.5B-OHF-Voice-GGUF \
    --quant F16
```

`scripts/quantize.py`:

1. Clones llama.cpp at PR #18641 branch (audio multimodal support, not yet on main).
2. Runs `convert_hf_to_gguf.py` twice on your checkpoint: once for the LM backbone,
   once with `--mmproj` for the audio encoder + projector.
3. Quantizes the LM to your target level (`--quant`) via `llama-quantize`. Mmproj
   stays F16.
4. Copies the upstream `vocoder` and `tokenizer` GGUFs (these don't change with
   fine-tuning) and the `runners/` folder of prebuilt binaries from
   `LiquidAI/LFM2.5-Audio-1.5B-GGUF`.
5. Pushes the four GGUFs + `runners/` to your target repo with a model card.

The resulting repo is self-contained: `scripts/eval.py` (or any other
llama-liquid-audio-server consumer) downloads everything it needs from one
place.

To verify the deployed model works end-to-end, point `configs/finetuned.yaml`
at the new repo and re-run `scripts/eval.py`. The numbers should match what
you got in Step 3 (any drift is the quantization tax, surfaced as a side-by-side
F16 vs Q8_0 comparison).

## What's next

- Try a smaller quant (`--quant Q4_0`) and measure the accuracy/size tradeoff.
- Run the published GGUF on a Pi 5 or a phone via `llama-liquid-audio-cli` to
  see real on-device latency.
- Swap the dataset for your own voice commands and re-run the pipeline.
- Join the [Liquid AI Discord](https://discord.com/invite/liquid-ai) to share
  what you build.

## File map

```
voice-assistant/
├── CONTEXT.md                                  glossary of domain terms
├── README.md                                   this file
├── configs/
│   ├── baseline.yaml                           eval config for the unmodified model
│   └── finetuned.yaml                          eval config for the fine-tuned model
├── docs/adr/
│   ├── 0001-eval-methodology.md                two-mode eval design
│   └── 0002-vendoring-strategy.md              why preprocess/train are vendored
├── pyproject.toml                              base deps + `finetune` group
└── scripts/
    ├── prepare_raw_data.py                     publish the 95/5 train/test split
    ├── preprocess_ohf_voice.py                 OHF-Voice → tensor dataset (vendored)
    ├── train.py                                fine-tune LFM2.5-Audio-1.5B (vendored)
    ├── eval.py                                 GGUF-based eval (server + OpenAI client)
    └── quantize.py                             safetensors → GGUF, push to HF
```
