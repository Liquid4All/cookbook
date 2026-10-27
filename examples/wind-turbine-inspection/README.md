# Wind Turbine Blade Inspection with LFM3-VL

Fine-tune [LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) to find surface defects on wind turbine blade photos and name their type. Given a blade photo, the model returns every defect it sees as JSON, with a type and a bounding box:

```json
[{"label": "oil leakage", "bbox_2d": [420, 525, 452, 625]}]
```

Coordinates are normalized to 0–1000. Training has two steps, both in Colab with LoRA:

1. **Supervised fine-tuning (SFT)** teaches the model the nine defect types and the answer format.
2. **Group Relative Policy Optimization (GRPO)** continues from the SFT adapter, with a reward computed from the predicted boxes.

A third notebook compares the fine-tuned model with the base model and frontier models on the same held-out test photos.

## Results

On the 1,372 held-out test photos:

| Model | Defect-type accuracy (95% CI) | Detection F1 (95% CI) | Box-only F1 | Output tokens per photo |
| --- | ---: | ---: | ---: | ---: |
| **LFM3-VL-2B fine-tuned (SFT + GRPO)** | **66.5%** [64.1, 69.2] | **0.106** [0.092, 0.121] | 0.130 | ~31 |
| LFM3-VL-2B base | 9.2% [7.7, 10.5] | 0.003 [0.000, 0.006] | 0.051 | ~85 |
| GPT-6 Luna | 8.5% [7.2, 9.9] | 0.010 [0.006, 0.014] | 0.058 | ~762 |

- **Defect-type accuracy**: the most frequent label among the predicted boxes matches the photo's annotated defect type. When labels tie, they share the credit.
- **Detection F1**: a predicted box counts only with the right type and an IoU of at least 0.5 with an annotated box. **Box-only F1** ignores the type.
- 95% confidence intervals come from a bootstrap over photos.

Fine-tuning teaches the model the nine defect types: type accuracy goes from 9.2% to 66.5%. Detection stays hard: the fine-tuned model finds 190 of the 2,030 annotated defects with the right type and box, and recall is lowest for tiny defects. Most of GRPO's gain over SFT comes from drawing fewer wrong boxes. The dataset only has defective blades and no written inspection notes, so it can't measure false alarms on healthy blades or grade free-text answers.

[RESULTS.md](RESULTS.md) has every checkpoint on the validation split and the GRPO training monitor.

## Files

| File | Purpose |
| --- | --- |
| [`notebooks/lfm3_vl_sft.ipynb`](notebooks/lfm3_vl_sft.ipynb) | Step 1: two epochs of LoRA fine-tuning on the annotated boxes. |
| [`notebooks/lfm3_vl_grpo.ipynb`](notebooks/lfm3_vl_grpo.ipynb) | Step 2: GRPO from the SFT adapter, with checkpoints scored on a fixed validation sample. |
| [`notebooks/evaluate.ipynb`](notebooks/evaluate.ipynb) | Score the base model, your fine-tuned adapter and frontier models on the test split. |
| [`notebooks/free_text_check.ipynb`](notebooks/free_text_check.ipynb) | Test the final model: ask for a brief written inspection finding on a few held-out photos, next to the base model's answer. |
| [`blade_eval.py`](blade_eval.py) | Dataset loader, deterministic split, prompt, model runners and scorer, shared by all the notebooks. |
| [`results/`](results/) | Scorecards for the test split, the validation split and the GRPO monitor. |
| [`.env.example`](.env.example) | API settings for the frontier models in `evaluate.ipynb`. |

## Data

The notebooks download [WTBs2025](https://doi.org/10.6084/m9.figshare.28876391) (about 480 MB, CC BY 4.0) on first run: blade photos, mostly taken by drone, with bounding boxes for nine defect types. `blade_eval.py` keeps each original photo and its augmented copies in the same split, so near-identical images never end up in both training and evaluation. It also leaves out photos that appear under two different defect types, since each folder labels only its own defect.

## Train

1. **SFT.** In Colab, pick an A100, add your Hugging Face token as the `HF_TOKEN` secret (key icon in the sidebar), and open [`lfm3_vl_sft.ipynb`](notebooks/lfm3_vl_sft.ipynb). Set `HUB_REPO` to a repository of yours, upload `blade_eval.py` when asked, and run it from the top.
2. **GRPO.** In a fresh runtime, open [`lfm3_vl_grpo.ipynb`](notebooks/lfm3_vl_grpo.ipynb), set `SFT_ADAPTER_REPO` to the adapter from step 1 and `HUB_REPO` to a new repository, and run it from the top.

Training (SFT, then GRPO) runs on a single A100; with data preparation and evaluation, the full walkthrough takes a few hours. Pick checkpoints on the 720 validation photos, and keep the test split for the end.

## Evaluate

[`evaluate.ipynb`](notebooks/evaluate.ipynb) runs LFM3-VL-2B locally (CUDA, Apple Silicon or CPU) and frontier models through any OpenAI-compatible API. To run it locally, from this folder:

```bash
cp .env.example .env  # then fill in OPENAI_API_KEY, and OPENAI_BASE_URL for a non-OpenAI endpoint
uv run --group dev jupyter lab
```

In the notebook, set `RUN` to `"lfm3_base"`, `"lfm3_finetune"` (with `LFM3_ADAPTER_ID` set to your GRPO adapter) or `"frontier"`, and `N_SAMPLES` to the number of test photos. Each run writes `notebooks/results/<run name>.jsonl`, and the last cells score every run on the same photos. To reach Gemini, Claude and GPT models with one key, point `OPENAI_BASE_URL` to a [LiteLLM](https://docs.litellm.ai/docs/simple_proxy) proxy.

## Going further

- **Change one thing at a time** and compare on the same 720 validation photos. Keep the prompt, box format and split from `blade_eval.py` fixed.
- **Track more than F1**: per-class type accuracy, F1 at IoU 0.3 and 0.5, recall by defect size, wrong boxes per photo and parse rate.
- **Check the labels with an expert**: some annotations are ambiguous, and only a wind turbine inspection specialist can settle them.
