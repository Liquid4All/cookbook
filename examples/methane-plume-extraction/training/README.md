# Fine-Tuning Recipe

This folder documents the training recipe used to produce the private demo
checkpoint:

```text
felipeliquid/LFM2.5-1.6B-VL-Extract-Plume-Demo
```

The runtime demo is intentionally "give them the fish"; this folder is the
"teach them to fish" part. It shows the schema, dataset shape, manifest, and
LEAP fine-tuning config used for the demo checkpoint.

## What Was Trained

The task is schema-conditioned visual extraction:

```text
rendered methane observation image + fixed schema -> one flat JSON object
```

Base model:

```text
LiquidAI/LFM2.5-VL-1.6B
```

Final training mix:

| Component | Rows |
|---|---:|
| Methane/context rows | 17,234 |
| Structured-output replay rows | 3,416 |
| Total | 20,650 |

The methane rows used the fixed schema in [`schema.yaml`](schema.yaml). Replay
rows came from other strict-JSON VLM extraction tasks and were mixed in to
reduce regression on general structured-output behavior.

## Direct Reproduction Path

This recipe is written as the final path. To reproduce the demo checkpoint
process, start from the schema in this folder and build toward the final bundle
shape directly:

1. Acquire source imagery from STARCOP and MethaneSET, or replace it with
   customer-owned imagery that you are allowed to train on.
2. Render each source observation into the same image contract used by the demo:
   the methane product, visible/context panel, timestamp, and source context in
   one image.
3. Label each rendered image against [`schema.yaml`](schema.yaml) so every row
   has exactly the schema keys. Do not add intermediate schema versions, nested
   JSON, or renamed fields.
4. Write methane labels in the eval-style JSONL contract shown below.
5. Mix in structured-output replay rows if you want to preserve general JSON
   extraction behavior.
6. Run `prepare_modal_training_bundle_with_replay.py` to create the final LEAP
   message JSONL plus staged images.
7. Upload `train.jsonl` and `images/` to the Modal volume path used by
   [`leap_finetune_modal.yaml`](leap_finetune_modal.yaml), then run LEAP
   fine-tune.
8. Evaluate the trained checkpoint against held-out core and stress examples
   before using it in the browser demo.

Because this cookbook does not redistribute derived imagery or the private
20,650-row JSONL, exact byte-for-byte reproduction requires regenerating those
assets from licensed sources or using an approved internal copy of the training
bundle. The process, schema, row format, final mix, and training config are
documented here so the run can be recreated from the final schema and bundle
contract.

## Public Data Sources

The checkpoint was built from public methane remote-sensing sources rendered
into demo-friendly observation images.

| Source | How it was used | License / note |
|---|---|---|
| [STARCOP](https://zenodo.org/records/7863343) | Main absent/uncertain set and some present examples using AVIRIS-NG / simulated multispectral products | Zenodo lists CC-BY-NC-4.0. Do not redistribute derived imagery without checking usage. |
| [MethaneSET](https://huggingface.co/datasets/tacofoundation/methaneset) | Present examples from Sentinel-2, Landsat 8/9, and EMIT-derived products | Hugging Face lists cc-by-nc-sa-4.0. Do not redistribute derived imagery without checking usage. |
| [JPL AVIRIS-NG CH4/CO2 benchmark](https://avirisng.jpl.nasa.gov/benchmark_methane_carbon_dioxide.html) | Used in source/eval exploration and held-out demo evaluation; not counted as a final training source in the manifest | JPL page describes ten AVIRIS-NG scenes with methane and carbon dioxide point-source emissions. |

The cookbook does **not** redistribute source imagery, rendered train images, or
the 20,650-row training JSONL. Users should regenerate their own dataset from
source data they are allowed to use, or use their own customer imagery.

## Exact Dataset Manifest

[`training_manifest.json`](training_manifest.json)
records the exact final training mix, checksums, hyperparameters, and eval
scores. The key values are:

```text
train.jsonl SHA-256: b95c62e532769f8070fd1dcaa94e5207e8392389194f8f7add63ec36fc84e904
methane labels SHA-256: 97a439af58bf985a23fa6505fcae2237affd44af02030232d2a71b446b4eb3fa
schema SHA-256: dd7ddbdf57b272e71e29c6a86af98156bf2f158f5a244880300fe0de43066cfe
```

Final methane row counts:

| Field | Counts |
|---|---|
| `methane_plume_status` | absent 5,750; present 9,734; uncertain 1,750 |
| `source_dataset` | STARCOP 10,904; MethaneSET 6,146; unknown/context-extra 184 |
| `source_context` | not_applicable 5,750; unknown 3,430; well_pad 3,070; landfill 945; mine 839; agriculture 800; processing_facility 800; pipeline_corridor 800; compressor_station 800 |

The `unknown/context-extra` bucket is preserved from the original manifest. Most
of those rows were MethaneSET EMIT-derived context examples whose
`source_dataset` field was not normalized in the local manifest.

## Training Row Format

LEAP VLM SFT consumed a JSONL file where each row contains a `sample_id` and
`messages`. See [`sample_train_row.json`](sample_train_row.json).

The important shape is:

```json
{
  "sample_id": "methane:...",
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "Extract the following from the image:\n\n<schema.yaml contents>"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": "/outputs/.../image.png"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "{\"methane_plume_status\":\"absent\",...}"
        }
      ]
    }
  ]
}
```

The assistant text is a compact JSON string, not a nested object.

## Preparing A Training Bundle

If you already have labels in this local eval-style format:

```json
{
  "image": "/absolute/path/to/rendered_observation.png",
  "schema": {"methane_plume_status": "..."},
  "ground_truth": {"methane_plume_status": "present"},
  "metadata": {"sample_id": "source:id", "source_dataset": "starcop"}
}
```

you can convert them into the LEAP VLM SFT message format with:

```bash
cd examples/methane-plume-extraction

python training/scripts/prepare_modal_training_bundle_with_replay.py \
  --schema training/schema.yaml \
  --labels-jsonl /path/to/methane_train_observation_17234.jsonl \
  --bundle-root /path/to/local_bundle_root \
  --dataset-name methane_plume_observation_train17234_replay20650 \
  --remote-root /outputs/methane_plume_observation_train17234_replay20650 \
  --min-samples 10000 \
  --replay real_iad:/path/to/real_iad_train.jsonl:1500 \
  --replay visa:/path/to/visa_train.jsonl:1500 \
  --replay forge:/path/to/forge_train.jsonl:416
```

This writes:

```text
/path/to/local_bundle_root/methane_plume_observation_train17234_replay20650/train.local.jsonl
/path/to/local_bundle_root/methane_plume_observation_train17234_replay20650/train.jsonl
/path/to/local_bundle_root/methane_plume_observation_train17234_replay20650/manifest.json
/path/to/local_bundle_root/methane_plume_observation_train17234_replay20650/images/
```

`train.local.jsonl` points at local image paths. `train.jsonl` points at the
Modal volume paths used by the LEAP config.

The helper enforces the final schema directly: every methane row must have
exactly the keys in `schema.yaml`, and the assistant target is written as a
compact JSON string.

## Fine-Tuning With LEAP

The exact config used for the demo checkpoint is
[`leap_finetune_modal.yaml`](leap_finetune_modal.yaml).

After staging `train.jsonl` and its `images/` directory into the Modal volume,
the training container must see this path:

```text
/outputs/methane_plume_observation_train17234_replay20650/train.jsonl
```

If the LEAP Modal app mounts the `leap-finetune` volume at `/outputs`, upload
the prepared dataset directory to the matching path inside that volume:

```bash
modal volume put leap-finetune \
  /path/to/local_bundle_root/methane_plume_observation_train17234_replay20650 \
  /methane_plume_observation_train17234_replay20650
```

Then run LEAP fine-tune from a checkout with VLM SFT support:

```bash
cd examples/methane-plume-extraction
leap-finetune training/leap_finetune_modal.yaml
```

The demo run used:

```text
1x NVIDIA H100 on Modal
1 epoch
2,529 steps
LoRA r=16, alpha=32
learning rate 2e-5
runtime 1h 28m 47s after data upload
final train loss 0.04211
final eval loss 0.01687
```

## Eval Summary

The held-out evals are demo-curated labels, not customer-adjudicated production
gold.

| Split | Samples | JSON valid | Strict value exact | LLM judge | VLM judge |
|---|---:|---:|---:|---:|---:|
| Core | 235 | 100.0% | 94.4% | 0.959 | 0.889 |
| Stress | 265 | 100.0% | 77.5% | 0.821 | 0.826 |

Known limitation: ambiguous plume cases remained the hardest category on stress
examples and should be strengthened before a customer production deployment.

## Reproducibility Notes

- This folder documents the checkpoint recipe; it does not make the private
  model or the source datasets public.
- The final training labels were demo-curated labels. For a customer or
  production system, replace them with customer-approved human labels.
- The rendered context panel is demo metadata. It should not be described as
  independent GIS/source attribution.
- Before sharing rendered images or derived labels, review the source licenses
  and attribution requirements.
