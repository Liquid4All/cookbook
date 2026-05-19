# LFM-VL Methane Plume Extraction Demo

This cookbook example packages a local demo for methane plume triage in oil and
gas operations. It shows the structured-output workflow:

```text
rendered methane observation image + fixed customer schema -> JSON record
```

The demo compares the released base model against a methane-specialized fine-tune.
The fine-tuned model was adapted with LEAP fine-tune so the model follows the
schema used by this demo and returns one flat JSON object that an LDAR,
emissions-monitoring, or operations workflow can route downstream.

## Why This Demo Exists

Methane monitoring today often produces images, not decisions. Airborne,
satellite, drone, and site imagery still require reviewers to inspect each
observation, filter false alarms such as steam or surface artifacts, and decide
which events deserve repair-team follow-up.

Traditional detection or segmentation models can highlight pixels that look like
a plume. This demo shows the VLM layer on top: plume status, confidence,
lookalike risk, timestamp/source context, evidence summary, alignment, and
review priority in a customer-shaped JSON schema.

## What Is Included

- A browser demo with curated sample observations and ground-truth assessment.
- A local inference server for the base and fine-tuned models.
- The fixed methane schema used for prompting, evaluation, and demo output.
- A model download helper for Hugging Face-hosted checkpoints.
- A [training recipe](training/README.md) with the exact schema, LEAP config,
  dataset manifest, and sample row format used for the demo checkpoint.
- `AGENTS.md` and `agent.md` runbooks for another agent or teammate opening the demo.

This example intentionally does not include model weights. The base model and
fine-tuned private model are downloaded from Hugging Face.

## Model Access

The base model is:

```text
LiquidAI/LFM2.5-VL-1.6B
```

The fine-tuned model should be hosted as this private Hugging Face model repo:

```text
felipeliquid/LFM2.5-1.6B-VL-Extract-Plume-Demo
```

Anyone running the demo needs a Hugging Face token with read access to the
private fine-tuned model. See [MODEL_ACCESS.md](MODEL_ACCESS.md).

## Prompt Shape And Schema

This fine-tuned demo should be run with the same schema-conditioned prompt shape
used during training and evaluation:

```text
Extract the following from the image:

<schema.yaml contents>
```

The model is expected to return exactly one flat JSON object with the schema
keys below. For this demo, treat the schema as fixed: small wording edits to
field descriptions may still work, but changing key names, adding/removing
fields, changing allowed values, or moving to nested JSON should be treated as a
new schema and re-evaluated before showing it to a customer.

Exact schema:

```yaml
methane_plume_status: Exact methane plume assessment from the rendered image using one of present, absent, or uncertain.
plume_confidence: Confidence in the plume assessment using one of low, medium, or high.
plume_extent: Approximate visible plume extent in the methane product using one of none, small, medium, or large.
plume_location: Approximate visible plume location in the image using one of top_left, top_right, center, bottom_left, bottom_right, or unknown.
lookalike_risk: Most likely non-methane visual explanation using one of steam, dust, cloud_shadow, surface_artifact, sensor_artifact, albedo_confuser, none, or unknown.
visual_evidence_summary: One short image-only sentence describing the visible evidence without source metadata or operational recommendations.
observation_timestamp_utc: Observation timestamp shown in the observation context panel as ISO-8601 UTC, or unknown when unavailable.
source_context: Supplied source type shown in the observation context panel using one of well_pad, compressor_station, pipeline_corridor, processing_facility, landfill, mine, agriculture, unknown, or not_applicable.
source_alignment_assessment: Whether visible plume evidence aligns with supplied source context using one of aligned, source_context_unavailable, no_visible_plume, or unknown.
review_priority: Human review priority for the observation using one of low, medium, or high.
```

## Quickstart

Run from `examples/methane-plume-extraction/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install the Hugging Face CLI if needed:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
hf auth login
hf auth whoami
```

Download both models:

```bash
./scripts/download_models.sh
```

If the private model is hosted under a different namespace, override it with
`FINE_MODEL_ID`.

Start local inference. On Apple Silicon, the default `transformers` backend uses
MPS automatically.

```bash
python scripts/serve_local_methane.py --preload fine
```

In a second terminal:

```bash
source .venv/bin/activate
METHANE_INFER_URL=http://127.0.0.1:8791/infer \
  python scripts/serve_demo.py
```

Open:

```text
http://127.0.0.1:8787/demo
```

Select a sample image, choose Base or Fine-tuned, then click Run. The output is
generated live by the local model server.

## Demo Flow

1. The page loads curated rendered methane observations.
2. The page sends the selected image and the fixed schema to `/api/infer`.
3. `scripts/serve_demo.py` proxies the request to the local inference server.
4. `scripts/serve_local_methane.py` runs the selected model and returns:

```json
{
  "prediction_json": {
    "methane_plume_status": "present",
    "plume_confidence": "high",
    "plume_extent": "medium",
    "plume_location": "center",
    "lookalike_risk": "none",
    "visual_evidence_summary": "Coherent plume-like enhancement is visible in the methane product panel.",
    "observation_timestamp_utc": "2019-10-21T18:19:27Z",
    "source_context": "well_pad",
    "source_alignment_assessment": "aligned",
    "review_priority": "high"
  },
  "latency_ms": 2561
}
```

The demo page then compares the JSON against held-out ground truth for built-in
samples. Uploaded customer images run through the same schema, but they do not
have ground truth unless you add it.

## Fine-Tune Context

The demo fine-tune used public methane imagery and replay data:

- STARCOP / AVIRIS-NG style rendered methane plume products.
- MethaneSET-style Sentinel-2, Landsat 8/9, and EMIT observations.
- JPL AVIRIS-NG methane benchmark style examples.
- Replay rows to reduce regression against the base structured-output behavior.

Latest demo training package:

```text
20,650 rows total
17,234 methane/context rows
3,416 replay rows
1h 28m 47s training on 1x NVIDIA H100 after data upload
~$5.84 training cost + $0 upload cost for the latest run
```

This is a controlled demo artifact, not a production methane-compliance system.
Production deployment should use customer-owned imagery, customer-approved
labels, held-out evals, and the customer schema.

## Files

| Path | Purpose |
|---|---|
| `interactive_demo.html` | Self-contained browser UI with embedded sample images and ground truth. |
| `schema.yaml` | Fixed flat schema sent to the model. |
| `scripts/serve_demo.py` | Static web server plus `/api/infer` endpoint bridge. |
| `scripts/serve_local_methane.py` | Local base/fine model inference server. |
| `scripts/download_models.sh` | Downloads base and fine-tuned models from Hugging Face. |
| `AGENTS.md` | Canonical agent runbook for opening and validating the demo. |
| `agent.md` | Short human-facing local-demo shortcut. |

## Backend Notes

The default path uses Hugging Face Transformers because it matched the evaluated
fine-tuned checkpoint during local testing. MLX can be tried with
`--backend mlx`, but validate output parity before using MLX in a customer
meeting.

Model directories expected by default:

```text
models/base
models/fine
```

You can override them:

```bash
python scripts/serve_local_methane.py \
  --base-model /path/to/base \
  --fine-model /path/to/fine \
  --preload fine
```

## Troubleshooting

- If downloads fail with `401` or `403`, the token does not have access to the
  private fine-tuned model.
- If `AutoModelForImageTextToText` is missing, upgrade `transformers`.
- If the browser says the endpoint is unavailable, confirm both servers are
  running and that `METHANE_INFER_URL` points to `http://127.0.0.1:8791/infer`.
- CPU inference works but is slow. Apple Silicon MPS is the recommended local
  path for this package.
