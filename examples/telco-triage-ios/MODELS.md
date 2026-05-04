# Models

Telco Triage demonstrates an edge-first Liquid model stack for carrier support.
Large GGUF files are intentionally excluded from the cookbook repository; use
`bootstrap-models.sh` to copy them into the app bundle before generating the
Xcode project.

## Runtime Stack

| Layer | Artifact | Role |
| --- | --- | --- |
| Base model | `lfm25-350m-base-Q4_K_M.gguf` | Resident LFM2.5-350M base model |
| Shared classifier | `telco-shared-clf-v1.gguf` | One LoRA that drives the nine ADR-015 classifier heads |
| Tool adapter | `telco-tool-selector-v3.gguf` | Generates/extracts tool arguments when a local tool path is selected |
| Chat router fallback | `chat-mode-router-v2.gguf` | Generative fallback when classifier paths are unavailable |
| KB extractor | `kb-extractor-v1.gguf` | Grounded answer extraction from the local support KB |

Optional transitional classifier adapters:

- `chat-mode-clf-v1.gguf`
- `kb-extract-clf-v1.gguf`
- `tool-selector-clf-v1.gguf`

## Recommended Distribution

Do not commit the full GGUF model artifacts to the public cookbook example.
Keep the repository small and reviewable:

- Commit app source, model manifests, checksums, sample data, and the small
  classifier head binaries/metadata.
- Host large GGUFs in a versioned model registry such as Hugging Face Hub.
- Use gated model access for non-public demos, restricted licenses, early
  research weights, or customer-specific adapters.
- Pin a model revision or commit hash, and publish SHA-256 checksums for every
  GGUF used in a shared customer build.
- For a zero-friction hands-on demo, distribute a signed TestFlight, IPA, or
  release artifact with the models already bundled. Keep source clones light.

Git LFS can work for private engineering repos, but it is a poor default for a
public cookbook sample because every clone becomes a model-distribution event
and LFS bandwidth/account limits become part of the customer setup path.

## Nine Telco Classifier Heads

The shared classifier adapter feeds these small committed head artifacts:

| Head | Purpose |
| --- | --- |
| `support_intent` | Troubleshooting, billing, outage, appointment, account, handoff |
| `issue_complexity` | Simple, guided, backend-required, human-required |
| `routing_lane` | Local answer, local tool, cloud assist, human escalation, blocked |
| `cloud_requirements` | Live network, account, billing, appointment, inventory, catalog, auth |
| `required_tool` | Restart, diagnostics, speed test, technician scheduling, cloud-only |
| `customer_escalation_risk` | Frustration, churn risk, complaint, urgent |
| `pii_risk` | Whether egress needs redaction or blocking |
| `transcript_quality` | Clean, noisy, partial, uncertain |
| `slot_completeness` | Missing symptom, device, location, duration, auth, contact preference |

The key architectural idea is that one local forward pass produces a typed
decision vector. The app does not ask a generative model to decide whether it
should call a tool or cloud. It uses a deterministic router over model-produced
signals.

## Why This Matters

Carrier support contains many high-volume requests that do not need cloud
reasoning:

- Restart or diagnose local equipment.
- Explain common router lights.
- Walk through Wi-Fi setup.
- Answer support KB questions.
- Collect missing slots before an escalation.

Keeping those interactions local reduces latency and backend load. When a
request needs live network status, authenticated account state, billing records,
appointment systems, or human escalation, the app can hand off a redacted
structured bundle to cloud assist.

## Audio And Vision Packs

The app includes scaffolding for:

- **LFM2.5-Audio-1.5B** for stronger on-device voice transcription.
- **LFM2.5-VL** style visual troubleshooting for router lights, bill images,
  and error screens.

Those packs are optional in this example. The core Telco Triage path is text
plus on-device support routing.

## Model Placement

Put GGUFs here:

```text
examples/telco-triage-ios/models/telco/
```

or point the bootstrap script at another directory:

```bash
TELCO_MODELS_DIR=/path/to/telco-models ./bootstrap-models.sh
```

Then run:

```bash
xcodegen generate
```

XcodeGen includes whatever GGUFs are present under
`TelcoTriage/Resources/Models/` at generation time.
