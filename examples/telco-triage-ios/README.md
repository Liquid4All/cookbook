# Telco Triage iOS

Telco Triage is a SwiftUI reference app for a private, on-device home internet
support assistant powered by Liquid Foundation Models.

The app demonstrates an edge-first support architecture:

1. A compact LFM runs in the iOS app.
2. A shared telco classifier adapter produces nine support decisions in one
   forward pass.
3. A deterministic router decides whether to answer locally, propose a local
   tool, use cloud assist, escalate to a human, or block.
4. Cloud assist receives only a redacted support bundle, and only when the
   workflow needs live account, billing, outage, or appointment systems.

This is a carrier-agnostic example. The Xcode project, scheme, app target,
source module, bundle ID, and visible app name are all generic Telco Triage
values.

## What This Shows

- Real LFM inference on device, not a scripted frontend.
- One resident LFM2.5-350M base model with LoRA adapters.
- Nine ADR-015 classifier heads for support routing, cloud requirements,
  escalation risk, PII risk, tool selection, transcript quality, and slot
  completeness.
- Local support tools such as restart router, speed test, diagnostics, WPS,
  extender reboot, and technician scheduling.
- A visible pipeline trace for model latency, confidence, route, and tool
  selection.
- Optional audio and vision pack scaffolding for voice support and visual
  troubleshooting.

## Architecture

```text
User text, voice transcript, or image
        |
        v
TelcoTopicGate
        |
        v
LFM2.5-350M base + telco-shared-clf-v1 LoRA
        |
        v
TelcoDecisionVector
        |
        v
TelcoDecisionRouter
        |
        +--> Local KB answer
        +--> Local tool proposal and confirmation
        +--> Cloud assist with redacted payload
        +--> Human escalation
        +--> Local refusal / blocked path
```

The model emits typed signals. The router owns the policy decision. This keeps
the agentic behavior auditable and testable.

## Model Artifacts

Large GGUF files are intentionally not committed to the cookbook repository.
The small classifier head files and metadata are committed under
`TelcoTriage/Resources/`.

Recommended distribution pattern:

1. Keep source, manifests, sample KB, and small classifier heads in Git.
2. Host full GGUF model/adaptor artifacts in a versioned model registry such
   as Hugging Face Hub, gated if the license or customer terms require it.
3. Pin the exact model revision and checksums in release notes or an internal
   manifest before sharing a customer build.
4. Use `bootstrap-models.sh` to copy the downloaded artifacts into the app
   bundle before `xcodegen generate`.

Required local GGUFs:

| File | Purpose |
| --- | --- |
| `lfm25-350m-base-Q4_K_M.gguf` | Resident LFM2.5-350M base model |
| `telco-shared-clf-v1.gguf` | Shared classifier LoRA for the nine telco heads |
| `telco-tool-selector-v3.gguf` | Tool argument and selection adapter |
| `chat-mode-router-v2.gguf` | Generative fallback router |
| `kb-extractor-v1.gguf` | Grounded KB answer adapter |

Optional transitional adapters:

- `chat-mode-clf-v1.gguf`
- `kb-extract-clf-v1.gguf`
- `tool-selector-clf-v1.gguf`

Put these files in `examples/telco-triage-ios/models/telco/`, or set
`TELCO_MODELS_DIR` to a directory containing them.

## Build

Requirements:

- Xcode 15+
- iOS 17+ simulator or device
- `xcodegen`

Install XcodeGen if needed:

```bash
brew install xcodegen
```

Prepare and open the app:

```bash
cd examples/telco-triage-ios

# Option A: models live in ./models/telco
./bootstrap-models.sh

# Option B: models live elsewhere
TELCO_MODELS_DIR=/path/to/telco-models ./bootstrap-models.sh

xcodegen generate
open TelcoTriage.xcodeproj
```

Then run the `TelcoTriage` scheme. The display name on device is
`Telco Triage`.

## Test

Fast non-LFM unit tests:

```bash
cd examples/telco-triage-ios
xcodegen generate
xcodebuild test \
  -project TelcoTriage.xcodeproj \
  -scheme TelcoTriage \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -skip-testing:TelcoTriageTests/LFMValidationTests \
  -skip-testing:TelcoTriageTests/LlamaBackendSmokeTests
```

Full tests require the GGUFs to be copied with `./bootstrap-models.sh`.

## Demo Prompts

Try these in the Chat tab:

```text
Restart my router
Run a speed test
My wifi is slow in the bedroom
What do the lights on my router mean?
Block my son's tablet from the internet
Is there an outage in my area?
Why is my bill higher this month?
I want to talk to a person
```

In engineering mode, expand the trace card to show which decisions came from
the local model, which tool was selected, and whether the request stayed local
or moved to cloud assist.

## Customize

- Replace `TelcoTriage/Resources/knowledge-base.json` with your carrier
  support corpus.
- Add or edit themes in `TelcoTriage/Core/Branding/`.
- Add tools under `TelcoTriage/Core/Tools/` and register them in
  `ToolRegistry`.
- Retrain the shared classifier adapter if you change the support taxonomy,
  tool catalog, or cloud-assist policy labels.

## Notes

- Use the **Base** LFM2.5-350M GGUF. The included LoRA adapters were trained
  against Base weights.
- The iOS Simulator runs without GPU offload, so physical devices are more
  representative for latency.
- The project and module are named `TelcoTriage`; carrier-specific forks can
  keep that stable or rename it deliberately with XcodeGen.
