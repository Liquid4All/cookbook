# Model Architecture

Telco Triage shows how a mobile app can use Liquid Foundation Models as a
private first-response layer for carrier support. The design is intentionally
small: one resident base model, a set of task adapters, and lightweight
classifier heads that turn natural language into auditable support decisions.

The goal is not to replace every backend workflow. It is to keep high-volume,
low-risk support interactions on device, and to hand off only the cases that
need live carrier systems such as outage, account, billing, appointment, or
inventory services.

## Runtime Stack

| Layer | Artifact | Role |
| --- | --- | --- |
| Resident base model | `lfm25-350m-base-Q4_K_M.gguf` | On-device LFM loaded once at app launch |
| Shared classifier adapter | `telco-shared-clf-v1.gguf` | Shapes the hidden state for all telco routing heads |
| Chat-mode adapter | `chat-mode-router-v2.gguf` | Classifies the customer-visible mode boundary: KB question, tool action, personal summary, or out of scope |
| Tool adapter | `telco-tool-selector-v3.gguf` | Selects the local tool and extracts tool arguments |
| KB fallback adapter | `kb-extractor-v1.gguf` | Legacy generative KB selector retained for comparison |
| Classifier heads | `*_classifier_weights.bin`, `*_classifier_bias.bin`, `*_classifier_meta.json` | Small linear projections over the LFM hidden state |
| Knowledge base | `knowledge-base.json` | Carrier-agnostic home internet support corpus |

All adapters are trained against **LFM2.5-350M Base**. They should not be
applied to DPO or instruction-tuned weights unless they are retrained against
that exact base distribution.

## Decision Vector

The main runtime pattern is a shared forward pass followed by multiple small
heads. The app produces a typed `TelcoDecisionVector` for trace, cloud
requirements, privacy, escalation, and tool hints. It then asks the dedicated
`chat-mode-router-v2` adapter for the customer-visible chat branch. This keeps
model responsibility aligned with training data: ADR-015 owns telco support
signals, while the chat-mode LFM owns the question-versus-action boundary.

| Head | Output | Why it exists |
| --- | --- | --- |
| `support_intent` | Troubleshooting, outage, billing, appointment, setup, plan/account, returns, handoff | Explains the customer goal |
| `routing_lane` | Local answer, local tool, cloud assist, human escalation, blocked | Chooses the support lane |
| `required_tool` | Restart, diagnostics, speed test, technician scheduling, no tool, cloud only | Selects a local action when one is appropriate |
| `issue_complexity` | Simple, guided, multi-step, backend required, human required | Feeds trace and UX hints |
| `cloud_requirements` | Live network, account state, billing record, appointment system, inventory, plan catalog, auth | Describes what a cloud handoff would need |
| `customer_escalation_risk` | Low, frustrated, churn risk, complaint, urgent | Keeps the automation boundary customer-aware |
| `pii_risk` | Safe, account data, contact data, payment/identity data | Controls redaction and egress posture |
| `transcript_quality` | Clean, noisy, partial, ASR uncertain | Gives voice workflows a confidence signal |
| `slot_completeness` | Missing device, symptom, duration, location, auth, contact preference | Identifies clarifying questions before escalation |

The heads are tiny compared with the base model. They are committed with the
sample app because they make the architecture concrete and inspectable without
turning the repository into a model-weight distribution channel.

## RAG On Device

The primary RAG path is deliberately practical. `KeywordKBExtractor` retrieves
from a small local support corpus using curated aliases and topic terms, then
the resident LFM writes the final answer from the selected article.

That choice is intentional. For compact carrier FAQs, aliases such as
`pause internet`, `ssid`, `router lights`, and `slow bedroom wifi` are strong
human-authored retrieval signals. Keyword/BM25-style retrieval is fast,
deterministic, easy to debug, and resilient to classifier retraining. Embedding
retrieval remains useful for larger or less-curated corpora, but the best
production pattern is usually hybrid: lexical precision first, embedding
fallback for paraphrase.

The important boundary is that retrieval and generation are separate:

1. The classifier decides whether the query belongs on the knowledge lane.
2. The retriever selects the local article.
3. The LFM turns that article into a concise support response.
4. The UI shows source metadata and pipeline trace details.

## Tool And Cloud Boundaries

The same model stack also supports local agentic actions. When the routing lane
is `local_tool`, the app proposes a tool such as speed test, diagnostics,
restart, WPS pairing, extender reboot, parental controls, or technician
scheduling. User-visible confirmation sits between the model decision and any
state-changing action.

When the routing lane is `cloud_assist`, the demo prepares a redacted payload
instead of silently calling a backend. In a carrier integration, that payload is
the contract to live systems of record. The cloud path is reserved for data the
phone cannot know locally: outage state, authenticated account status, billing
records, device inventory, appointment availability, or plan catalog changes.

## Optional Packs

The app includes extension points for richer multimodal support:

| Pack | Role |
| --- | --- |
| LFM2.5-Audio | On-device voice transcription and future end-to-end voice assistance |
| LFM2.5-VL-style vision | Router-light, bill, and error-screen troubleshooting |

These packs are optional in this sample. The core architecture works with text,
local KB retrieval, classifier heads, and local tools.

## Model Distribution

Large GGUF artifacts are intentionally excluded from the public cookbook
repository. The repository carries the app source, sample KB, manifests, and
small classifier heads. Full model artifacts belong in a versioned model
registry such as Hugging Face Hub or an internal Liquid/customer registry.

Recommended customer distribution patterns:

| Use case | Pattern |
| --- | --- |
| Public source example | Keep GGUFs out of Git; document expected filenames and checksums |
| Hands-on demo | Ship a signed TestFlight, IPA, or release artifact with models bundled |
| Private customer pilot | Use a gated model registry with pinned revisions and SHA-256 checksums |
| Enterprise fork | Host customer-specific adapters in the customer's approved artifact store |

Git LFS can be useful in private engineering repos, but it is not the best
default for a public cookbook example. It makes every clone depend on model
download bandwidth and account quota, while model registries provide clearer
versioning, access control, and model-card metadata.

## Local Artifact Layout

The sample app expects GGUFs under:

```text
examples/telco-triage-ios/models/telco/
```

`bootstrap-models.sh` copies those files into `TelcoTriage/Resources/Models/`
before `xcodegen generate`. `TELCO_MODELS_DIR` can point at another local model
cache when artifacts are managed outside the repository.
