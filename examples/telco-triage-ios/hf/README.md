---
license: other
language:
  - en
tags:
  - gguf
  - ios
  - on-device
  - rag
  - telco
  - liquid-ai
library_name: llama.cpp
---

# Telco Triage iOS Runtime Pack

This private runtime pack supports the **Telco Triage iOS** cookbook example:
an on-device telco support assistant for home-internet Q&A, cited answers,
safe local action confirmation, multi-turn follow-ups, and account or live-agent
handoff when the request should not be answered locally.

The app is designed so the model helps classify the situation, while Swift code
owns the final customer decision. Support answers are composed from canonical
RAG units and always carry a source chip or a handoff/clarification route.

## How the Runtime Works

```text
User turn
  -> ChatViewModel keeps UI state and pending actions
  -> relation classifier decides how this turn relates to prior context
  -> support classifiers identify intent, tool, cloud, and safety signals
  -> dialogue blackboard records active task, prior page, pending tool, and repairs
  -> BM25 retriever searches canonical RAG units
  -> policy engine chooses exactly one route
  -> deterministic answer composer renders from selected evidence
  -> UI shows the answer, citation, link, handoff, or confirmation
```

Normal support Q&A does **not** use free-form generation to invent answers. The
runtime retrieves a structured support unit, applies a deterministic policy, and
renders a grounded response from selected evidence. The bounded dialogue-repair
verbalizer is only used for selected repair/follow-up wording and cannot choose
routes, tools, citations, or handoffs.

## Contents

| File | Runtime role |
| --- | --- |
| `lfm25-350m-base-Q4_K_M.gguf` | LFM2.5-350M base model used for on-device inference. |
| `telco-shared-clf-v1.gguf` | Support-understanding adapter used by the customer-intent classifiers. |
| `telco-*_classifier_{weights,bias,meta}` | Classifier projections for support intent, routing lane, required tool, cloud need, issue complexity, escalation risk, PII risk, transcript quality, and slot completeness. |
| `telco-turn-relation-v4.gguf` | Stateful follow-up adapter used after there is prior chat context. |
| `telco-turn-relation_classifier_{weights,bias,meta}` | Classifier projection for follow-up, repair, confirmation, topic-switch, and escalation relation labels. |
| `telco-topic-scope_classifier_{weights,bias,meta}` | Optional off-domain scope projection. The app ignores it when absent. |
| `telco_shared_clf_schema.json` | Label schema for the support-understanding classifiers. |
| `telco-dialogue-repair-v4.gguf` | Bounded verbalizer for selected repair/follow-up turns. |
| `telco-tool-selector-v3.gguf` | Tool-support adapter for ambiguous local-action paths. |
| `rag-units-v1.json` | Canonical RAG corpus used by the retriever and answer composer. |
| `page-link-table-v1.json` | Canonical in-app links used for source chips and navigation. |
| `model_manifest.json` | Machine-readable pack contract. |
| `checksums.sha256` | SHA-256 checksums for shipped artifacts. |

The suffixes such as `v1`, `v3`, and `v4` are artifact compatibility versions,
not product tiers. They let the iOS app verify that the bundled model, classifier
projection, and schema were trained/exported for the same runtime path.

No raw source documents, customer account data, ColBERT index, legacy chat-mode
router, legacy topic gate, legacy refusal-flags adapter, or old pairwise
relation adapter are required by the current customer runtime.

## Download

```bash
hf auth login
hf download LiquidAI/TelcoTriage --local-dir models/telco
```

Then from the cookbook example:

```bash
cd examples/telco-triage-ios
TELCO_MODELS_DIR=models/telco ./bootstrap-models.sh
xcodegen generate
open TelcoTriage.xcodeproj
```

## Runtime Boundary

Normal support Q&A:

- understanding forwards: one support-understanding pass
- relation forwards: only when prior dialogue state exists
- first-turn answer generation calls: zero
- repair verbalizer calls: selected repair/follow-up acts only
- retriever: BM25 over canonical structured support units
- citation source: selected RAG unit canonical URL
- route owner: Swift policy engine

Unsupported or account-specific requests are handled as clarification, account
handoff, cloud/system handoff, or live-agent escalation. The pack does not
include customer account data.

## Quality Gates

The cookbook app includes focused tests for:

- BM25 retrieval parity against the canonical RAG fixture
- deterministic answer composition
- dialogue blackboard and state-operation resolution
- policy routing for grounded answers, tool confirmation, handoff, and decline
- opt-in phone-flow situation validation via `PHONE_FLOW_EVAL=1`

These gates are situation-quality checks: they evaluate whether the app made the
right decision for the turn, not whether it exactly reproduced one transcript.

## Expected Repository Files

The Hugging Face repository should contain only:

- `README.md`
- `checksums.sha256`
- `lfm25-350m-base-Q4_K_M.gguf`
- `telco-shared-clf-v1.gguf`
- `telco-turn-relation-v4.gguf`
- `telco-dialogue-repair-v4.gguf`
- `telco-tool-selector-v3.gguf`
- `telco-*_classifier_{weights,bias,meta}`
- `telco_shared_clf_schema.json`
- `rag-units-v1.json`
- `page-link-table-v1.json`
- `model_manifest.json`

## Access

This pack is intended for private proof-of-concept delivery. Grant access through
the Hugging Face organization or use a gated/private repo. Uploads should use a
write token or a fine-grained token scoped to this repository.
