# Telco Triage iOS Integration Guide

This guide explains how to integrate the Telco Triage reference architecture
into an existing iOS app. It is intended for teams that want the same control
plane as the cookbook app: on-device understanding, local RAG, deterministic
policy, safe tool confirmation, and grounded answer composition.

For the conceptual overview, read [ARCHITECTURE.md](ARCHITECTURE.md) first.

## Recommended Integration Target

Do not copy the entire demo app into your product app.

Instead, port the text-support runtime path:

```text
user message
  -> turn relation
  -> shared support understanding
  -> dialogue blackboard
  -> BM25 retrieval over rag-units-v1.json
  -> TelcoPolicyEngine
  -> DeterministicAnswerComposer
  -> TelcoSupportSession result
  -> host app chat UI / source chip / confirmation UI
```

This gives you the behavior demonstrated in Telco Triage without bringing over
demo tabs, engineering panels, voice, vision, or sample household screens.

## Required vs Optional Pieces

### Required for Text Support + Tool Control Plane

| Piece | Purpose |
| --- | --- |
| `Packages/LFMEngine/Sources/LFMEngine` | Local llama.cpp-backed runtime, adapter loading, classifier heads. |
| `TelcoTriage/Core/Model` | Adapter backend, model bundle names, model provider bridge. |
| `TelcoTriage/Core/Orchestration` | Headless `TelcoSupportSession` API. This is the preferred integration boundary. |
| `TelcoTriage/Core/Intelligence` | Dispatcher, shared understanding, policy, support lanes, heuristics. |
| `TelcoTriage/Core/Intelligence/Understanding` | Turn relation, blackboard, state operation resolver, conversation state. |
| `TelcoTriage/Core/RAG` | Structured RAG unit schema and corpus loader. |
| `TelcoTriage/Core/Retrieval/BM25HierarchyRetriever.swift` | Local RAG ranker used by the composer path. |
| `TelcoTriage/Core/Composer` | Deterministic answer rendering from policy + evidence. |
| `TelcoTriage/Core/Tools` | Typed tool protocol and registry. Register host-owned tools here. |
| `TelcoTriage/Core/Routing` | Tool intent, tool alias map, tool execution/result types. |
| `TelcoTriage/Core/Privacy` | Optional deterministic PII masking/scanning helpers. |
| `TelcoTriage/Core/Observability` | Lightweight app logging used by runtime code. |
| `TelcoTriage/Resources/rag-units-v1.json` | Canonical support corpus. Replace with your own corpus for production. |
| classifier head files | Required for shared understanding and turn relation. |
| `TelcoTriage/Resources/Models/*.gguf` | Base model and telco adapters. |

### Optional Demo Surfaces

These are useful in the sample app, but are not required for a first
integration:

| Optional piece | Include only if |
| --- | --- |
| `TelcoTriage/Features/Chat` | You want to reuse the demo chat UI. Most apps should use their own UI. |
| `TelcoTriage/Features/Branding` | You want the cookbook Liquid/Verizon demo styling. |
| `TelcoTriage/Features/DemoCustomer` | You want the sample household/device state. Production apps should use their own customer state. |
| `TelcoTriage/Features/DemoTools` | You want the sample tool implementations. Production apps should implement `Tool` against their own APIs. |
| `TelcoTriage/Features/Plan`, `Packs`, `Starters` | You want the cookbook demo shell. |
| `TelcoTriage/Core/Voice` | You want the audio/voice path. Requires LeapSDK. |
| `TelcoTriage/Core/Vision` | You want vision-language experiments. Requires MLX packages. |
| `TelcoTriage/Features/Metrics` | You want the device-impact/telemetry tab. |
| engineering/eval views | You want debug traces and harness screens in your dev build. |

## Swift Package Dependencies

For the core text/RAG path:

```yaml
llama.swift:
  url: https://github.com/mattt/llama.swift.git
  exactVersion: "2.8851.0"
```

Optional:

```yaml
mlx-swift-lm:
  url: https://github.com/ml-explore/mlx-swift-lm.git
  exactVersion: "2.30.6"

LeapSDK:
  url: https://github.com/Liquid4All/leap-ios.git
  from: "0.9.4"
```

Use `mlx-swift-lm` only if you are including the vision path. Use `LeapSDK`
only if you are including the audio/voice path. The text-support control plane
does not require either dependency.

## Required Model and Resource Bundle

Expected bundle layout:

```text
Resources/
  rag-units-v1.json
  page-link-table-v1.json
  telco_shared_clf_schema.json
  telco-support-intent_classifier_weights.bin
  telco-support-intent_classifier_bias.bin
  telco-support-intent_classifier_meta.json
  telco-issue-complexity_classifier_*.bin/json
  telco-routing-lane_classifier_*.bin/json
  telco-cloud-requirements_classifier_*.bin/json
  telco-required-tool_classifier_*.bin/json
  telco-customer-escalation-risk_classifier_*.bin/json
  telco-pii-risk_classifier_*.bin/json
  telco-transcript-quality_classifier_*.bin/json
  telco-slot-completeness_classifier_*.bin/json
  telco-turn-relation_classifier_*.bin/json
  Models/
    lfm25-350m-base-Q4_K_M.gguf
    telco-shared-clf-v1.gguf
    telco-turn-relation-v4.gguf
    telco-tool-selector-v3.gguf
    telco-dialogue-repair-v4.gguf
```

Notes:

- There is no separate PII model. PII risk is one classifier head in the
  shared support-understanding stack.
- Large GGUF files are not committed to git. Use your internal artifact
  delivery process or the provided model bootstrap flow.
- The base model must match the adapters. Do not swap in an instruct/DPO base
  unless the adapters were trained for that base.

## Step-by-Step Integration Plan

### Step 1: Run the Cookbook App Unmodified

Before porting anything, build and run the reference app:

```bash
cd examples/telco-triage-ios
./bootstrap-models.sh
xcodegen generate
open TelcoTriage.xcodeproj
```

Validate a few baseline prompts:

```text
How do I restart my router?
Can you help me?
Can you do it for me?
What is my network SSID?
Why is my Wi-Fi slow?
Run diagnostics
I want to talk to a person
```

This confirms the model/resource bundle is installed correctly before you
debug integration code.

### Step 2: Add the Runtime Dependency

Bring the `LFMEngine` source or package into your app target and add
`llama.swift`.

The cookbook vendors `Packages/LFMEngine/Sources/LFMEngine` directly into the
app target. You can either follow that pattern or package it as an internal
Swift package.

### Step 3: Port the Control Plane

Port these source groups:

```text
TelcoTriage/Core/Model
TelcoTriage/Core/Orchestration
TelcoTriage/Core/RAG
TelcoTriage/Core/Retrieval/BM25HierarchyRetriever.swift
TelcoTriage/Core/Retrieval/RetrievalContext.swift
TelcoTriage/Core/Retrieval/RAGStackStatus.swift
TelcoTriage/Core/Composer
TelcoTriage/Core/Intelligence
TelcoTriage/Core/Intelligence/Understanding
TelcoTriage/Core/Tools
TelcoTriage/Core/Routing
TelcoTriage/Core/Observability
```

If your host app already has its own tool abstractions, keep the Telco tool
protocol/registry and implement small adapters from your internal action
system. You do not need the cookbook `Features/Chat`, `Features/Branding`,
`Features/DemoCustomer`, or `Features/DemoTools` folders unless you want to
reuse the sample UI and simulated tools.

### Step 4: Wire Startup

Mirror the dependency construction in `AppState.buildLFMStack(...)`.

Required startup operations:

1. Resolve model paths from the app bundle.
2. Create one session-scoped `LlamaBackend`.
3. Load the base model off the main thread.
4. Build `LlamaAdapterBackend`.
5. Load `RAGUnitCorpus`.
6. Build `BM25HierarchyRetriever`.
7. Create `DeterministicAnswerComposer`.
8. Register only executable host-app tools.
9. Create `ToolAliasMap`.
10. Create `TelcoChatDispatcher`.
11. Create `TelcoSharedUnderstandingClassifier`.
12. Create `TelcoTurnRelationV4Strategy`.

Pseudocode:

```swift
let backend = LlamaBackend()
try await backend.loadModel(path: basePath, contextLength: 8192, gpuLayers: gpuLayers)

let adapterBackend = LlamaAdapterBackend(backend: backend)
let corpus = try RAGUnitCorpus.loadFromBundle()
let retriever = BM25HierarchyRetriever(corpus: corpus)
let composer = DeterministicAnswerComposer()
let toolRegistry = ToolRegistry(tools: [
    YourRestartRouterTool(),
    YourSpeedTestTool(),
])
let aliasMap = ToolAliasMap.default()

let dispatcher = TelcoChatDispatcher(
    stageA: nil,
    stageB: nil,
    kbFallback: KeywordKBExtractor(),
    kb: [],
    composer: composer,
    corpus: corpus,
    lexicalRetriever: retriever,
    toolRegistry: toolRegistry,
    toolAliasMap: aliasMap,
    dialogueRepairVerbalizer: DialogueRepairVerbalizer.bundled(backend: adapterBackend)
)

let sharedUnderstanding = try TelcoSharedUnderstandingClassifier.bundled(backend: backend)
let turnRelation = try TelcoTurnRelationV4Strategy.bundled(backend: backend)

let session = TelcoSupportSession(
    dispatcher: dispatcher,
    understandingClassifier: sharedUnderstanding,
    relationalStrategy: turnRelation
)
```

### Step 5: Wire One Chat Turn

Your UI should call the headless session and render the returned value. On each
user message:

```swift
let turn = try await session.handle(userText)

// Render these however your app wants:
turn.text
turn.citation
turn.deepLink
turn.requiresConfirmation
turn.executableToolIntent
turn.blackboard
```

`TelcoSupportSession` owns the dialogue blackboard and calls the same relation
classifier, shared-understanding classifier, RAG retriever, policy engine, and
composer path used by the sample app. Your UI owns presentation only:

1. Render the resulting answer in your UI.
2. Show source/deep-link chips when present.
3. Show confirmation UI only when `executableToolIntent` is present.
4. Execute confirmed tools through your app's action layer.
5. Call `recordPendingToolExecuted()` or `recordPendingToolCancelled(...)` so
   the session blackboard stays in sync.

Do not let a generated answer execute an action. Only typed tool intent plus
explicit user confirmation should execute a tool.

### Step 6: Replace Demo Tools With Host-App Tools

The cookbook tools simulate carrier support actions. In your app, register only
actions that are real and safe:

```swift
ToolRegistry(tools: [
    RestartRouterTool(...),
    RunDiagnosticsTool(...),
    RunSpeedTestTool(...),
])
```

If a corpus `link_id` does not map to a real tool, leave it out of
`ToolAliasMap`. The app will keep that response as guidance or navigation
instead of creating a fake confirmation card.

### Step 7: Replace the Corpus

For production, generate your own `rag-units-v1.json` from your support flows.
Each unit should represent one support topic, page, or safe action.

Preserve these fields:

```text
page_id
title
section
aliases
steps
body
link_id
canonical_url
action_affordance
availability
answer_summary
answer_facts
```

The fields have separate jobs:

- `title`, `aliases`, `steps`, hierarchy, and `body` support retrieval.
- `answer_summary` and `answer_facts` are source-approved response evidence.
- `canonical_url` and `link_id` control navigation and tool resolution.
- `availability` records applicability constraints the policy layer may check.

Do not copy a long support page into the response and expect the composer to
discover the answer. For high-volume customer tasks, provide a direct summary
and no more than three facts that are explicitly supported by the source. Keep
retrieval aliases coherent: one unit should not collect unrelated phrases just
to win a benchmark.

Good aliases matter. Include real user phrasing such as:

```text
"wifi slow"
"network name"
"ssid"
"restart router"
"pause internet"
"connected devices"
```

After any corpus change, test both retrieval and the rendered answer. At a
minimum, assert the expected page ID, direct answer text, citation/deep link,
and absence of unrelated page-navigation prose.

## What Not To Do

- Do not use "top BM25 chunk wins" as the final production behavior.
- Do not ask a free-form model to invent the final support answer.
- Do not show a tool confirmation unless a real registered tool exists.
- Do not copy voice, vision, or demo screens into the first POC unless they are
  part of the product requirement.
- Do not treat `knowledge-base.json` as the current RAG source of truth.
  `rag-units-v1.json` is the current structured corpus.
- Do not rely on a standalone PII model. PII risk is a classifier head plus
  deterministic policy.

## Minimum Parity Checklist

A host-app integration is close to Telco Triage parity when all of these pass:

```text
Single-turn RAG:
- "How do I restart my router?" returns restart-router guidance and source.

Multi-turn continuation:
- "How do I restart my router?"
- "Can you help me?"
  keeps the restart-router context.

Tool confirmation:
- "How do I restart my router?"
- "Can you do it for me?"
  offers restart confirmation only if restart is registered.

Topic switch:
- "How do I restart my router?"
- "What is my network SSID?"
  starts a new SSID/network-name task.

Ambiguous RAG:
- "My internet is slow"
  answers with appropriate troubleshooting/speed guidance or asks a
  clarification if candidates are too close.

Deflection:
- "Can you pay my bill?"
  deflects to account/cloud/app navigation instead of local RAG.

Human handoff:
- "I want to talk to a person"
  escalates to live-agent handoff.

Out of scope:
- "What's the weather?"
  declines or redirects instead of forcing a support answer.
```

## Suggested First POC Scope

If you are starting from scratch, build in this order:

1. Text-only query box.
2. Load model bundle.
3. Load `rag-units-v1.json`.
4. Return deterministic RAG/composer answers.
5. Add shared understanding labels.
6. Add turn relation + blackboard.
7. Add tool registry + confirmation UI.
8. Add host-app deep links.
9. Add cloud/account/human deflection.
10. Add optional voice, vision, or metrics surfaces later.

This sequence avoids mixing runtime, RAG quality, state handling, and UI
complexity into one debugging problem.

## Where To Look In The Reference App

| Question | Start here |
| --- | --- |
| How are dependencies built? | `TelcoTriage/App/TelcoTriageApp.swift` |
| What should my app call per turn? | `TelcoTriage/Core/Orchestration/TelcoSupportSession.swift` |
| How does a turn get dispatched? | `TelcoTriage/Core/Intelligence/TelcoChatDispatcher.swift` |
| What decides answer/tool/deflection? | `TelcoTriage/Core/Intelligence/TelcoPolicyEngine.swift` |
| What is the conversation memory? | `TelcoTriage/Core/Intelligence/Understanding/TelcoDialogueBlackboard.swift` |
| How is RAG ranked? | `TelcoTriage/Core/Retrieval/BM25HierarchyRetriever.swift` |
| How is the answer rendered? | `TelcoTriage/Core/Composer/AnswerComposer.swift` |
| How are tools registered? | `TelcoTriage/Core/Tools/ToolRegistry.swift` |
| How do corpus IDs map to tools? | `TelcoTriage/Core/Routing/ToolAliasMap.swift` |
| Which model files are expected? | `TelcoTriage/Core/Model/LlamaAdapterBackend.swift` |

## Summary

The integration should preserve the control-plane boundary:

```text
LFM classifiers produce structured labels.
Swift policy decides the route.
RAG supplies evidence.
The composer renders grounded text.
The host app executes only registered, confirmed tools.
```

That boundary is what makes the assistant useful for production-style support:
it is local and model-assisted, but still auditable, grounded, and controlled by
the application.
