# Models And Runtime Boundaries

Telco Triage is an on-device iOS reference app for home-internet support. The
normal customer path uses Liquid models for closed-set understanding and uses
deterministic Swift code for retrieval, policy, citations, action safety, and
answer composition.

The key distinction:

- Models classify the turn and selected follow-up relation.
- Swift owns state, retrieval, route policy, citations, confirmation, and tools.
- The grounded answer text is composed from canonical support evidence.
- First-turn procedural answer generation calls: 0.

## Core Runtime Loop

```text
User turn
  -> ChatViewModel owns UI state and pending actions
  -> telco-turn-relation-v4 decides how this turn relates to prior context
  -> telco-shared-clf-v1 reads support intent/tool/cloud/safety signals
  -> TelcoDialogueBlackboard records active task, prior page, pending tool, repair count
  -> BM25HierarchyRetriever retrieves from canonical RAG units
  -> TelcoPolicyEngine chooses exactly one route
  -> DeterministicAnswerComposer renders from selected evidence
  -> UI shows answer, citation, in-app link, handoff, or confirmation
```

## Active Customer Path

| Component | Runtime role |
| --- | --- |
| `lfm25-350m-base-Q4_K_M.gguf` | Base LFM2.5-350M loaded by the local runtime. |
| `telco-shared-clf-v1.gguf` | Support-understanding adapter. One mean-pooled forward feeds the shared classifier heads. |
| Nine shared classifier triplets | Required heads for support intent, issue complexity, routing lane, cloud requirements, required tool, escalation risk, PII risk, transcript quality, and slot completeness. |
| `telco-topic-scope_classifier_*` | Optional off-domain scope head. If absent, the app ignores it; if present, policy uses it only with weak grounding. |
| `telco-turn-relation-v4.gguf` | Stateful relation adapter for follow-up, repair, confirmation, clarification answer, topic switch, and escalation turns. |
| `telco-turn-relation_classifier_*` | Required relation-head triplet for the relation adapter. |
| `rag-units-v1.json` | Canonical support corpus for BM25 retrieval and deterministic composition. |
| `page-link-table-v1.json` | Canonical app destinations for source chips and open buttons. |
| `telco-tool-selector-v3.gguf` | Optional helper for ambiguous local action/argument paths. Pure grounded Q&A does not invoke it. |
| `telco-dialogue-repair-v4.gguf` | Bounded follow-up/repair verbalizer. It can adjust wording only; route, evidence, handoff, and confirmation stay Swift-owned. |

Each `*_classifier_*` head is a `{weights.bin, bias.bin, meta.json}` triplet.

## Shared Understanding Heads

| Head | Product question answered |
| --- | --- |
| support intent | What broad support situation is this turn about? |
| issue complexity | Can this be answered locally, or does it need backend/human help? |
| routing lane | Should policy prefer local answer, local tool, cloud assist, human escalation, or block? |
| cloud requirements | Which live systems would be needed if local data is insufficient? |
| required tool | Which supported local tool, if any, is needed? |
| escalation risk | Is the user low-risk, frustrated, complaining, churn-risk, or urgent? |
| PII risk | Does the turn include sensitive account/contact/payment identity data? |
| transcript quality | Is the turn clean enough, or should the app ask for clarification? |
| slot completeness | Which argument categories are missing? Values are not extracted by this head. |
| topic scope | Optional high-precision off-domain signal. It is a veto input, not a hard route by itself. |

The shared heads do not execute actions and do not write answers. They are
evidence for `TelcoPolicyEngine`; hard product decisions still require
deterministic priors, state, retrieval evidence, tool registration, and policy
thresholds.

## Stateful Relation Head

`telco-turn-relation-v4` runs only when there is prior dialogue state or a
pending confirmation to interpret. It labels how the current turn relates to
the prior turn:

```text
independent_new_task
continuation_same_task
continuation_same_section
step_focus
clarification_answer
confirmation_yes / confirmation_no
repair_cannot_find / repair_failed
topic_switch
escalation_request
ambiguous_short_turn
```

The relation label feeds `TelcoStateOperationResolver`, which decides whether to
reuse active evidence, retrieve with prior bias, retrieve fresh, ask for
clarification, clear context, or hand off. The relation head informs state; it
does not directly route the answer.

## Policy And Grounding Contract

`TelcoPolicyEngine` is the single route owner. It consumes:

- relation label
- shared-understanding labels
- dialogue blackboard snapshot
- deterministic priors such as explicit human request or account/billing terms
- BM25 retrieval candidates
- selected RAG unit
- `ToolRegistry` and `ToolAliasMap`

It emits exactly one route:

```text
ragAnswer
answerPlusAction
toolAction
accountNav
liveAgent
outOfScope
clarify
greeting
noRagAnswer
```

Soft head deflections such as escalation/cloud/clarify are corroboration-gated:
when a real local support page grounds the turn, the grounded answer wins. The
optional topic-scope head and the peripheral-hardware lexicon can decline only
when the top RAG unit is weakly grounded.

## Inactive Or Legacy Artifacts

The normal customer path does not invoke:

| Artifact | Current status |
| --- | --- |
| `chat-mode-router-v2.gguf` | Legacy router; bypassed by composer path. |
| `vz-topic-gate-clf-v1.lora.gguf` | Legacy topic gate; bypassed by composer path. |
| `vz-refusal-flags-clf-v1.lora.gguf` | Legacy refusal adapter; bypassed by composer path. |
| `telco-relational-v1.gguf` | Legacy pairwise relational pass; superseded by `telco-turn-relation-v4`. |
| Stage B answer generator | Not part of normal grounded answers. |
| ColBERT | Not adopted for this closed corpus; BM25 hierarchy plus aliases is the current retrieval path. |

## Runtime Cost Contract

| Component | Cost profile |
| --- | --- |
| Shared understanding | One LFM2.5-350M classifier forward plus small head projections. |
| Turn relation | One relation forward on stateful turns only; deterministic fallback otherwise. |
| Retrieval | Pure Swift BM25 over canonical RAG units. |
| Route policy | Pure Swift, confidence-gated. |
| Composer | Pure Swift, deterministic. |
| Dialogue repair verbalizer | Optional model call for selected repair/follow-up wording only. |
| Tool execution | Swift fast path after confirmation; tool selector only for ambiguous action/argument selection. |
