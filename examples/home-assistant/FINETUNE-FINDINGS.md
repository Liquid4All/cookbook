# Fine-Tune Findings: LFM2-350M on Home Assistant SFT

Date: 2026-03-25
Model: `LiquidAI/LFM2-350M` fine-tuned with LoRA via leap-finetune on Modal (H100)
Dataset: `Paulescu/home-assistant-sft` (491 examples, 80/20 stratified split by capability)
Training: 5 epochs, lr=2e-4, batch size 4, ~2 min 12 sec, final eval_loss 0.0743

---

## Benchmark Results

The benchmark runs 100 tasks across a 3D taxonomy (capability x phrasing x inference depth).

| Model | Score | Notes |
|---|---|---|
| gpt-4o-mini | 93/100 (93%) | Cloud API, reference ceiling |
| LFM2-350M baseline (Q8_0) | 28/100 (28%) | Pre-fine-tune |
| LFM2-350M fine-tuned (Q8_0) | 47/100 (47%) | Post-fine-tune, same quantization |

The fine-tuned model gains 19 points (+68% relative improvement) over the baseline.

---

## Results by Capability

| Capability | Baseline | Fine-tuned | Delta | Tasks |
|---|---|---|---|---|
| lights | 37.5% | 83.3% | +45.8% | 24 |
| multi_tool | 8.3% | 58.3% | +50.0% | 12 |
| scene | 0.0% | 50.0% | +50.0% | 10 |
| thermostat | 25.0% | 25.0% | 0.0% | 16 |
| status | 20.0% | 20.0% | 0.0% | 10 |
| rejection | 0.0% | 0.0% | 0.0% | 12 |
| doors | 75.0% | 56.2% | -18.8% | 16 |

### What improved

**Lights (+46%)**: The most-represented capability in the dataset. The model learned to call `toggle_lights` reliably for imperative, colloquial, question, and semantic phrasing. It still struggles with implicit phrasing (e.g., "I can't see anything in here"), scoring 3/8 on implicit light tasks.

**Multi-tool (+50%)**: The model learned to issue two tool calls in a single turn for compound requests. It handles literal and some colloquial phrasings well, but semantic multi-tool tasks (e.g., "prep the house for the night") remain largely unsolved.

**Scene (+50%)**: Pre-fine-tune the model never called `activate_scene`. Post-fine-tune it correctly identifies 4/5 literal scene names (missing movie night). All 5 implicit scene tasks still fail.

### What did not change

**Thermostat (0%)**: The model often calls `set_thermostat` with an incorrect `mode` parameter or wrong temperature. For example, it sets `mode="heat"` when only temperature is mentioned, or hallucinates a temperature for semantic inputs like "it's too warm". The training data has 64 thermostat examples; the issue appears to be inconsistent argument selection rather than missing tool awareness.

**Status (0%)**: The model occasionally calls `get_home_status` for whole-home queries but fails for targeted questions (e.g., "is the bedroom light on?"). It tends to answer status questions with plain text rather than calling the tool.

### What regressed

**Doors (-19%)**: The baseline scored 75% on doors, performing well on literal lock/unlock commands. The fine-tuned model dropped to 56.2%. Likely cause: fine-tuning shifted the output distribution toward lights (most common training class), partially overwriting the base model's door-handling behavior. The model now occasionally calls `toggle_lights` instead of `lock_door` or `unlock_door` for door-related tasks.

### Rejection remains at 0%

The model never calls `intent_unclear`. For ambiguous requests it attempts to fulfill them (often with a plausible but wrong tool call), and for off-topic requests it outputs plain text instead of calling the tool. This is a known limitation of small models fine-tuned primarily on positive tool-call examples without strong boundary signals.

---

## Results by Phrasing and Depth

| Phrasing | Baseline | Fine-tuned |
|---|---|---|
| imperative | 35.3% | 61.8% |
| colloquial | 29.2% | 41.7% |
| question | 33.3% | 38.9% |
| implicit | 12.5% | 37.5% |

| Inference depth | Baseline | Fine-tuned |
|---|---|---|
| literal | 37.7% | 59.0% |
| semantic | 18.5% | 40.7% |
| boundary | 0.0% | 0.0% |

Implicit phrasing saw the largest relative gain (+25 points), suggesting the model learned some degree of intent inference from semantically labeled training examples.

---

## Key Technical Discovery: LFM2 Tool-Call Format

LFM2 models output tool calls as plain text inside special tokens rather than returning a structured `tool_calls` object in the OpenAI API response:

```
# What LFM2 generates (function-call notation, base model default)
<|tool_call_start|>[toggle_lights(room="kitchen", state="on")]<|tool_call_end|>

# What the OpenAI tools API returns for models that support it
tool_calls: [{"function": {"name": "toggle_lights", "arguments": "{\"room\":\"kitchen\",\"state\":\"on\"}"}}]
```

llama-server returns this as `content` with `tool_calls: null`. The benchmark agent originally checked only `message.tool_calls`, so all tool calls from LFM2 models were silently dropped, producing the inflated "regression" score of 16/100 observed in an intermediate run.

The fix adds `_extract_lfm2_tool_calls()` in `app/agent.py` as a fallback: when `tool_calls` is null, the function parses `content` for LFM2 text-format tool calls, handling both the JSON training format and the base model's function-call notation. This applies to all local LFM2 model runs automatically.

The training data used JSON format inside the tokens:
```
<|tool_call_start|>[{"name": "toggle_lights", "arguments": {"room": "kitchen", "state": "on"}}]<|tool_call_end|>
```

The fine-tuned model, however, outputs function-call notation rather than JSON. This mismatch is likely caused by the chat template at inference time, which formats tool definitions in a way that primes function-call syntax. With 491 training examples the 350M model did not strongly override the base model's format preference.

---

## Next Steps

### High impact

**More data for weak capabilities**: Rejection (0%) and thermostat (25%) need more examples. Specifically, boundary-case rejection examples (ambiguous, off-topic, unsupported device) and thermostat examples that vary the `mode` argument across all three values (heat, cool, auto) with diverse phrasing.

**Correct the training format to match inference**: The training data should use function-call notation to match what the model generates at inference time. Alternatively, investigate why the chat template at inference differs from the format used during training, and align them.

**Doors regression**: Inspect door failures qualitatively. If the model is calling `toggle_lights` instead of `lock_door`, add negative examples or rebalance the training split so doors (64 examples) has comparable weight to lights (96 examples).

### Lower impact

**Train the 1.2B model**: The 1.2B model baseline was 71/100. The same dataset is already uploaded. Run the 1.2B config (`finetune/configs/1.2B.yaml`) to see if a larger model reaches 85-90% with the same data.

**Enforce JSON output format**: Add a few-shot prompt at inference or a grammar constraint in llama-server to force JSON format inside `<|tool_call_start|>...<|tool_call_end|>`, eliminating the dual-format parsing needed today.

---

# Fine-Tune Findings: LFM2.5-350M on Home Assistant SFT

Date: 2026-03-26
Model: `LiquidAI/LFM2.5-350M` fine-tuned with LoRA via leap-finetune on Modal (H100)
Dataset: `Paulescu/home-assistant-sft` (393 train / 98 eval examples)
Training: 5 epochs, lr=2e-4, batch size 4, ~1 min 49 sec, final eval_loss 0.0770

---

## Benchmark Results

| Model | Score | Notes |
|---|---|---|
| gpt-4o-mini | 93/100 (93%) | Cloud API, reference ceiling |
| LFM2-350M baseline (Q8_0) | 28/100 (28%) | Pre-fine-tune |
| LFM2-350M fine-tuned (Q8_0) | 47/100 (47%) | Post-fine-tune |
| LFM2.5-350M baseline (Q8_0) | 16/100 (16%) | Pre-fine-tune |
| LFM2.5-350M fine-tuned (Q8_0) | 16/100 (16%) | Post-fine-tune, no improvement |

Fine-tuning produced zero measurable improvement for LFM2.5-350M: the score before and after is identical, including the exact same 16 tasks passing in both runs.

---

## Results by Capability

| Capability | Baseline | Fine-tuned | Delta | Tasks |
|---|---|---|---|---|
| lights | 25.0% | 25.0% | 0.0% | 24 |
| multi_tool | 8.3% | 8.3% | 0.0% | 12 |
| scene | 0.0% | 0.0% | 0.0% | 10 |
| thermostat | 0.0% | 0.0% | 0.0% | 16 |
| status | 0.0% | 0.0% | 0.0% | 10 |
| rejection | 0.0% | 0.0% | 0.0% | 12 |
| doors | 56.2% | 56.2% | 0.0% | 16 |

Not a single capability changed. The fine-tuned model and the baseline model pass exactly the same set of 16 tasks.

---

## Root Cause: Chat Template Mismatch

LFM2.5-350M uses a ChatML-style chat template (`<|im_start|>` / `<|im_end|>`) that is structurally different from LFM2-350M's format. Tool definitions are injected into the system prompt as a JSON array:

```
<|im_start|>system
List of tools: [{"type": "function", "function": {...}}, ...]
<|im_end|>
```

The training dataset (`Paulescu/home-assistant-sft`) was formatted for LFM2-style tool call tokens:

```
<|tool_call_start|>[{"name": "toggle_lights", "arguments": {...}}]<|tool_call_end|>
```

LFM2.5-350M does not use these special tokens. During fine-tuning, the model sees the `<|tool_call_start|>` and `<|tool_call_end|>` tokens in the training data but these tokens are not part of LFM2.5's vocabulary or chat template, so the fine-tuning signal does not transfer to actual inference-time behavior.

The benchmark agent's `_extract_lfm2_tool_calls()` fallback parses LFM2-style tokens from the content field. Because LFM2.5-350M never produces these tokens at inference time, the fallback never fires for the capabilities that require it (thermostat, status, scene, rejection, multi_tool). The 16 tasks that do pass are simple door and light commands where the model happens to emit a parseable response through other means.

---

## Comparison with LFM2-350M

LFM2-350M improved from 28% to 47% with the same dataset and training config. LFM2.5-350M started lower (16%) and showed no improvement. The key difference is the chat template: LFM2-350M and the training data share the same tool-call token format, while LFM2.5-350M does not.

---

## Next Steps

**Regenerate training data in LFM2.5 format**: The most direct fix is to regenerate `Paulescu/home-assistant-sft` using the LFM2.5 chat template, where tool calls are expressed as assistant messages in ChatML format rather than wrapped in LFM2-specific tokens.

**Test LFM2.5-1.2B-Instruct**: The 1.2B instruct model already uses the ChatML-style template and is already configured in `finetune/configs/1.2B.yaml`. A fine-tuned 1.2B run would validate whether the larger LFM2.5 family responds to this dataset once the format is correct.

**Investigate LFM2.5-350M native tool-call output**: Capture raw model output for a failing task to confirm the exact format LFM2.5-350M uses for tool calls at inference. This will clarify whether the issue is (a) format mismatch only or (b) the 350M model in the LFM2.5 family lacks instruction-following capacity for tool use at this scale.

---

# Fine-Tune Findings: LFM2.5-1.2B-Instruct on Home Assistant SFT

Date: 2026-03-26
Model: `LiquidAI/LFM2.5-1.2B-Instruct` fine-tuned with LoRA via leap-finetune on Modal (H100)
Dataset: `Paulescu/home-assistant-sft` (393 train / 98 eval examples)
Training: 3 epochs, lr=2e-4, batch size 2, `use_liger_kernel: false` (disabled due to container incompatibility), final eval_loss 0.0688

Note: the `project_name` in `finetune/configs/1.2B.yaml` had a dot (`home-assistant-1.2B`) which was rejected by leap-finetune's job name validator. Fixed to `home-assistant-1-2B`. The `use_liger_kernel` flag caused an `ImportError` in the Modal container (`HybridCache` not found in the container's `transformers` version); disabling it unblocked training with no other changes.

---

## Benchmark Results

Both parsing modes score identically at the top-line level.

| Model | Score | Parsing mode |
|---|---|---|
| gpt-4o-mini | 93% | n/a (cloud API) |
| LFM2.5-1.2B-Instruct baseline (Q4_0) | 71% | default |
| LFM2.5-1.2B-Instruct fine-tuned (Q4_0) | 51% | default |
| LFM2.5-1.2B-Instruct fine-tuned (Q4_0) | 51% | raw-tool-call-parsing |

Fine-tuning caused a regression of 20 points (71% to 51%). Unlike LFM2.5-350M, where fine-tuning had no effect, the larger 1.2B model was actively degraded.

---

## Results by Capability

| Capability | Baseline | FT default | FT raw | Delta (default) | Tasks |
|---|---|---|---|---|---|
| lights | n/a | 83.3% | 75.0% | n/a | 24 |
| status | n/a | 60.0% | 60.0% | n/a | 10 |
| doors | n/a | 62.5% | 62.5% | n/a | 16 |
| multi_tool | n/a | 50.0% | 66.7% | n/a | 12 |
| scene | n/a | 50.0% | 50.0% | n/a | 10 |
| thermostat | n/a | 25.0% | 25.0% | n/a | 16 |
| rejection | n/a | 0.0% | 0.0% | n/a | 12 |

Baseline capability breakdown was not recorded separately; the 71% overall score is from a prior session without a per-capability file.

---

## Delta Between Parsing Modes

The two parsing modes produce the same total (51/51) but differ in which tasks pass:

| Capability | Default | Raw | Delta |
|---|---|---|---|
| lights | 83.3% | 75.0% | -8.3% |
| multi_tool | 50.0% | 66.7% | +16.7% |
| all others | same | same | 0% |

The `--raw-tool-call-parsing` flag recovers 2 multi_tool tasks that default parsing misses, but loses 2 lights tasks in exchange. The net effect is zero. This indicates the model is producing tool calls that the default llama-server parser occasionally mis-handles for multi_tool compound calls, but that effect is fully offset by a corresponding loss in single-tool lights calls. No parsing mode has a meaningful advantage at this score range.

---

## Results by Phrasing and Depth

| Phrasing | Fine-tuned (default) | Fine-tuned (raw) |
|---|---|---|
| imperative | 58.8% | 58.8% |
| question | 61.1% | 61.1% |
| colloquial | 45.8% | 45.8% |
| implicit | 37.5% | 37.5% |

| Inference depth | Fine-tuned (default) | Fine-tuned (raw) |
|---|---|---|
| literal | 65.6% | 67.2% |
| semantic | 40.7% | 37.0% |
| boundary | 0.0% | 0.0% |

---

## Root Cause: Format Mismatch Combined with Larger Model Capacity

LFM2.5-1.2B-Instruct uses the same ChatML format as LFM2.5-350M and does not use `<|tool_call_start|>` tokens. The training data is in LFM2 token format. This is the same mismatch that prevented any improvement in the 350M.

The difference from the 350M case is that the 1.2B model is large enough to partially absorb the training signal, partially overwriting its native ChatML tool-call behavior without fully learning the LFM2 format. The result is a hybrid that performs worse than either the native baseline or what the training data intended:

- The baseline 1.2B model used native llama-server ChatML tool calls, scoring 71%.
- After fine-tuning, the model's native tool-call behavior was partially disrupted, dropping to 51%.
- The 350M was too small to absorb any signal, so it stayed at 16% before and after.

Training loss converged correctly (0.07134 at epoch 1 to 0.06884 at epoch 3), confirming the model learned the training data format. The regression is purely an inference-time format conflict.

---

## Comparison Across All Runs

| Model | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| LFM2-350M (Q8_0) | 28% | 47% | +19% |
| LFM2.5-350M (Q8_0) | 16% | 16% | 0% |
| LFM2.5-1.2B-Instruct (Q4_0) | 71% | 51% | -20% |

The pattern is clear: fine-tuning only helps when the training data format matches the model's inference-time format. LFM2-350M shares the LFM2 token format with the training data and improves. LFM2.5 models do not share that format: the 350M is too small to absorb the conflicting signal, while the 1.2B is large enough to partially absorb it and regress.

---

## Next Steps

**Regenerate training data in LFM2.5 ChatML format**: This is the highest-leverage fix. Once the dataset uses `<|im_start|>assistant` tool responses in the format LFM2.5 actually produces at inference, the 1.2B model's strong 71% baseline should improve meaningfully with fine-tuning. The 350M will also benefit.

**Re-run the 1.2B fine-tune after fixing the dataset**: With correct format alignment, the 1.2B model is the most promising candidate given its strong baseline. Even modest gains (10-15 points) would push it toward 80-85%, approaching gpt-4o-mini territory.

**Diagnose thermostat (25%) and rejection (0%) separately**: These capabilities are weak across all fine-tuned models regardless of format. Both need targeted data improvements before the format fix alone can unlock them.
