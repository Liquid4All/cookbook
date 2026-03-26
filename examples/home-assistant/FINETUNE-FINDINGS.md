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
