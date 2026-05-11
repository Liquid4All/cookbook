# Voice Assistant

End-to-end tutorial showing how to fine-tune `LiquidAI/LFM2.5-Audio-1.5B` to map spoken Home Assistant voice commands directly to function calls, and to deploy the resulting model on-device via llama.cpp.

## Language

**OHF-Voice dataset**:
The HF dataset `LiquidAI/OHF-Voice-audio-20260504`, 55,302 (audio, function call) pairs across 41 Home Assistant **Function signatures**. Ships a single `train` split only.
_Avoid_: "OHF dataset" alone (ambiguous), "voice command dataset"

**Function call**:
The pipe-delimited string the model emits and the dataset's ground truth uses: `FuncName|$arg1=val1|$arg2=val2`. The unit of correct prediction.
_Avoid_: "tool call" (the sibling `home-assistant` cookbook example uses that name for a different, JSON-with-tokens format), "command", "intent"

**Function signature**:
One of 41 closed-set Home Assistant operations the model can emit, e.g. `HassStartTimer`, `HassLightSet`.
_Avoid_: "skill", "intent class", "action"

**Argument**:
A `$name=value` pair inside a **Function call**. Values are free-text and may contain whitespace, e.g. `$area=living room`, `$duration=5 minutes`.
_Avoid_: "parameter", "slot"

**Baseline mode**:
Inference with the unmodified LFM2.5-Audio-1.5B in ASR mode (system_prompt = `"Perform ASR."`, no in-context function specs because the runtime rejects arbitrary system prompts). The model transcribes the audio verbatim, which essentially never matches the **Function call** format. Establishes the floor against which fine-tuning is justified.
_Avoid_: "zero-shot eval", "prompted mode"

**Fine-tuned mode**:
Inference with the fine-tuned model, audio-only input, no function specs in the prompt. The closed set of **Function signatures** is baked into the weights.
_Avoid_: "trained mode", "production mode"

**Mmproj**:
The multimodal projector. A separate GGUF file containing the audio encoder + adapter that bridges raw audio into the LFM2 backbone's embedding space. llama.cpp inference of LFM2.5-Audio requires two GGUFs: a language-model GGUF (produced by `convert_hf_to_gguf.py`) and an Mmproj GGUF (produced by `convert_hf_to_gguf.py --mmproj`).
_Avoid_: "audio encoder" alone (the Mmproj is encoder + adapter, not just the encoder), "vision projector"

**Test set**:
The held-out 5% partition of the **OHF-Voice dataset**, disjoint from training data. Lives as the `test` split of `Paulescu/OHF-Voice-audio-20260504`, a 95/5 fork of the upstream dataset (which ships only a `train` split). Produced once by `scripts/prepare_raw_data.py`.
_Avoid_: "validation set" (a different role, used inside the training loop), bare "eval set" (ambiguous with **Eval subset**)

**Eval subset**:
The stratified per-**Function-signature** draw from the **Test set** used for an individual eval run. Up to 10 samples per function across 41 functions (~410 samples; capped lower for rare functions like `HassRespond` whose test partition has fewer than 10 examples).
_Avoid_: "benchmark", "eval split"

**Format compliance**:
Layered metric. Fraction of predictions that parse as a valid **Function call** (a function name + zero or more `$key=value` pairs separated by `|`).

**Function-name accuracy**:
Layered metric. Among parseable predictions, the fraction whose **Function signature** matches the ground truth.

**Argument accuracy**:
Layered metric. Among predictions with a correct **Function signature**, the fraction whose **Argument** keys and values all match ground truth under strict equality (whitespace-stripped, no case folding, no unit normalisation).

## Relationships

- An **OHF-Voice dataset** sample is an (audio, **Function call**) pair
- A **Function call** has exactly one **Function signature** and zero or more **Arguments**
- The **Eval subset** is drawn from the **Test set**, stratified by **Function signature**
- **Format compliance**, **Function-name accuracy**, and **Argument accuracy** are all computed on the **Eval subset** and reported per mode (**Baseline mode** or **Fine-tuned mode**)

## Example dialogue

> **Reader:** "If **Baseline mode** is expected to score near zero, why include it at all?"
> **Author:** "It's the floor. The tutorial's headline result is the gap between **Baseline mode** and **Fine-tuned mode**. Without the floor, a reader can't tell whether the improvement came from fine-tuning or from prompting."

> **Reader:** "Why is the **Eval subset** only ~410 samples when the dataset has 55,302?"
> **Author:** "Per-**Function-signature** stratification: up to 10 samples each across 41 functions. That keeps statistics stable for rare functions and keeps the eval cheap enough for a tutorial walkthrough. `HassRespond` has only 94 total samples (4-5 in test); for it we use what we have rather than oversample."

## Flagged ambiguities

- **"tool call" vs "function call"**: the sibling `examples/home-assistant/` uses *tool call* for JSON-with-tokens output (`<|tool_call_start|>[{"name": ...}]<|tool_call_end|>`). Voice-assistant uses **Function call** for the pipe-delimited string format. Different shape, different name. Don't conflate.
- **`LFM2-Audio` vs `LFM2.5-Audio`**: the canonical model id is `LiquidAI/LFM2.5-Audio-1.5B`. `LFM2-Audio-1.5B` is the prior version. When in doubt, use the full HF repo id.
