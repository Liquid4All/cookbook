# Home Assistant powered by a local LFM

This project builds a home assistant system powered entirely by a local LFM model. The focus
is practical: every step of the journey is covered, from a first working prototype to a
fine-tuned model for tool calling running fully on your own hardware.

In this tutorial you will learn how to:

1. Build a [proof of concept](#step-1-build-a-proof-of-concept) for a fully local Home Assistant.
2. [Benchmark](#benchmark) its tool-calling accuracy so you have a clear baseline to improve on.
3. Generate [synthetic data](#step-3-generate-synthetic-data) for model fine-tuning.
4. [Fine-tune](#step-4-fine-tune-the-model) the model on this synthetic data to maximise accuracy.

## Quick start

**Requirements**

- [uv](https://docs.astral.sh/uv/getting-started/installation/) for running the Python app
- [llama.cpp](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#installation) for running the model locally (`llama-server` must be on your PATH)

**1. Start the app server**

```bash
uv run uvicorn app.server:app --port 5173 --reload
```

**2. Open the app**

```bash
open http://localhost:5173
```

![Demo](assets/Demo.gif)

The UI includes a model selector. When you pick a model, the app automatically downloads
and starts `llama-server` in the background. No manual model server setup is needed.

## Step 1: Build a proof of concept

The main components of our solution are: 

- **Browser** renders the UI and sends chat messages to the server
- **FastAPI server** handles HTTP requests, manages home state, and starts the llama.cpp server on model selection
- **Agent loop** drives the conversation, calls the model for inference, and dispatches tool calls
- **Tools** read and mutate the home state (lights, thermostat, doors, scenes)
- **llama.cpp server** runs the LFM model locally and exposes an OpenAI-compatible API

```mermaid
graph LR
    Browser <-->|chat / state| FastAPI[FastAPI server]
    FastAPI -->|start process| LFM[llama.cpp server]
    FastAPI -->|run| Agent[Agent loop]
    Agent <-->|inference| LFM
    Agent <-->|execute| Tools
```

The brain of the system is a small language model (hello LFM!) that can map English sentences to the right tool calls.

- `toggle_lights`: turn lights on or off in a specific room
- `set_thermostat`: change the temperature and operating mode
- `lock_door`: lock or unlock a door
- `get_device_status`: read the current state of any device
- `set_scene`: activate a preset that adjusts multiple devices at once

and

- `intent_unclear`: the most important tool for robustness. The model must call it whenever the request falls outside what the system can handle, whether the request is ambiguous, off-topic (ordering food, asking about the weather), incomplete (a pronoun with no prior context like "turn it on"), or refers to an unsupported device like a TV or camera. Getting this tool right is what separates a reliable assistant from one that hallucinates actions.


The sequence diagram below shows how the system starts and processes a chat message step by step. Solid arrows are calls, dashed arrows are responses:

```mermaid
sequenceDiagram
    participant Browser
    participant FastAPI as FastAPI server
    participant Agent as Agent loop
    participant Tools
    participant LFM as llama.cpp server

    Note over Browser,LFM: Startup
    Browser->>FastAPI: select model
    FastAPI->>LFM: start process (background thread)
    LFM-->>FastAPI: ready

    Note over Browser,LFM: Chat request
    Browser->>FastAPI: POST /chat
    FastAPI->>Agent: run(message, history)
    Agent->>LFM: inference request (with tool schemas)
    LFM-->>Agent: tool call
    Agent->>Tools: execute tool
    Tools-->>Agent: result
    Agent->>LFM: inference request (with tool result)
    LFM-->>Agent: text response
    Agent-->>FastAPI: text response
    FastAPI-->>Browser: text response
    Browser->>FastAPI: GET /state
    FastAPI-->>Browser: updated home state
```

The FastAPI server, the agent loop, and the tools are all implemented in Python. That said, feel free to re-implement them in any other language for higher performance. Rust, for example, would be a good choice.

## Step 2: Benchmarking tool-calling accuracy <a name="benchmark"></a>

Play with the UI using one of the local models and you will quickly notice: 

- sometimes it works

  ![Happy path](assets/happy_path.gif)

- sometimes it doesn't.

  ![Unhappy path](assets/unhappy_path.gif)

That's fine for a proof of concept. But the full power of small language models only comes out
  when you fine-tune them.

  Before you fine-tune, though, you need to know where you stand. You need to measure. You cannot ship to production based on vibes or things that more or less work. You ship based on good benchmarks and evals.


### What's a good benchmark?

A good benchmark covers the space of possible inputs by systematic taxonomy, not intuition. Here is the methodology we use to build `benchmark/`, a 100-task suite designed from the ground up around these principles.

**1. Start with a taxonomy**

Define the input space BEFORE writing prompts. A taxonomy makes coverage gaps visible and prevents accidental clustering around the examples you happened to think of first.

Our taxonomy has three dimensions:

| Dimension | Values |
|-----------|--------|
| Capability | `lights`, `thermostat`, `doors`, `status`, `scene`, `rejection`, `multi_tool` |
| Phrasing | `imperative`, `colloquial`, `implicit`, `question` |
| Inference depth | `literal` (words map 1:1 to tool + args), `semantic` (requires translation), `boundary` (model must reject) |

**2. Sample from every cell**

The Cartesian product of those dimensions defines the universe of task types. Sample at least one task per non-empty cell. This forces you to write prompts you would not have thought of otherwise, such as 
- an implicit-semantic thermostat request ("It feels like a sauna in here") or
- a boundary-case door request ("Is the house secure right now?").

**3. Write programmatic verifiers**

Every task has a pure Python verifier that inspects

- the final `home_state` dict, or
- captured `tool_calls` for read-only and rejection tasks.

No LLM-as-judge. Deterministic, fast, cheap.

```python
# State check: was the right final state reached?
passed = state["lights"]["kitchen"]["state"] == "on"

# Tool-call check (for status queries and rejections): was the right tool called with the right args?
call = _find_last_call(tool_calls, "intent_unclear")
passed = call is not None and call["args"].get("reason") == "off_topic"
```

You can run the benchmark for a given model as follows:

```bash
uv run python benchmark/run.py \
    --hf-repo LiquidAI/LFM2.5-1.2B-Instruct-GGUF \
    --hf-file LFM2.5-1.2B-Instruct-Q4_0.gguf
```

**Run a single task by number (1-101)**, for example:

```bash
uv run python benchmark/run.py \
    --hf-repo LiquidAI/LFM2.5-1.2B-Instruct-GGUF \
    --hf-file LFM2.5-1.2B-Instruct-Q4_0.gguf \
    --task 5
```

It's also worth running the benchmark against a frontier model like GPT-4o-mini.

  Why? Because a frontier model scoring near-perfect tells you the agent harness is correct. The
  prompts, the tool schemas, the verification logic. If a state-of-the-art model doesn't pass almost
  everything, the problem is not the model. The problem is your code.


**Run against OpenAI gpt-4o-mini** (requires `OPENAI_API_KEY` in `.env`):

```bash
uv run python benchmark/run.py --backend openai
```

Results are printed to the console and saved as a Markdown file in `benchmark/results/`.

**Evaluation results**

| Model | Parameters | Score | Accuracy |
|-------|------------|-------|----------|
| gpt-4o-mini | n/a | 93/100 | 93% |
| LFM2.5-1.2B-Instruct Q4_0 | 1.2B | 71/100 | 71% |
| LFM2-350M Q8_0 | 350M | 28/100 | 28% |

These are not vibes anymore. These are actual numbers we can use to understand where we stand.

In the following sections, we will see how to improve the performance of our local LFM models to bridge the gap with gpt-4o-mini.


## Step 3: Generate synthetic data <a name="step-3-generate-synthetic-data"></a>

To fine-tune the model you need labelled training data. We generate it synthetically: `gpt-4o-mini` produces user utterances, each one validated by a second independent inference pass through the real agent.

The target is 500 examples distributed across the same taxonomy as the benchmark, with heavier sampling on the hardest cells (`rejection`, `multi_tool`) where local models lose the most points.

**The contamination problem**

If a benchmark utterance leaks into training, the model can memorise the correct answer instead of learning the underlying skill. Scores would look great on paper but overestimate real-world performance. We prevent this with a four-layer pipeline:

```mermaid
flowchart TD
    A["Generate candidates\n(prompt includes benchmark blocklist)"]
    A --> B{"Exact or substring\nmatch with benchmark?"}
    B -->|yes| R1((discard))
    B -->|no| C{"Trigram Jaccard\nsimilarity > 0.5?"}
    C -->|yes| R2((discard))
    C -->|no| D{"Agent cross-validation:\ngpt-4o-mini agrees?"}
    D -->|no| R3((discard))
    D -->|yes| E[("sft_data.jsonl")]
```

Layer 1 lives in the generation prompt itself: every benchmark utterance for the relevant taxonomy cell is listed and the model is told not to reproduce them. Layers 2 and 3 are deterministic post-generation checks. Layer 4 sends each candidate to the real agent and keeps only the examples where the agent's tool call matches the generator's expected answer, which also filters out genuinely ambiguous phrasings that produce inconsistent results.

**Generate the dataset**

```bash
# Dry run: show the generation plan without calling the API
uv run python benchmark/datasets/generate.py --dry-run

# Generate 500 examples (default)
uv run python benchmark/datasets/generate.py

# Custom count and output path
uv run python benchmark/datasets/generate.py --count 500 --output benchmark/datasets/sft_data.jsonl
```

Output goes to `benchmark/datasets/sft_data.jsonl` (gitignored). After generation the script prints a rejection breakdown and a coverage matrix so you can see exactly how many examples ended up in each taxonomy cell.

## Step 4: Fine-tune the model <a name="step-4-fine-tune-the-model"></a>

Fine-tuning adapts the base model to our specific task. Instead of retraining all weights from scratch, we use LoRA (Low-Rank Adaptation): a technique that injects a small set of trainable weight matrices on top of the frozen base model. This keeps GPU memory usage low and training fast, while still producing meaningful accuracy gains on the target task.

Training runs on [Modal](https://modal.com) (a serverless GPU cloud) via [leap-finetune](https://github.com/Liquid4All/leap-finetune), Liquid AI's fine-tuning tool. The full loop:

1. Convert the synthetic dataset to the format the LFM2 tokenizer expects, then push it to HuggingFace Hub.
2. Run LoRA training on an H100 via leap-finetune.
3. Download the checkpoint from Modal, merge the adapter into the base model, and convert to GGUF.
4. Re-run the benchmark and compare against the baseline.

### Data preparation

`sft_data.jsonl` stores assistant tool calls in OpenAI format: a `tool_calls` list with `"content": null`. The LFM2 tokenizer's chat template expects tool calls as plain text content wrapped in special tokens:

```
<|tool_call_start|>[{"name": "toggle_lights", "arguments": {"room": "living_room", "state": "on"}}]<|tool_call_end|>
```

`finetune/prepare_data.py` converts every assistant message to this format, then splits the 491 examples 80/20 stratified by capability (so each category is proportionally represented in both train and eval), and pushes the result to HuggingFace Hub.

| Capability | Total | Train | Eval |
|------------|-------|-------|------|
| lights | 120 | 96 | 24 |
| doors | 80 | 64 | 16 |
| thermostat | 71 | 57 | 14 |
| rejection | 60 | 48 | 12 |
| multi_tool | 60 | 48 | 12 |
| status | 50 | 40 | 10 |
| scene | 50 | 40 | 10 |
| **Total** | **491** | **393** | **98** |

### Training configuration

`finetune/configs/` contains one YAML per model. Both extend the leap-finetune defaults and point at the same dataset. The key differences reflect each model's starting point:

| Config | Model | Epochs | Batch size | Rationale |
|--------|-------|--------|------------|-----------|
| `350M.yaml` | LFM2-350M | 5 | 4 | Starts at 28%, needs more passes over the data to absorb the signal. |
| `1.2B.yaml` | LFM2.5-1.2B-Instruct | 3 | 2 | Already at 71%, 3 epochs is enough to close the gaps without catastrophic forgetting. |

Both use `learning_rate: 2e-4`, LoRA adapters via `DEFAULT_LORA`, and stream training curves to a [Trackio](https://huggingface.co/trackio) dashboard in real time.

Expected Modal cost: roughly $1.50 for the 350M model and $3.00 for the 1.2B, well within the $30 monthly free credit.

### Checkpoint download and export

Once training finishes, the LoRA checkpoint is stored in a Modal Volume. After downloading it locally, `finetune/export.py` merges the adapter weights into the frozen base model, producing a standard HuggingFace model directory. That directory is then converted to GGUF with `llama.cpp`'s conversion script, using the same quantization type as the original benchmark model for a fair before-and-after comparison.

### Commands

**One-time setup**

```bash
# Clone and install leap-finetune (run once, from the home-assistant directory)
git clone https://github.com/Liquid4All/leap-finetune
cd leap-finetune && uv sync && cd -

# Authenticate
huggingface-cli login
modal setup
```

**Prepare the data**

```bash
uv run --group finetune python finetune/prepare_data.py
```

**Train on Modal** (run from the `leap-finetune/` subdirectory)

```bash
cd leap-finetune

# 350M model
uv run leap-finetune ../finetune/configs/350M.yaml

# 1.2B model
uv run leap-finetune ../finetune/configs/1.2B.yaml
```

**Download checkpoints** (run from the `leap-finetune/` subdirectory)

```bash
# Verify actual paths first
uv run modal volume ls leap-finetune

uv run modal volume get leap-finetune /outputs/home-assistant-350M ../finetune/output/350M-lora
uv run modal volume get leap-finetune /outputs/home-assistant-1.2B ../finetune/output/1.2B-lora
```

**Merge LoRA adapters and convert to GGUF** (run from the `home-assistant` directory)

```bash
# Merge adapters into the base models
uv run --group export python finetune/export.py \
    --lora-path finetune/output/350M-lora \
    --output-path finetune/output/350M-merged

uv run --group export python finetune/export.py \
    --lora-path finetune/output/1.2B-lora \
    --output-path finetune/output/1.2B-merged

# Convert to GGUF (requires llama.cpp)
python /path/to/llama.cpp/convert_hf_to_gguf.py \
    finetune/output/350M-merged --outtype q8_0 \
    --outfile models/LFM2-350M-finetuned-Q8_0.gguf

python /path/to/llama.cpp/convert_hf_to_gguf.py \
    finetune/output/1.2B-merged --outtype q4_0 \
    --outfile models/LFM2.5-1.2B-finetuned-Q4_0.gguf
```

**Re-run the benchmark**

```bash
uv run python benchmark/run.py \
    --hf-repo LiquidAI/LFM2-GGUF \
    --hf-file LFM2-350M-finetuned-Q8_0.gguf

uv run python benchmark/run.py \
    --hf-repo LiquidAI/LFM2.5-1.2B-Instruct-GGUF \
    --hf-file LFM2.5-1.2B-finetuned-Q4_0.gguf
```
