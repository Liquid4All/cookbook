# Home assistant

An example of a locally running home assistant that can

- act on your voice
- remember your preferences
- ask you questions

## Quickstart

**1. Start the model server**

Pick a model and quantization level. llama-server downloads and caches it automatically:

```bash
# Model options:
#   LiquidAI/LFM2.5-1.2B-Instruct-GGUF
#   LiquidAI/LFM2.5-1.2B-Thinking-GGUF
#
# Quantization options (smallest to largest):
#   Q4_0 (696 MB), Q4_K_M (731 MB), Q5_K_M (843 MB),
#   Q6_K (963 MB), Q8_0 (1.25 GB)

llama-server \
  --hf-repo LiquidAI/LFM2.5-1.2B-Instruct-GGUF \
  --hf-file LFM2.5-1.2B-Instruct-Q4_0.gguf \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 99
```

**2. Install dependencies**

```bash
uv sync
```

**3. Start the app server**

```bash
uv run uvicorn app.server:app --port 5173 --reload
```

**4. Open the app**

```bash
open http://localhost:5173
```

## Agent design notes

Two patterns are essential when building multi-turn tool-calling apps with small models.

### 1. Maintain conversation history across requests

Each request must carry the full prior conversation, not just the current message. Without
it, the model has no context to resolve references like "it", "that room", or "the same
setting" and will guess incorrectly.

The server keeps a `conversation_history` list and prepends it to every `messages` array:

```python
# server.py
conversation_history: list[dict] = []

text = run_agent(req.message, history=conversation_history, on_tool_call=on_tool_call)
conversation_history.append({"role": "user",      "content": req.message})
conversation_history.append({"role": "assistant", "content": text})
```

Only plain `user`/`assistant` text turns are stored. Internal tool-call and tool-result
messages from the agent loop are not included.

### 2. Force a text response after tool execution with `tool_choice="none"`

Small models often re-issue the same tool call after receiving the result, instead of
generating a text confirmation. The agent loop detects this duplicate and breaks early.
A final API call with `tool_choice="none"` then forces the model to summarise what it
just did in plain text.

```python
# agent.py
if duplicate:
    break  # exit the tool loop cleanly

# forced text summary
final = client.chat.completions.create(
    model="local",
    messages=messages,
    tools=TOOL_SCHEMAS,
    tool_choice="none",
    temperature=0.1,
    max_tokens=256,
)
return final.choices[0].message.content or "Done."
```

Without this, the fallback is always an error string because a tool-call-only response
has `message.content = None`.
