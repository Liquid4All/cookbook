# Tool calling with LFM2 models and llama.cpp

## What's the problem?

LFM2 models have been post-trained to output tool calls in Python format.

See this example taken from the [LFM2-1.2B-Tool model card](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool)

```
<|startoftext|><|im_start|>system
List of tools: <|tool_list_start|>[{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|tool_list_end|><|im_end|>
<|im_start|>user
What is the current status of candidate ID 12345?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
```

When serving an LFM model using an OpeanAI compatible server, for example `llama-server` the inference engine needs to parse the list of tool calls `[get_candidate_status(candidate_id="12345")]` into the corresponding JSON format

```
[
    {
      "id": "call_0",
      "type": "function",
      "function": {
        "name": "get_candidate_status",
        "arguments": "{\"candidate_id\": \"12345\"}"
      }
    }
  ]
```
If the inference engine cannot parse these, the downstream framework/tool that connects to this llama-server (e.g. [Agno](https://github.com/agno-agi/agno)) will report no tool calls, and the model will not work.


## Experiments

```python
user_prompt = "What are the time rightnow ?"
```

### `LFM2-1.2B-Tool-GGUF:Q8_0`

```
# does not work
uv run main.py --user-prompt "What are the time rightnow ?" --model "LiquidAI/LFM2-1.2B-Tool-GGUF:Q8_0"

# works
uv run main.py --user-prompt "What are the time rightnow ?" --model "LiquidAI/LFM2-1.2B-Tool-GGUF:Q8_0" --system-prompt naive

# works
uv run main.py --user-prompt "What are the time rightnow ?" --model "LiquidAI/LFM2-1.2B-Tool-GGUF:Q8_0" --system-prompt use-json-schema
```

### `LFM2.5-1.2B-Thinking-GGUF:Q8_0`
```
# does not work
uv run main.py --user-prompt "What are the time rightnow ?" --model "LiquidAI/LFM2.5-1.2B-Thinking-GGUF:Q8_0"

# works
uv run main.py --user-prompt "What are the time rightnow ?" --model "LiquidAI/LFM2.5-1.2B-Thinking-GGUF:Q8_0" --system-prompt naive

# works
uv run main.py --user-prompt "What are the time rightnow ?" --model "LiquidAI/LFM2.5-1.2B-Thinking-GGUF:Q8_0" --system-prompt use-json-schema
```

### `LFM2.5-1.2B-Instruct-GGUF:Q8_0`
```
# does not work
uv run main.py --user-prompt "What are the time rightnow ?" --model "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q8_0"

# works
uv run main.py --user-prompt "What are the time rightnow ?" --model "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q8_0" --system-prompt naive

# does not work
uv run main.py --user-prompt "What are the time rightnow ?" --model "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q8_0" --system-prompt use-json-schema
```