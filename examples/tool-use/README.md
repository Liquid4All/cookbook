# Tool calling with LFM2 models

## The basics

Tool calling with Transformers passing tools as Python callables.
```
uv run 01_tool_call_with_python_callables.py --model-id LiquidAI/LFM2-1.2B
uv run 01_tool_call_with_python_callables.py --model-id LiquidAI/LFM2-350M
uv run 01_tool_call_with_python_callables.py --model-id LiquidAI/LFM2-1.2B-Tool
```

Tool calling with Transformers passing tools as JSON schemas:
```
uv run 02_tool_call_with_json_schemas.py --model-id LiquidAI/LFM2-1.2B
uv run 02_tool_call_with_json_schemas.py --model-id LiquidAI/LFM2-350M
uv run 02_tool_call_with_json_schemas.py --model-id LiquidAI/LFM2-1.2B-Tool
```

## Multi-turn tool use

Can these models sequentially call different tools to compile all necessary context to respond the user query?

```
uv run 03_iterative_tool_calling.py --model-id LiquidAI/LFM2-1.2B
```

## Inferece with llama.cpp

Start the llama.cpp server
```shell
llama-server -hf LiquidAI/LFM2-1.2B-Tool-GGUF --jinja --temp 0
```

Run examples:
```
uv run 04_toll_call_with_llama_cpp.py
```





