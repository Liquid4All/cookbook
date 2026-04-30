# Tool Calling Normalizer for LFM2/LFM2.5

This package fixes issue #72 by normalizing LFM2/LFM2.5 tool-call output from inconsistent Pythonic, JSON, and XML embeddings into OpenAI-compatible function call data.

## Installation

```bash
pip install transformers torch
```

## Usage

### Normalizer standalone

```python
from lfm2_tool_normalizer import LFMToolNormalizer

normalizer = LFMToolNormalizer()
result = normalizer.parse("<|tool_call_start|>do_something(a=1)<|tool_call_end|>")
print(result.tool_calls[0].to_openai_dict())
```

### Agent loop with plain Python functions

```python
from lfm2_agent_loop import LFMAgentLoop

from quick_start import get_time, add
```

The `LFMAgentLoop` accepts plain Python callables and will execute returned tool calls automatically.

### Tool list compatibility

The agent loop also accepts LangChain `BaseTool` and smolagents `Tool` objects if they expose `name` and `func`/`run`.

## Benchmark results

| Task | Baseline | Normalized | Format |
|---|---|---|---|
| Tokenized Pythonic | FAIL | OK | pythonic |
| Tokenized JSON | FAIL | OK | json |
| Tokenized XML | FAIL | OK | xml |
| Bare Pythonic | FAIL | OK | pythonic |
| Bare JSON | OK | OK | json |
| Bare XML | FAIL | OK | xml |
| Surrounding text | FAIL | OK | json |
| Multi-tool JSON | FAIL | OK | json |
| Bare command | FAIL | OK | pythonic |
| Mixed XML tokens | FAIL | OK | xml |

baseline success: 1/10
normalized success: 10/10
delta: 9
tasks rescued: 9

## Run benchmark

```bash
python examples/tool-calling-normalizer/benchmark.py
```

## Link

Issue #72: https://github.com/Liquid4All/cookbook/issues/72
