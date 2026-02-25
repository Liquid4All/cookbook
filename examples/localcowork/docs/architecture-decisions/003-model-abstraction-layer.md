# ADR-003: Model Abstraction Layer (OpenAI-Compatible API)

## Status
Accepted

## Context
LocalCowork must work with multiple LLM backends:
- **Development:** Qwen2.5-32B-Instruct via Ollama (localhost:11434)
- **Production target:** LFM2.5-24B via llama.cpp, MLX, or LEAP SDK
- **Fallbacks:** Qwen3-30B-A3B (MoE), Mistral Small 24B

The application code must not change when the model changes. The model swap should be a configuration change only.

## Decision
The Agent Core communicates with the inference layer exclusively via an OpenAI-compatible chat completions API at localhost.

## API Contract

All inference backends must expose this endpoint:

```
POST http://localhost:{port}/v1/chat/completions

{
  "model": "{model-name}",
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "...", "tool_calls": [...] },
    { "role": "tool", "tool_call_id": "...", "content": "..." }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "filesystem.list_dir",
        "description": "...",
        "parameters": { ... }
      }
    }
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 4096
}
```

### Tool Call Response Format

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_001",
        "type": "function",
        "function": {
          "name": "filesystem.list_dir",
          "arguments": "{\"path\": \"/Users/chintan/Documents\"}"
        }
      }]
    }
  }]
}
```

## Model-Specific Adaptations

The only model-specific code lives in the Inference Client configuration:

| Setting | Qwen2.5-32B | LFM2.5-24B | Qwen3-30B-A3B |
|---------|-------------|------------|----------------|
| Port | 11434 (Ollama) | 8080 (llama.cpp) | 11434 (Ollama) |
| Tool call format | Native JSON | JSON mode via system prompt | Native JSON |
| Temperature | 0.7 | 0.7 (TBD) | 0.7 |
| Max tokens | 4096 | 4096 | 4096 |
| Context window | 32768 | 32768 | 32768 |

### LFM2.5 Tool Call Normalization

LFM2.5 natively outputs Pythonic function calls. The Inference Client includes a normalizer:

```
LFM2.5 output: filesystem.list_dir(path="/Users/chintan/Documents")
Normalized to: {"name": "filesystem.list_dir", "arguments": {"path": "/Users/chintan/Documents"}}
```

This normalization is transparent to the Agent Core — it always receives standard JSON tool calls.

## Model Swap Procedure

1. Download the new model (GGUF Q4_K_M recommended for LFM2.5-24B)
2. Update `_models/config.yaml`:
   ```yaml
   active_model: lfm25-24b
   models:
     lfm25-24b:
       path: ~/.localcowork/models/lfm25-24b-q4_k_m.gguf
       runtime: llama.cpp
       port: 8080
       context_window: 32768
       tool_call_format: pythonic  # triggers normalizer
   ```
3. Restart the inference backend
4. Run `/model-test` to validate tool-calling accuracy
5. Compare results with previous model's baseline

No application code changes needed.

## Rationale
The OpenAI chat completions API has become the de facto standard for LLM inference. Ollama, llama.cpp, vLLM, and most frameworks support it natively. This means:
- No custom client code per backend
- Community tooling (monitoring, testing) works out of the box
- Future models that support this API work without changes

## Trade-offs
- The OpenAI API doesn't expose all model-specific features (e.g., LFM2.5's hybrid attention/convolution architecture)
- Tool call format normalization adds a small processing step for non-standard models
- Streaming behavior may differ subtly across backends (handled in the streaming module)
