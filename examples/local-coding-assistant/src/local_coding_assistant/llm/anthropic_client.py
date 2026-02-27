import anthropic

from ..config import Config
from .base import LLMResponse


class AnthropicClient:
    """Anthropic API backend."""

    def __init__(self, config: Config) -> None:
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._model = config.anthropic_model
        self._max_tokens = config.max_tokens

    def chat(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse:
        # Translate neutral tool schema (parameters) -> Anthropic format (input_schema)
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
            tools=anthropic_tools,
        )

        # Normalize content blocks to plain dicts
        content: list[dict] = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return LLMResponse(
            stop_reason=response.stop_reason or "end_turn",
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
