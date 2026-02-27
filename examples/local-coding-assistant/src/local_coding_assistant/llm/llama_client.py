import json

from openai import OpenAI

from ..config import Config
from .base import LLMResponse


class LlamaClient:
    """llama.cpp backend via OpenAI-compatible REST API."""

    def __init__(self, config: Config) -> None:
        self._client = OpenAI(
            base_url=config.llama_base_url,
            api_key=config.llama_api_key,
        )
        self._model = config.llama_model
        self._max_tokens = config.max_tokens

    def chat(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse:
        # OpenAI format: system message prepended to the messages list
        openai_messages = [{"role": "system", "content": system}] + messages

        # Translate tool_result blocks (Anthropic format) to tool role messages (OpenAI format)
        translated: list[dict] = []
        for msg in openai_messages:
            if isinstance(msg.get("content"), list):
                # Check if this is a list of tool_result blocks
                blocks = msg["content"]
                if blocks and blocks[0].get("type") == "tool_result":
                    for block in blocks:
                        translated.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block["content"],
                        })
                    continue
                # Assistant message with content blocks -> extract text + tool_calls
                if blocks and blocks[0].get("type") in ("text", "tool_use"):
                    text_parts = [b["text"] for b in blocks if b["type"] == "text"]
                    tool_calls = [
                        {
                            "id": b["id"],
                            "type": "function",
                            "function": {
                                "name": b["name"],
                                "arguments": json.dumps(b["input"]),
                            },
                        }
                        for b in blocks
                        if b["type"] == "tool_use"
                    ]
                    openai_msg: dict = {"role": "assistant", "content": " ".join(text_parts) or None}
                    if tool_calls:
                        openai_msg["tool_calls"] = tool_calls
                    translated.append(openai_msg)
                    continue
            translated.append(msg)

        # OpenAI tools format wraps schema in {"type": "function", "function": {...}}
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=translated,  # type: ignore[arg-type]
            tools=openai_tools,  # type: ignore[arg-type]
        )

        choice = response.choices[0]
        msg = choice.message

        # Normalize to the same content block format used by AnthropicClient
        content: list[dict] = []
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        if msg.tool_calls:
            for tc in msg.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })

        stop_reason = "tool_use" if msg.tool_calls else "end_turn"

        usage = response.usage
        return LLMResponse(
            stop_reason=stop_reason,
            content=content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
