from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    stop_reason: str      # "end_turn" | "tool_use"
    content: list[dict]   # mix of text blocks and tool_use blocks
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    """
    Minimal protocol that both Anthropic and llama.cpp backends implement.
    The agentic loop only talks to this interface.
    """

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
    ) -> LLMResponse: ...
