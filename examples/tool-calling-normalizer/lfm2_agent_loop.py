from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from lfm2_tool_normalizer import LFMToolNormalizer, ToolCall


class LFMAgentLoop:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        tools: Sequence[Any],
        system_prompt: str = "",
        force_json: bool = False,
        max_turns: int = 3,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.tools = {tool["name"]: tool["func"] for tool in self._normalize_tools(tools)}
        self.system_prompt = system_prompt
        self.force_json = force_json
        self.max_turns = max_turns
        self.device = device
        self.normalizer = LFMToolNormalizer()

    def _normalize_tools(self, tools: Sequence[Any]) -> List[Dict[str, Any]]:
        normalized = []
        for tool in tools:
            if callable(tool):
                normalized.append({"name": getattr(tool, "__name__", "tool"), "func": tool})
                continue
            name = getattr(tool, "name", None)
            func = getattr(tool, "func", None) or getattr(tool, "run", None) or getattr(tool, "__call__", None)
            if name and callable(func):
                normalized.append({"name": name, "func": func})
                continue
            raise ValueError("unsupported tool type")
        return normalized

    def run(self, query: str) -> str:
        prompt = self._build_prompt(query)
        for turn in range(self.max_turns):
            output = self._generate(prompt)
            result = self.normalizer.parse(output)
            if not result.success or not result.tool_calls:
                return output
            tool_outputs = []
            for tool_call in result.tool_calls:
                tool_outputs.append(self._execute_tool(tool_call))
            prompt = self._build_tool_feedback(query, output, tool_outputs)
        return output

    def _build_prompt(self, query: str) -> str:
        lines = [self.system_prompt.strip()] if self.system_prompt else []
        if self.force_json:
            lines.append("Output function calls as JSON.")
        if self.tools:
            lines.append("Available tools: " + ", ".join(self.tools.keys()))
        lines.append(f"User: {query}")
        return "\n".join(lines)

    def _build_tool_feedback(self, query: str, previous_output: str, tool_outputs: List[str]) -> str:
        lines = [self.system_prompt.strip()] if self.system_prompt else []
        if self.force_json:
            lines.append("Output function calls as JSON.")
        lines.append("Available tools: " + ", ".join(self.tools.keys()))
        lines.append(f"User: {query}")
        lines.append(f"Assistant: {previous_output}")
        lines.append("Tool outputs:")
        lines.extend(tool_outputs)
        return "\n".join(lines)

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _generate(self, prompt: str) -> str:
        inputs = self._encode(prompt)
        outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _execute_tool(self, tool_call: ToolCall) -> str:
        func = self.tools.get(tool_call.name)
        if func is None:
            return f"Tool {tool_call.name} not found"
        try:
            result = func(**tool_call.arguments)
        except TypeError as exc:
            return f"Tool {tool_call.name} failed: {exc}"
        return f"{tool_call.name}: {result}"
