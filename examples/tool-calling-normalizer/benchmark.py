import json
from typing import Any, Dict, List, Tuple

from lfm2_tool_normalizer import LFMToolNormalizer, ToolCall


class BaselineJSONParser:
    def parse(self, text: str) -> Tuple[bool, List[ToolCall]]:
        content = self._extract_token_content(text)
        if content is None:
            return False, []
        try:
            payload = json.loads(content)
            return True, self._payload_to_calls(payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False, []

    def _extract_token_content(self, text: str) -> Any:
        start = text.find("<|tool_call_start|>")
        end = text.find("<|tool_call_end|>")
        if start != -1 and end != -1:
            return text[start + 18 : end].strip()
        return text.strip()

    def _payload_to_calls(self, payload: Any) -> List[ToolCall]:
        if isinstance(payload, dict):
            if payload.get("name") and isinstance(payload.get("arguments"), dict):
                return [ToolCall(payload["name"], payload["arguments"])]
        if isinstance(payload, list):
            return [ToolCall(item["name"], item["arguments"]) for item in payload if isinstance(item, dict)]
        raise ValueError("invalid payload")


TEST_CASES: List[Tuple[str, str]] = [
    (
        "Tokenized Pythonic",
        "<|tool_call_start|>do_math(a=2, b=3)<|tool_call_end|>",
    ),
    ("Tokenized JSON", "<|tool_call_start|>[{\"name\": \"weather\", \"arguments\": {\"city\": \"Paris\"}}]<|tool_call_end|>"),
    (
        "Tokenized XML",
        "<|tool_call_start|><tool_call><name>echo</name><arguments><message>Hello</message></arguments></tool_call><|tool_call_end|>",
    ),
    ("Bare Pythonic", "compute(sum=10, flag=True)"),
    (
        "Bare JSON",
        "[{\"name\": \"translate\", \"arguments\": {\"text\": \"hi\"}}]",
    ),
    (
        "Bare XML",
        "<tool_call><name>status</name><arguments><code>200</code></arguments></tool_call>",
    ),
    (
        "Surrounding text",
        "Please call the tool: <|tool_call_start|>{\"name\": \"search\", \"arguments\": {\"query\": \"weather\"}}<|tool_call_end|> thank you.",
    ),
    (
        "Multi-tool JSON",
        "<|tool_call_start|>[{\"name\": \"first\", \"arguments\": {}}, {\"name\": \"second\", \"arguments\": {\"value\": 1}}]<|tool_call_end|>",
    ),
    ("Bare command", "ping()"),
    (
        "Mixed XML tokens",
        "<|tool_call_start|>\n<tool_call><name>lookup</name><arguments><term>agent</term></arguments></tool_call>\n<|tool_call_end|>",
    ),
]


def run_benchmark() -> None:
    normalizer = LFMToolNormalizer()
    baseline = BaselineJSONParser()
    rows: List[Tuple[str, str, str, str]] = []
    baseline_success = 0
    normalized_success = 0

    for name, text in TEST_CASES:
        base_ok, base_calls = baseline.parse(text)
        norm = normalizer.parse(text)
        baseline_success += int(base_ok)
        normalized_success += int(norm.success)
        rows.append(
            (
                name,
                "OK" if base_ok else "FAIL",
                "OK" if norm.success else "FAIL",
                norm.format_detected,
            )
        )

    self_count = len(TEST_CASES)
    rescued = normalized_success - baseline_success
    print_table(rows)
    print()
    print(f"baseline success: {baseline_success}/{self_count}")
    print(f"normalized success: {normalized_success}/{self_count}")
    print(f"delta: {normalized_success - baseline_success}")
    print(f"tasks rescued: {rescued}")


def print_table(rows: List[Tuple[str, str, str, str]]) -> None:
    widths = [max(len(cell) for cell in column) for column in zip(*(rows + [("Task", "Baseline", "Normalized", "Format")]))]
    headers = ["Task", "Baseline", "Normalized", "Format"]
    line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    separator = " | ".join("-" * width for width in widths)
    print(line)
    print(separator)
    for row in rows:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))


if __name__ == "__main__":
    run_benchmark()
