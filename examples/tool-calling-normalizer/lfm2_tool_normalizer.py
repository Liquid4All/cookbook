import ast
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

TOKEN_RE = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.S)
XML_TOOL_RE = re.compile(r"<tool_call\b.*?</tool_call>", re.S)
PYTHON_CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\((.*)\)$", re.S)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]

    def to_openai_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class NormalizationResult:
    tool_calls: List[ToolCall]
    format_detected: str
    success: bool
    error: Optional[str] = None


class LFMToolNormalizer:
    def parse(self, text: str) -> NormalizationResult:
        content_list = self._extract_token_content(text)
        if not content_list:
            return NormalizationResult([], "none", False, "no tool call tokens or content found")

        for content in content_list:
            content = content.strip()
            for parser in (self._parse_json, self._parse_xml, self._parse_pythonic):
                try:
                    calls = parser(content)
                except ValueError as exc:
                    continue
                if calls:
                    return NormalizationResult(calls, parser.__name__[7:], True, None)

        return NormalizationResult([], "unknown", False, "unable to normalize tool call")

    def _extract_token_content(self, text: str) -> List[str]:
        matches = TOKEN_RE.findall(text)
        if matches:
            return [match.strip() for match in matches if match.strip()]
        return [text]

    def _parse_json(self, content: str) -> List[ToolCall]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("json decode failed") from exc

        if isinstance(payload, dict):
            if payload.get("name") and isinstance(payload.get("arguments"), dict):
                return [ToolCall(payload["name"], payload["arguments"])]
            if payload.get("tool_call") and isinstance(payload["tool_call"], dict):
                inner = payload["tool_call"]
                return [ToolCall(inner["name"], inner.get("arguments", {}))]
            raise ValueError("json object missing tool call fields")

        if isinstance(payload, list):
            calls: List[ToolCall] = []
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError("json list item is not object")
                name = item.get("name")
                args = item.get("arguments")
                if not name or not isinstance(args, dict):
                    raise ValueError("json list item missing name or arguments")
                calls.append(ToolCall(name, args))
            return calls

        raise ValueError("json payload is not tool call structure")

    def _parse_xml(self, content: str) -> List[ToolCall]:
        raw = content if content.strip().startswith("<tool_call") else "".join(XML_TOOL_RE.findall(content))
        if not raw:
            raise ValueError("no xml tool_call found")

        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            raise ValueError("xml parse failed") from exc

        elements = [root] if root.tag == "tool_call" else list(root.findall(".//tool_call"))
        calls: List[ToolCall] = []
        for element in elements:
            name = element.findtext("name") or element.attrib.get("name")
            if not name:
                raise ValueError("xml tool_call missing name")
            arguments = self._xml_arguments(element)
            calls.append(ToolCall(name, arguments))
        return calls

    def _xml_arguments(self, element: ET.Element) -> Dict[str, Any]:
        arguments = {}
        container = element.find("arguments")
        source = container if container is not None else element
        for child in source:
            if child.tag == "name":
                continue
            arguments[child.tag] = self._xml_to_value(child)
        return arguments

    def _xml_to_value(self, element: ET.Element) -> Any:
        if list(element):
            if all(child.tag == "item" for child in element):
                return [self._xml_to_value(child) for child in element]
            return {child.tag: self._xml_to_value(child) for child in element}
        text = element.text or ""
        text = text.strip()
        if not text:
            return {}
        if text.lower() in {"true", "false", "null"}:
            return json.loads(text.lower())
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text

    def _parse_pythonic(self, content: str) -> List[ToolCall]:
        try:
            tree = ast.parse(content, mode="eval")
        except SyntaxError:
            try:
                tree = ast.parse(content, mode="exec")
            except SyntaxError as exc:
                raise ValueError("pythonic parse failed") from exc

        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._call_name(node.func)
                args = self._call_arguments(node)
                calls.append(ToolCall(name, args))
        if not calls:
            raise ValueError("no pythonic calls found")
        return calls

    def _call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        raise ValueError("unsupported call name")

    def _call_arguments(self, call: ast.Call) -> Dict[str, Any]:
        arguments: Dict[str, Any] = {}
        for index, arg in enumerate(call.args):
            arguments[f"arg{index}"] = self._safe_literal(arg)
        for keyword in call.keywords:
            if keyword.arg is None:
                continue
            arguments[keyword.arg] = self._safe_literal(keyword.value)
        return arguments

    def _safe_literal(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Dict):
            return {
                self._safe_literal(key): self._safe_literal(value)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.List):
            return [self._safe_literal(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return [self._safe_literal(item) for item in node.elts]
        if isinstance(node, ast.Set):
            return [self._safe_literal(item) for item in node.elts]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._safe_literal(node.operand)
            if isinstance(value, (int, float)):
                return -value
        raise ValueError("unsupported literal")
