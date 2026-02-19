"""
JSON-RPC 2.0 Transport Utilities — Python

Low-level JSON-RPC message handling for MCP server communication.
Used by mcp_base.py; typically not imported directly by tool implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class JsonRpcRequest:
    """A JSON-RPC 2.0 request."""

    jsonrpc: str
    id: str | int
    method: str
    params: dict[str, Any] | None = None


@dataclass
class JsonRpcResponse:
    """A JSON-RPC 2.0 response."""

    jsonrpc: str = "2.0"
    id: str | int = 0
    result: Any = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


def success_response(request_id: str | int, result: Any) -> dict[str, Any]:
    """Create a JSON-RPC success response."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(
    request_id: str | int,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    """Create a JSON-RPC error response."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a JSON-RPC notification (no id, no response expected)."""
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def is_valid_request(msg: dict[str, Any]) -> bool:
    """Validate that a parsed dict is a valid JSON-RPC 2.0 request."""
    return (
        isinstance(msg, dict)
        and msg.get("jsonrpc") == "2.0"
        and isinstance(msg.get("method"), str)
        and ("id" in msg and isinstance(msg["id"], (str, int)))
    )
