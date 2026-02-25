"""
MCP Server Base Classes — Python

Shared foundation for all Python MCP servers in LocalCowork.
Implements JSON-RPC 2.0 over stdio transport and tool registration.

Usage:
    from mcp_base import MCPServer, MCPTool, MCPResult, MCPError
"""

from __future__ import annotations

import asyncio
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

# ─── Type Variables ──────────────────────────────────────────────────────────

TParams = TypeVar("TParams", bound=BaseModel)
TResult = TypeVar("TResult", bound=BaseModel)

# ─── Result & Error Types ────────────────────────────────────────────────────


@dataclass
class MCPResult(Generic[TResult]):
    """Result returned by a tool execution."""

    success: bool
    data: TResult | None = None


class MCPError(Exception):
    """Structured error for MCP tool failures."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class ErrorCodes:
    """Standard MCP error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # Custom codes for LocalCowork
    SANDBOX_VIOLATION = -32001
    CONFIRMATION_REQUIRED = -32002
    FILE_NOT_FOUND = -32003
    PERMISSION_DENIED = -32004


# ─── Tool Base Class ─────────────────────────────────────────────────────────


class MCPTool(ABC, Generic[TParams, TResult]):
    """
    Abstract base class for MCP tools.

    Every tool must define:
    - name: fully qualified (e.g., 'ocr.extract_text_from_image')
    - description: for the LLM
    - Params type: pydantic BaseModel for input validation
    - Result type: pydantic BaseModel for output structure
    - confirmation_required: whether Agent Core needs user confirmation
    - undo_supported: whether this action can be reversed
    - execute(): the implementation
    """

    name: str = ""
    description: str = ""
    confirmation_required: bool = False
    undo_supported: bool = False

    @abstractmethod
    async def execute(self, params: TParams) -> MCPResult[TResult]:
        """Execute the tool with validated parameters."""
        ...

    def get_params_model(self) -> type[BaseModel]:
        """Get the Pydantic model class for params validation."""
        # Extract from Generic type args
        for base in type(self).__orig_bases__:  # type: ignore[attr-defined]
            if hasattr(base, "__args__") and len(base.__args__) >= 1:
                return base.__args__[0]
        raise TypeError(f"Tool {self.name} must specify Generic params type")

    def get_input_schema(self) -> dict[str, Any]:
        """Generate JSON Schema from the Pydantic params model."""
        model = self.get_params_model()
        return model.model_json_schema()

    def to_definition(self) -> dict[str, Any]:
        """Generate the MCP tool definition for the Rust MCP client."""
        return {
            "name": self.name,
            "description": self.description,
            "params_schema": self.get_input_schema(),
            "confirmation_required": self.confirmation_required,
            "undo_supported": self.undo_supported,
        }


# ─── MCP Server ──────────────────────────────────────────────────────────────


class MCPServer:
    """
    Base MCP Server.

    Registers tools, handles JSON-RPC over stdio, validates params,
    and dispatches tool calls.

    Usage:
        server = MCPServer(
            name="ocr",
            version="1.0.0",
            tools=[ExtractTextFromImage(), ExtractTextFromPdf()],
        )
        server.start()
    """

    def __init__(self, name: str, version: str, tools: list[MCPTool[Any, Any]]) -> None:
        self.name = name
        self.version = version
        self.tools: dict[str, MCPTool[Any, Any]] = {}

        for tool in tools:
            self.tools[tool.name] = tool

    def start(self) -> None:
        """Start the JSON-RPC listener on stdio (blocking)."""
        asyncio.run(self._run())

    async def _run(self) -> None:
        """Main event loop: read stdin, dispatch, write stdout."""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line = await reader.readline()
            if not line:
                break  # stdin closed

            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue

            try:
                request = json.loads(line_str)
                response = await self._handle_request(request)
                if response:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError:
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": 0,
                    "error": {"code": ErrorCodes.PARSE_ERROR, "message": "Invalid JSON"},
                }
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()

    def _build_init_result(self) -> dict[str, Any]:
        """Build the initialization result payload (tool manifest)."""
        tool_defs = [tool.to_definition() for tool in self.tools.values()]
        return {
            "server_info": {"name": self.name, "version": self.version},
            "tools": tool_defs,
            "capabilities": {},
        }

    async def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch a JSON-RPC request."""
        method = request.get("method", "")
        request_id = request.get("id", 0)

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": self._build_init_result(),
            }

        if method == "tools/call":
            return await self._handle_tool_call(request)

        if method == "tools/list":
            return self._handle_tool_list(request)

        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"status": "ok"}}

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": ErrorCodes.METHOD_NOT_FOUND,
                "message": f"Unknown method: {method}",
            },
        }

    async def _handle_tool_call(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a tools/call request."""
        request_id = request.get("id", 0)
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        tool = self.tools.get(tool_name)
        if not tool:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": ErrorCodes.METHOD_NOT_FOUND,
                    "message": f"Unknown tool: {tool_name}",
                },
            }

        # Validate params
        try:
            params_model = tool.get_params_model()
            validated_params = params_model(**arguments)
        except ValidationError as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": ErrorCodes.INVALID_PARAMS,
                    "message": f"Invalid parameters: {e}",
                },
            }

        # Execute tool
        try:
            result = await tool.execute(validated_params)
            result_data = result.data
            if isinstance(result_data, BaseModel):
                result_data = result_data.model_dump()

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result_data)}],
                },
            }
        except MCPError as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": e.code, "message": str(e)},
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": ErrorCodes.INTERNAL_ERROR,
                    "message": f"Internal error: {e!s}",
                },
            }

    def _handle_tool_list(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a tools/list request."""
        tool_defs = [tool.to_definition() for tool in self.tools.values()]
        return {
            "jsonrpc": "2.0",
            "id": request.get("id", 0),
            "result": {"tools": tool_defs},
        }
