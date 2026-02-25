#!/usr/bin/env python3
"""
Python tool wrapper for calling MCP Python tools from TypeScript tests.

Usage: python3 call-python-tool.py <server_name> <tool_name> <json_params>

Prints JSON result to stdout.
"""

import asyncio
import json
import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHARED_PY = os.path.join(PROJECT_ROOT, "mcp-servers", "_shared", "py")
sys.path.insert(0, SHARED_PY)

from validation import init_sandbox  # noqa: E402

# Initialize sandbox for temp dirs
init_sandbox([tempfile.gettempdir(), "/private/var/folders", "/private/tmp", "/tmp"])

SERVER_PATHS = {
    "security": "mcp-servers/security/src",
    "document": "mcp-servers/document/src",
    "meeting": "mcp-servers/meeting/src",
    "ocr": "mcp-servers/ocr/src",
    "knowledge": "mcp-servers/knowledge/src",
}

TOOL_CLASS_MAP = {
    "scan_for_pii": "ScanForPii",
    "scan_for_secrets": "ScanForSecrets",
    "find_duplicates": "FindDuplicates",
    "propose_cleanup": "ProposeCleanup",
    "extract_text": "ExtractText",
    "diff_documents": "DiffDocuments",
    "transcribe_audio": "TranscribeAudio",
    "extract_action_items": "ExtractActionItems",
    "extract_commitments": "ExtractCommitments",
}


async def main() -> None:
    if len(sys.argv) != 4:
        print(json.dumps({"error": "Usage: <server> <tool> <json_params>"}))
        sys.exit(1)

    server_name = sys.argv[1]
    tool_name = sys.argv[2]
    params_json = sys.argv[3]

    server_path = SERVER_PATHS.get(server_name)
    if not server_path:
        print(json.dumps({"error": f"Unknown server: {server_name}"}))
        sys.exit(1)

    full_path = os.path.join(PROJECT_ROOT, server_path)
    tools_path = os.path.join(full_path, "tools")
    sys.path.insert(0, full_path)
    sys.path.insert(0, tools_path)

    class_name = TOOL_CLASS_MAP.get(tool_name)
    if not class_name:
        print(json.dumps({"error": f"Unknown tool: {tool_name}"}))
        sys.exit(1)

    try:
        module = __import__(tool_name)
        tool_class = getattr(module, class_name)
        tool_instance = tool_class()
        params_model = tool_instance.get_params_model()
        params = params_model(**json.loads(params_json))
        result = await tool_instance.execute(params)
        data = result.data
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        print(json.dumps({"success": result.success, "data": data}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())