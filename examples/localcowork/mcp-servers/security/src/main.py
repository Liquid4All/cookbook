"""
Security MCP Server — Entry Point

Registers all security tools and starts the JSON-RPC listener.
This server provides PII/secrets scanning, duplicate detection,
cleanup proposals, and file encryption/decryption.

Tools (6):
  security.scan_for_pii     — scan files for PII (SSN, credit cards, etc.)
  security.scan_for_secrets  — scan files for exposed secrets
  security.find_duplicates   — find duplicate files by hash/name/content
  security.propose_cleanup   — generate cleanup proposals for findings
  security.encrypt_file      — encrypt a file with Fernet (confirm)
  security.decrypt_file      — decrypt an encrypted file (confirm)
"""

from __future__ import annotations

import os
import sys

# Add shared path and own package root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared", "py"))
sys.path.insert(0, os.path.dirname(__file__))

from mcp_base import MCPServer  # noqa: E402
from validation import init_sandbox  # noqa: E402

from tools.scan_for_pii import ScanForPii  # noqa: E402
from tools.scan_for_secrets import ScanForSecrets  # noqa: E402
from tools.find_duplicates import FindDuplicates  # noqa: E402
from tools.propose_cleanup import ProposeCleanup  # noqa: E402
from tools.encrypt_file import EncryptFile  # noqa: E402
from tools.decrypt_file import DecryptFile  # noqa: E402

# ─── Sandbox Initialization ─────────────────────────────────────────────────

allowed_paths_str = os.environ.get("LOCALCOWORK_ALLOWED_PATHS", os.path.expanduser("~"))
allowed_paths = allowed_paths_str.split(os.pathsep)
init_sandbox(allowed_paths)

# ─── Server Setup ───────────────────────────────────────────────────────────

server = MCPServer(
    name="security",
    version="1.0.0",
    tools=[
        ScanForPii(),
        ScanForSecrets(),
        FindDuplicates(),
        ProposeCleanup(),
        EncryptFile(),
        DecryptFile(),
    ],
)

if __name__ == "__main__":
    server.start()
