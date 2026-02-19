"""
Meeting MCP Server — Entry Point

Registers all meeting tools and starts the JSON-RPC listener.
This server provides audio transcription (WS-5A) and transcript
extraction/minutes generation (WS-5B).

Tools (4):
  meeting.transcribe_audio       — transcribe audio to text with optional diarization
  meeting.extract_action_items   — extract action items from transcript
  meeting.extract_commitments    — extract commitments, decisions, open questions
  meeting.generate_minutes       — generate formatted meeting minutes (confirm)
"""

from __future__ import annotations

import os
import sys

# Add shared path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared", "py"))

# Add src directory so tool modules can import transcription / meeting_types / extraction
sys.path.insert(0, os.path.dirname(__file__))

from mcp_base import MCPServer  # noqa: E402

from tools.transcribe_audio import TranscribeAudio  # noqa: E402
from tools.extract_action_items import ExtractActionItems  # noqa: E402
from tools.extract_commitments import ExtractCommitments  # noqa: E402
from tools.generate_minutes import GenerateMinutes  # noqa: E402

# ─── Server Setup ────────────────────────────────────────────────────────────

server = MCPServer(
    name="meeting",
    version="0.1.0",
    tools=[
        TranscribeAudio(),
        ExtractActionItems(),
        ExtractCommitments(),
        GenerateMinutes(),
    ],
)

if __name__ == "__main__":
    server.start()
