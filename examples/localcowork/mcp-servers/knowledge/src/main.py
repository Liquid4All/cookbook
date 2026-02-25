"""
Knowledge MCP Server — Entry Point

Registers all knowledge tools and starts the JSON-RPC listener.
This server provides a local RAG pipeline: indexing, semantic search,
question answering over indexed documents, and chunk retrieval.

Tools (5):
  knowledge.index_folder       — index a folder of documents
  knowledge.search_documents   — semantic search across indexed documents
  knowledge.ask_about_files    — question answering grounded in documents
  knowledge.update_index       — update index for changed/new files
  knowledge.get_related_chunks — find chunks related to a passage
"""

from __future__ import annotations

import os
import sys

# Add shared path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared", "py"))

# Add src directory so tool modules can import db / embeddings
sys.path.insert(0, os.path.dirname(__file__))

from mcp_base import MCPServer  # noqa: E402

from tools.index_folder import IndexFolder  # noqa: E402
from tools.search_documents import SearchDocuments  # noqa: E402
from tools.ask_about_files import AskAboutFiles  # noqa: E402
from tools.update_index import UpdateIndex  # noqa: E402
from tools.get_related_chunks import GetRelatedChunks  # noqa: E402

# ─── Server Setup ───────────────────────────────────────────────────────────

server = MCPServer(
    name="knowledge",
    version="1.0.0",
    tools=[
        IndexFolder(),
        SearchDocuments(),
        AskAboutFiles(),
        UpdateIndex(),
        GetRelatedChunks(),
    ],
)

if __name__ == "__main__":
    server.start()
