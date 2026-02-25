# ADR-002: MCP Server Language Selection

## Status
Accepted

## Context
LocalCowork has 13 MCP servers. Each server is an independent process communicating via JSON-RPC over stdio. They can be written in any language. We need a principled approach to language selection.

## Decision
Use TypeScript for I/O-centric servers (8 servers) and Python for ML/data-processing servers (5 servers).

## Language Assignments

### TypeScript (8 servers)
- **filesystem** — Node.js `fs` is excellent; Tauri bridge for sandboxing
- **calendar** — ical.js for .ics parsing; clean TypeScript APIs
- **email** — MBOX parsing, SMTP — well-served by Node.js
- **task** — Pure SQLite CRUD; lightweight
- **data** — papaparse for CSV, better-sqlite3 for SQLite
- **audit** — SQLite reads + PDF generation
- **clipboard** — Thin Tauri bridge wrapper; must be TypeScript for IPC
- **system** — Thin Tauri bridge wrapper; must be TypeScript for IPC

### Python (5 servers)
- **document** — python-docx, PyPDF2, LibreOffice bindings, Pandoc wrappers
- **ocr** — PaddleOCR (Python-native), Tesseract (pytesseract)
- **knowledge** — sentence-transformers, SQLite-vec, chunking — Python ML ecosystem
- **meeting** — pywhispercpp, pyannote-audio — Python ML ecosystem
- **security** — Regex patterns + model classification; Python is natural for both

## Rationale
The deciding factor is **dependency availability**. ML libraries (PaddleOCR, Whisper, sentence-transformers, pyannote-audio) only have mature Python bindings. I/O operations and database CRUD are well-served by TypeScript with strong typing.

The shared base classes (`mcp-servers/_shared/ts/` and `mcp-servers/_shared/py/`) ensure consistency across languages. Both bases implement the same JSON-RPC protocol, tool registration, and error handling patterns.

## Trade-offs
- Two language runtimes means two sets of dependencies, two test frameworks, two linting configs.
- TypeScript and Python have different async models (Promises vs asyncio), requiring care in the shared base classes.
- Developers need to be comfortable in both languages.

## Alternatives Considered
- **All TypeScript:** Would require Node.js bindings for PaddleOCR, Whisper, etc. — these don't exist at quality.
- **All Python:** Would work but TypeScript offers better typing for the simpler servers, and the Tauri bridge servers must be TypeScript for IPC.
- **All Rust:** Maximum performance but dramatically higher development cost for the ML servers.
