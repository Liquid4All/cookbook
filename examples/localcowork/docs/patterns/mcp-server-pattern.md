# MCP Server Implementation Pattern

> Canonical pattern for building MCP servers in LocalCowork.
> Every server follows this structure regardless of language.

## Directory Structure

### TypeScript Server
```
mcp-servers/<name>/
├── src/
│   ├── index.ts              # Entry point: register tools, start JSON-RPC listener
│   └── tools/
│       ├── tool_a.ts          # One file per tool
│       ├── tool_b.ts
│       └── ...
├── tests/
│   ├── tool_a.test.ts         # One test file per tool
│   ├── tool_b.test.ts
│   └── fixtures/              # Test data
├── package.json
└── tsconfig.json
```

### Python Server
```
mcp-servers/<name>/
├── src/
│   ├── __init__.py           # Entry point: register tools, start JSON-RPC listener
│   └── tools/
│       ├── tool_a.py
│       ├── tool_b.py
│       └── ...
├── tests/
│   ├── test_tool_a.py
│   ├── test_tool_b.py
│   └── fixtures/
└── pyproject.toml
```

## JSON-RPC over Stdio

All MCP servers communicate with the Agent Core via JSON-RPC 2.0 over stdio. The Tauri backend spawns each server as a child process and communicates by writing JSON to stdin and reading JSON from stdout.

### Handshake (initialize)

On startup, the server sends its tool manifest:

```json
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "result": {
    "serverInfo": {
      "name": "filesystem",
      "version": "1.0.0"
    },
    "tools": [
      {
        "name": "filesystem.list_dir",
        "description": "List contents of a directory",
        "inputSchema": {
          "type": "object",
          "properties": {
            "path": { "type": "string", "description": "Absolute path to directory" },
            "recursive": { "type": "boolean", "default": false }
          },
          "required": ["path"]
        },
        "metadata": {
          "confirmationRequired": false,
          "undoSupported": false
        }
      }
    ]
  }
}
```

### Tool Invocation

```json
// Request (stdin)
{
  "jsonrpc": "2.0",
  "id": "call-001",
  "method": "tools/call",
  "params": {
    "name": "filesystem.list_dir",
    "arguments": { "path": "/Users/chintan/Documents", "recursive": false }
  }
}

// Success Response (stdout)
{
  "jsonrpc": "2.0",
  "id": "call-001",
  "result": {
    "content": [
      { "type": "text", "text": "[{\"name\": \"report.pdf\", ...}]" }
    ]
  }
}

// Error Response (stdout)
{
  "jsonrpc": "2.0",
  "id": "call-001",
  "error": {
    "code": -32602,
    "message": "Directory not found: /Users/chintan/Documents"
  }
}
```

## TypeScript Tool Template

```typescript
// mcp-servers/filesystem/src/tools/list_dir.ts
import { z } from 'zod';
import { MCPTool, MCPResult, MCPError } from '../../_shared/ts/mcp-base';
import { Logger } from '@shared/logger';

// 1. Define typed params — must match docs/mcp-tool-registry.yaml
const paramsSchema = z.object({
  path: z.string().describe('Absolute path to directory'),
  recursive: z.boolean().optional().default(false),
  filter: z.string().optional(),
});

type Params = z.infer<typeof paramsSchema>;

// 2. Define the tool
export const listDir: MCPTool<Params> = {
  name: 'filesystem.list_dir',
  description: 'List contents of a directory',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  // 3. Implement
  async execute(params: Params): Promise<MCPResult> {
    const logger = Logger.child({ tool: 'filesystem.list_dir' });

    try {
      // Validate path is within sandbox
      assertSandboxed(params.path);

      const entries = await fs.readdir(params.path, { withFileTypes: true });

      const results = entries.map((entry) => ({
        name: entry.name,
        path: path.join(params.path, entry.name),
        type: entry.isDirectory() ? 'dir' : entry.isFile() ? 'file' : 'symlink',
        // ... additional metadata
      }));

      logger.info('Listed directory', { path: params.path, count: results.length });
      return { success: true, data: results };

    } catch (err) {
      logger.error('Failed to list directory', { path: params.path, error: err });
      throw new MCPError(-32602, `Failed to list directory: ${params.path}`);
    }
  },
};
```

## Python Tool Template

```python
# mcp-servers/ocr/src/tools/extract_text_from_image.py
from pydantic import BaseModel, Field
from typing import Optional
from ..mcp_base import MCPTool, MCPResult, MCPError
from _shared.services.logger import Logger

logger = Logger.child(tool="ocr.extract_text_from_image")

# 1. Define typed params — must match docs/mcp-tool-registry.yaml
class Params(BaseModel):
    path: str = Field(description="Path to image file")
    language: Optional[str] = Field(default="eng", description="OCR language")

# 2. Define typed result
class Result(BaseModel):
    text: str
    confidence: float

# 3. Define the tool
class ExtractTextFromImage(MCPTool[Params, Result]):
    name = "ocr.extract_text_from_image"
    description = "Extract text from an image file using OCR"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        try:
            # PaddleOCR for structured docs, Tesseract fallback
            result = self._run_paddleocr(params.path, params.language)
            if result.confidence < 0.5:
                result = self._run_tesseract(params.path, params.language)

            logger.info("Extracted text", path=params.path, confidence=result.confidence)
            return MCPResult(success=True, data=result)

        except FileNotFoundError:
            raise MCPError(-32602, f"Image not found: {params.path}")
        except Exception as e:
            logger.error("OCR failed", path=params.path, error=str(e))
            raise MCPError(-32603, f"OCR failed: {str(e)}")
```

## Server Entry Point Template

### TypeScript

```typescript
// mcp-servers/filesystem/src/index.ts
import { MCPServer } from '../../_shared/ts/mcp-base';
import { listDir } from './tools/list_dir';
import { readFile } from './tools/read_file';
import { writeFile } from './tools/write_file';
// ... import all tools

const server = new MCPServer({
  name: 'filesystem',
  version: '1.0.0',
  tools: [listDir, readFile, writeFile, moveFile, copyFile, deleteFile, searchFiles, getMetadata, watchFolder],
});

server.start(); // Starts JSON-RPC listener on stdio
```

### Python

```python
# mcp-servers/ocr/src/__init__.py
from .mcp_base import MCPServer
from .tools.extract_text_from_image import ExtractTextFromImage
from .tools.extract_text_from_pdf import ExtractTextFromPdf
from .tools.extract_structured_data import ExtractStructuredData
from .tools.extract_table import ExtractTable

server = MCPServer(
    name="ocr",
    version="1.0.0",
    tools=[ExtractTextFromImage(), ExtractTextFromPdf(), ExtractStructuredData(), ExtractTable()],
)

if __name__ == "__main__":
    server.start()  # Starts JSON-RPC listener on stdio
```

## Testing Pattern

Every tool must have tests covering:

1. **Parameter validation** — invalid types, missing required params, extra params
2. **Happy path** — correct input → expected output shape and values
3. **Error paths** — file not found, permission denied, invalid input
4. **Sandbox enforcement** — attempts to access paths outside sandbox are rejected
5. **Confirmation metadata** — verify `confirmationRequired` and `undoSupported` match the registry

```typescript
// mcp-servers/filesystem/tests/list_dir.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { listDir } from '../src/tools/list_dir';
import { setupTestDir, teardownTestDir } from './helpers';

describe('filesystem.list_dir', () => {
  let testDir: string;

  beforeAll(async () => {
    testDir = await setupTestDir({ files: ['a.txt', 'b.pdf', 'c.md'] });
  });

  afterAll(async () => {
    await teardownTestDir(testDir);
  });

  it('should list files in a directory', async () => {
    const result = await listDir.execute({ path: testDir });
    expect(result.success).toBe(true);
    expect(result.data).toHaveLength(3);
    expect(result.data[0]).toHaveProperty('name');
    expect(result.data[0]).toHaveProperty('path');
    expect(result.data[0]).toHaveProperty('type');
  });

  it('should support recursive listing', async () => {
    const result = await listDir.execute({ path: testDir, recursive: true });
    expect(result.success).toBe(true);
  });

  it('should return error for non-existent path', async () => {
    await expect(
      listDir.execute({ path: '/nonexistent/path' })
    ).rejects.toThrow();
  });

  it('should reject paths outside sandbox', async () => {
    await expect(
      listDir.execute({ path: '/etc/passwd' })
    ).rejects.toThrow('outside sandbox');
  });

  it('has correct metadata', () => {
    expect(listDir.confirmationRequired).toBe(false);
    expect(listDir.undoSupported).toBe(false);
  });
});
```

## Checklist Before Committing

- [ ] All tools match `docs/mcp-tool-registry.yaml` signatures
- [ ] Every tool has a unit test
- [ ] `confirmationRequired` and `undoSupported` match the registry
- [ ] No `any` types (TS) or `Any` types (Python) without justification
- [ ] Lint passes: `tsc --noEmit` / `mypy --strict`
- [ ] All uses of Logger, no print() or console.log()
- [ ] Error handling uses structured MCPError, not raw exceptions
- [ ] Run `/validate-server <name>` and get all green
