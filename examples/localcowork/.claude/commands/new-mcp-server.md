# /new-mcp-server

Scaffold a new MCP server from the canonical template.

**Usage:** `/new-mcp-server <server-name> <language: ts|py>`

## Steps

1. Read `docs/mcp-tool-registry.yaml` and extract the tool definitions for the `$ARGUMENTS` server.
2. Read `docs/patterns/mcp-server-pattern.md` for the canonical implementation pattern.
3. Read the `.claude/skills/mcp-server-dev/SKILL.md` skill for detailed guidance.

### If language is `ts` (TypeScript):

4. Create directory structure under `mcp-servers/<server-name>/`:
   ```
   mcp-servers/<server-name>/
   ├── src/
   │   ├── index.ts          # Server entry: registers tools, starts JSON-RPC listener
   │   └── tools/
   │       └── <tool_name>.ts  # One file per tool from the registry
   ├── tests/
   │   └── <tool_name>.test.ts  # One test file per tool
   ├── package.json
   └── tsconfig.json
   ```
5. Copy shared base classes from `mcp-servers/_shared/ts/`.
6. Generate `package.json` with dependencies (zod, typescript, vitest, the shared base).
7. Generate `tsconfig.json` with `"strict": true`.
8. For each tool in the registry: create `src/tools/<tool_name>.ts` with:
   - Typed params (zod schema matching the registry exactly)
   - Typed returns (matching the registry exactly)
   - `confirmationRequired` and `undoSupported` metadata from registry
   - Stub implementation with `// TODO: implement` and the logic described in the PRD
9. Create `src/index.ts` that imports and registers all tools.
10. For each tool: create `tests/<tool_name>.test.ts` with test stubs for:
    - Parameter validation (valid params, missing required, invalid types)
    - Happy path (expected output shape)
    - Error paths (file not found, permission denied, etc.)

### If language is `py` (Python):

4. Create directory structure under `mcp-servers/<server-name>/`:
   ```
   mcp-servers/<server-name>/
   ├── src/
   │   ├── __init__.py       # Server entry: registers tools, starts JSON-RPC listener
   │   └── tools/
   │       └── <tool_name>.py  # One file per tool from the registry
   ├── tests/
   │   └── test_<tool_name>.py  # One test file per tool
   └── pyproject.toml
   ```
5. Copy shared base classes from `mcp-servers/_shared/py/`.
6. Generate `pyproject.toml` with dependencies (pydantic, pytest, the shared base).
7. For each tool in the registry: create `src/tools/<tool_name>.py` with:
   - Typed params (pydantic BaseModel matching the registry exactly)
   - Typed returns (pydantic BaseModel matching the registry exactly)
   - `confirmation_required` and `undo_supported` metadata from registry
   - Stub implementation with `# TODO: implement` and the logic described in the PRD
8. Create `src/__init__.py` that imports and registers all tools.
9. For each tool: create `tests/test_<tool_name>.py` with test stubs.

## Post-Scaffold

10. Run initial lint to verify the scaffolded code passes type checking:
    - TS: `npx tsc --noEmit`
    - PY: `mypy --strict src/`
11. Confirm to the user: "Scaffolded `<server-name>` MCP server with N tools. Run `/validate-server <server-name>` to verify against the PRD spec."
