# /validate-server

Validate an MCP server implementation against the PRD tool registry.

**Usage:** `/validate-server <server-name>`

## Steps

1. Read `docs/mcp-tool-registry.yaml` for the expected tool definitions of the `$ARGUMENTS` server.
2. Read the server's source code in `mcp-servers/$ARGUMENTS/`.
3. For each tool defined in the registry for this server:

### Existence Check
   a. Verify a tool implementation file exists (e.g., `src/tools/<tool_name>.ts` or `.py`).
   b. If missing, flag as ❌ **Missing tool**.

### Signature Check
   c. Verify parameter names match the registry (exact names, same required/optional).
   d. Verify parameter types match (string, number, boolean, object, array).
   e. Verify return type structure matches the registry.
   f. If mismatched, flag as ⚠️ **Signature mismatch** with details.

### Metadata Check
   g. Verify `confirmation_required` matches the registry value.
   h. Verify `undo_supported` matches the registry value.
   i. If mismatched, flag as ⚠️ **Metadata mismatch**.

### Test Coverage Check
   j. Verify a test file exists for this tool in `tests/`.
   k. If missing, flag as 📋 **Missing test**.

### Implementation Check
   l. Check if the tool implementation contains `TODO` markers (still a stub).
   m. If it does, flag as 🔨 **Stub only** (not yet implemented).

## Report Format

```
=== Validation Report: <server-name> ===

Tools in registry: N
Tools implemented: M / N

✅ list_dir          — signature OK, tests OK, implemented
✅ read_file         — signature OK, tests OK, implemented
⚠️  write_file       — param 'encoding' missing (optional in spec)
❌ watch_folder      — not implemented
📋 search_files      — implemented but no tests
🔨 get_metadata      — stub only (contains TODO)

Summary: M/N tools complete, K warnings, J missing
```

## Exit

If all tools pass, confirm: "✅ `<server-name>` server validates against the PRD spec."
If there are issues, list them and suggest: "Run `/new-mcp-server <server-name> <lang>` to regenerate stubs for missing tools."
