# /add-smoke-test — Scaffold a Smoke Test for a Server or Tool

You are adding a per-tool smoke test to the LocalCowork smoke test suite. Smoke tests are fast round-trip checks that validate a tool doesn't crash and returns the right shape.

## Arguments

```
/add-smoke-test <server-name> [tool-name]
```

- If only `server-name` is provided, scaffold smoke tests for ALL tools in that server.
- If `tool-name` is also provided, scaffold a smoke test for just that one tool.

## How Smoke Tests Differ from Unit Tests

| Aspect | Unit Test | Smoke Test |
|--------|-----------|------------|
| Speed | Can be thorough (seconds) | Must be fast (<500ms per tool) |
| Coverage | Edge cases, error paths | Happy path only |
| Dependencies | Mocked where possible | Can use real file system (temp dir) |
| Purpose | Verify correctness | Detect regressions |
| File name | `tool_name.test.ts` | `tool_name.smoke.test.ts` |
| Run frequency | During development | Before every push |

## Steps

### 1. Read the tool registry

Look up the server in `docs/mcp-tool-registry.yaml` and note each tool's:
- `params` (with required/optional/defaults)
- `returns` (expected output shape)
- `confirmation_required` and `undo_supported`

### 2. Create the smoke test file

Place it in the server's test directory:
- TypeScript: `mcp-servers/<server>/tests/<tool_name>.smoke.test.ts`
- Python: `mcp-servers/<server>/tests/<tool_name>_smoke_test.py`

### 3. Follow the template

#### TypeScript template:

```typescript
// mcp-servers/<server>/tests/<tool_name>.smoke.test.ts
import { describe, it, expect } from 'vitest';
import { <ToolName> } from '../src/tools/<tool_name>';

describe('<server>.<tool_name> (smoke)', () => {
  // Smoke 1: Tool can be instantiated / imported
  it('should export a valid tool definition', () => {
    expect(<ToolName>).toBeDefined();
    expect(<ToolName>.name).toBe('<tool_name>');
    expect(<ToolName>.schema).toBeDefined();
  });

  // Smoke 2: Params schema validates correct input
  it('should accept valid params', () => {
    const result = <ToolName>.schema.safeParse({
      // Provide MINIMAL valid params from the registry
    });
    expect(result.success).toBe(true);
  });

  // Smoke 3: Params schema rejects garbage input
  it('should reject invalid params', () => {
    const result = <ToolName>.schema.safeParse({
      nonexistent_field: 12345,
    });
    expect(result.success).toBe(false);
  });

  // Smoke 4: Metadata matches registry
  it('should have correct metadata', () => {
    expect(<ToolName>.metadata.confirmationRequired).toBe(<true|false>);
    expect(<ToolName>.metadata.undoSupported).toBe(<true|false>);
  });

  // Smoke 5: Basic execution (if tool is non-destructive)
  // Only include this for read-only / non-destructive tools.
  // For mutable tools, the param validation tests above are sufficient.
  it('should execute without crashing', async () => {
    const result = await <ToolName>.execute({
      // Minimal valid params (use temp dir if needed)
    });
    expect(result.success).toBeDefined();
  });
});
```

#### Python template:

```python
# mcp-servers/<server>/tests/<tool_name>_smoke_test.py
"""Smoke test for <server>.<tool_name>."""

import pytest
from pydantic import ValidationError

from src.tools.<tool_name> import <ToolClass>


class TestSmoke:
    """Fast regression checks — happy path only."""

    def test_tool_definition(self):
        """Tool can be instantiated with valid metadata."""
        tool = <ToolClass>()
        assert tool.name == "<tool_name>"
        assert tool.schema is not None

    def test_valid_params(self):
        """Params model accepts valid input."""
        params = <ToolClass>.Params(
            # Provide MINIMAL valid params from the registry
        )
        assert params is not None

    def test_invalid_params(self):
        """Params model rejects garbage input."""
        with pytest.raises(ValidationError):
            <ToolClass>.Params(nonexistent_field=12345)

    def test_metadata(self):
        """Confirmation and undo metadata match registry."""
        tool = <ToolClass>()
        assert tool.metadata.confirmation_required == <True|False>
        assert tool.metadata.undo_supported == <True|False>

    @pytest.mark.asyncio
    async def test_basic_execution(self, tmp_path):
        """Tool executes without crashing (read-only tools only)."""
        tool = <ToolClass>()
        result = await tool.execute(<ToolClass>.Params(
            # Minimal valid params using tmp_path if needed
        ))
        assert result.success is not None
```

### 4. Verify the test runs

```bash
# Single test
npx vitest run mcp-servers/<server>/tests/<tool_name>.smoke.test.ts
# or
pytest mcp-servers/<server>/tests/<tool_name>_smoke_test.py

# Full smoke suite
./scripts/smoke-test.sh --server <server>
```

### 5. Report what was created

Tell the user:
- How many smoke test files were created
- Which tools they cover
- How to run them (`./scripts/smoke-test.sh` or directly)

## Important Notes

- **Smoke tests must be FAST.** If a test takes more than 500ms, it's too heavy for a smoke test. Move the slow part to a unit test instead.
- **Don't duplicate unit test logic.** Smoke tests check "does it work at all?" — unit tests check "does it work correctly in all cases?"
- **The smoke test runner auto-discovers these files** by naming convention (`*.smoke.test.ts` / `*_smoke_test.py`). No registration needed.
- **For mutable/destructive tools**, skip the execution test (Smoke 5) and only test the schema + metadata. Actual execution is tested in unit tests with proper mocking.
