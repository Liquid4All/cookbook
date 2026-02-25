# Feature Development Skill — Doc-Sync & Maintenance

> MANDATORY TRIGGERS: "add feature", "new feature", "implement", "build", "new tool", "add tool",
> "new endpoint", "add server", "modify tool", "change tool signature", "update tool", "new use case".
> Also trigger whenever a session involves creating or modifying source files that affect user-facing
> behavior, tool contracts, or architecture.

## Purpose

This skill prevents documentation from going stale. It covers two concerns:

1. **Reactive sync** — when you change code, update the matching docs (the matrix below).
2. **Proactive maintenance** — principles that prevent drift, duplication, and rot over time.

Claude Code reads this **before writing any feature code** and follows the checklists **after the code is written**.

---

## Part 1: Doc Maintenance Principles

These five rules apply to every doc in the project. Follow them whenever you create or edit documentation.

### Rule 1: Single Source of Truth

Every fact lives in exactly one place. Other docs reference it — they never copy it.

| Fact | Source of Truth | How Others Reference It |
|---|---|---|
| Tool definitions (params, returns, metadata) | `docs/mcp-tool-registry.yaml` | "See registry for full schema" |
| Product requirements and use cases | `docs/PRD.md` | "Per UC-3 in the PRD" |
| Architecture decisions | `docs/architecture-decisions/NNN-*.md` | "See ADR-003" |
| Workstream status and progress | `PROGRESS.yaml` | "Check PROGRESS.yaml for current status" |
| Build commands and project structure | `CLAUDE.md` | "See CLAUDE.md Commands section" |
| Pattern implementations | `docs/patterns/*.md` | "See the HITL pattern doc" |

**When you're tempted to copy content into a new doc, write a cross-reference instead.**
If the same information appears in two places, it will eventually disagree.

### Rule 2: Docs Have Owners

Every doc is owned by a workstream. When that workstream's code changes, the owner is responsible for updating the doc.

| Doc | Owner |
|---|---|
| `docs/PRD.md` | Product (not changed by code sessions — treat as read-only unless an ADR supersedes a section) |
| `docs/mcp-tool-registry.yaml` | Whichever workstream adds/modifies tools |
| `docs/patterns/*.md` | WS-2 (Agent Core) for HITL and context; WS-0 (Foundation) for server pattern and error handling |
| `docs/architecture-decisions/` | Whoever makes the decision — each ADR lists its author |
| `docs/contributing.md` | Infrastructure / WS-0 |
| `CLAUDE.md` | Infrastructure / WS-0 (update whenever commands, skills, paths, or process changes) |
| `PROGRESS.yaml` | Every session (via `/session-end`) |
| `README.md` | Infrastructure — update when user-facing behavior changes |
| `mcp-servers/<server>/README.md` | The workstream building that server |

### Rule 3: Cross-References Must Resolve

Every internal reference (file path, doc name, section heading) must point to something that exists.

When you rename, move, or delete a file, search for references to it:
```bash
# Check what references a file before removing or renaming it
grep -r "old_filename" docs/ CLAUDE.md README.md .claude/ PROGRESS.yaml
```

When you rename a section heading in a doc, check if any other doc links to that heading by name.

**The `scripts/doc-health.sh` script automates this check.** Run it periodically or as part of `/session-end`.

### Rule 4: Code Examples Must Be Testable

If a doc contains a code example (TypeScript, Python, Rust, SQL, bash), it should be possible to verify the example still works. Practices:

- **Prefer referencing real files** over pasting code inline. Instead of a 30-line TypeScript snippet, write "See `mcp-servers/filesystem/src/tools/list_dir.ts` for the canonical implementation."
- **If you must inline code**, keep it minimal (under 15 lines) and add a comment noting which source file it mirrors: `// Mirrors: mcp-servers/filesystem/src/tools/list_dir.ts`
- **SQL schemas** in docs should match the actual migration files or init scripts in the codebase. If you change a schema in code, grep for it in docs.
- **Config examples** (YAML, JSON) should match the actual config files. Don't invent example configs that differ from reality.

### Rule 5: Deprecation Protocol

When a doc (or a section of a doc) becomes irrelevant:

1. **Don't silently delete it.** Other docs or skills may reference it.
2. **Mark it deprecated** with a note at the top: `> DEPRECATED: This section was superseded by ADR-NNN on YYYY-MM-DD. See [new location].`
3. **Update all cross-references** to point to the replacement.
4. **Remove the deprecated content** in a subsequent session after confirming nothing links to it.

For entire files: move to a `docs/_archive/` directory (create it if needed) rather than deleting.

---

## Part 2: Pre-Flight — Identify Change Type

Before coding, classify the change. This determines which docs need updating.

| Change Type | Example |
|---|---|
| **New MCP tool** | Adding `filesystem.rename_file` |
| **Modified tool signature** | Changing params/returns on an existing tool |
| **New MCP server** | Adding a 14th server (e.g., `browser`) |
| **Agent Core change** | Modifying ToolRouter, ConversationManager, ContextWindowManager |
| **Frontend component** | New UI component or changing confirmation flow |
| **New use case** | Adding UC-11 |
| **Shared service change** | Modifying model-gateway, state-manager, logger, config-loader |
| **Infrastructure** | Build system, CI, dev tooling, scripts |

---

## Part 3: Doc-Sync Matrix

After the code is written and tests pass, update every file marked for your change type.

### New MCP Tool

| File | What to Update |
|---|---|
| `docs/mcp-tool-registry.yaml` | Add tool definition (params, returns, confirmation, undo metadata) |
| `mcp-servers/<server>/README.md` | Add tool to the server's tool list |
| `PROGRESS.yaml` | Update workstream status, add to `tools:` list if applicable |
| Smoke test file | Create `<tool_name>.smoke.test.ts` or `<tool_name>_smoke_test.py` |
| Unit test file | Create `<tool_name>.test.ts` or `test_<tool_name>.py` |

### Modified Tool Signature

| File | What to Update |
|---|---|
| `docs/mcp-tool-registry.yaml` | Update params, returns, or metadata to match new signature |
| `docs/architecture-decisions/` | **Write an ADR** if this is a breaking change (see CLAUDE.md Breaking Changes) |
| Existing tests | Update to match new contract |
| Smoke test | Update if params changed |

### New MCP Server

| File | What to Update |
|---|---|
| `docs/mcp-tool-registry.yaml` | Add all tools for the new server |
| `docs/PRD.md` | Update if the server adds new capabilities described in the PRD |
| `CLAUDE.md` | Add server to the Key Paths section if not already listed |
| `docs/contributing.md` | Update if new language or pattern is introduced |
| `PROGRESS.yaml` | Update workstream status |
| `.claude/skills/mcp-server-dev/SKILL.md` | Update if new patterns are established |

### Agent Core Change

| File | What to Update |
|---|---|
| `docs/patterns/context-window-management.md` | If context budget or eviction changed |
| `docs/patterns/human-in-the-loop.md` | If confirmation flow changed |
| `docs/patterns/error-handling.md` | If error handling strategy changed |
| `docs/architecture-decisions/` | **Write an ADR** for any change listed in CLAUDE.md Breaking Changes |
| `PROGRESS.yaml` | Update workstream status |

### Frontend Component

| File | What to Update |
|---|---|
| `docs/patterns/human-in-the-loop.md` | If confirmation UI changed |
| `docs/PRD.md` | If new UI section is described in PRD |
| `PROGRESS.yaml` | Update workstream status |

### New Use Case

| File | What to Update |
|---|---|
| `docs/PRD.md` | Add UC definition if not already there |
| `tests/integration/` | Create `uc<N>_*.test.ts` |
| `tests/fixtures/` | Add sample data for the UC |
| `.claude/skills/tool-chain-test/SKILL.md` | Add UC fixture layout and expected chain |
| `PROGRESS.yaml` | Update workstream status |

### Shared Service Change

| File | What to Update |
|---|---|
| `docs/architecture-decisions/` | **Write an ADR** (mandatory — shared services are a breaking change surface) |
| `CLAUDE.md` | Update Shared Services Contract table if interface changed |
| `PROGRESS.yaml` | Update workstream status |

### Infrastructure

| File | What to Update |
|---|---|
| `CLAUDE.md` | Update Commands section if new scripts or commands added |
| `docs/contributing.md` | Update if dev workflow changed |
| `PROGRESS.yaml` | Update workstream status |
| `scripts/setup-dev.sh` | Update if new dev dependencies introduced |

---

## Part 4: Post-Flight Checks

After committing, verify:

1. **Registry sync** — Does `docs/mcp-tool-registry.yaml` match every implemented tool?
2. **Test coverage** — Does every new tool have a unit test AND a smoke test?
3. **ADR written** — Did any breaking change surface get modified without an ADR?
4. **PROGRESS.yaml** — Is the workstream status accurate?
5. **Cross-references** — Did you rename/move/delete any file? If so, did you update all references?
6. **No duplication** — Did you copy content that should be a cross-reference instead?
7. **README/contributing** — Would a new contributor understand the change?

If any answer is "no", fix it before moving on.

Run `./scripts/doc-health.sh` for an automated check of cross-references and staleness.

---

## Quick Reference: Always Update These

No matter what type of change:

- `PROGRESS.yaml` — workstream status
- The relevant test file(s) — unit + smoke
- `docs/mcp-tool-registry.yaml` — if any tool was added or changed

And always ask: "If I read these docs cold next week, would they be accurate?"
