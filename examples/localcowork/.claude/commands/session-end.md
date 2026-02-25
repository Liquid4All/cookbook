# /session-end — Checkpoint Current Session

You are wrapping up a Claude Code development session on LocalCowork. Perform a clean checkpoint so the next session knows exactly where to pick up.

## Steps

### 1. Gather Session Facts

Determine what happened in this session:
- Which workstream(s) were worked on?
- Which files were created or modified? (Run `git status` and `git diff --stat`)
- Are tests passing? (Run the relevant test command for the workstream's language)
- Are there any new blockers or pending decisions?
- What is the logical next step?

### 2. Update PROGRESS.yaml

Read `PROGRESS.yaml`, then update it with these changes:

**Header fields:**
- `last_updated`: today's date (YYYY-MM-DD)
- `last_session_id`: use the format `session-NNN` where NNN increments from the last entry
- `current_phase`: update if the phase changed

**Workstream statuses:**
- Set any completed workstreams to `status: complete` with `completed_in: "session-NNN"` and a one-line `notes:` summary
- Set any partially-done workstreams to `status: in_progress` with a `notes:` explaining what's done and what remains
- If a workstream is blocked, set `status: blocked` and add an entry to the `blockers:` section

**Session log entry:**
Append to the `sessions:` list:
```yaml
  - id: "session-NNN"
    date: "YYYY-MM-DD"
    focus: "<phase> — <workstream IDs>"
    completed:
      - "<WS-ID>: <one-line summary>"
    artifacts_created:
      - "<path/to/file>"
    next_recommended: "<WS-ID> (<name>) — <brief reason>"
```

### 3. Doc-Sync Audit

Run the automated doc health check, then verify manually:

```bash
./scripts/doc-health.sh
```

The script checks three things automatically:
- **Cross-references** — do all file paths mentioned in docs actually exist?
- **Staleness** — did code directories change more recently than their corresponding docs?
- **Drift** — are there tools in code missing from the registry, or tools without tests?

Then verify these manually (the script can't check intent):
1. **ADR check:** If any file listed in CLAUDE.md's "Breaking Changes" section was modified, confirm an ADR was written.
2. **No duplication:** Did you copy content that should be a cross-reference instead? (See Rule 1 in `.claude/skills/feature-dev/SKILL.md`)

**Report the results** in the session summary under a `Doc-Sync` line. Example:
```
  Doc-Sync:    ✓ registry, ✓ tests, ✓ patterns, ✓ refs (no ADRs needed)
```

If anything is out of sync, either fix it now or log it as a blocker in PROGRESS.yaml with a clear description of what needs updating.

### 4. Commit the Checkpoint

```bash
# Stage progress file and any remaining work
git add PROGRESS.yaml
git add -A  # or specific files if you prefer

# Commit with conventional format
git commit -m "chore: checkpoint session-NNN — <1-line summary of what was accomplished>"
```

### 5. Print Session Summary

Output a summary for the user in this format:

```
═══════════════════════════════════════════════════
  Session NNN Complete
═══════════════════════════════════════════════════

  Phase:       <current phase>
  Workstreams: <WS-IDs worked on>
  Status:      <complete | in_progress | blocked>

  Completed:
    • <item 1>
    • <item 2>

  Files changed: <N files>
  Tests:         <pass/fail summary>
  Doc-Sync:      <✓/✗ registry, tests, patterns, ADRs>

  Next session should:
    → <WS-ID>: <description>

  Blockers: <none | list>
═══════════════════════════════════════════════════
```

## Important Notes

- **Never skip the PROGRESS.yaml update.** This is the single most important artifact for session continuity.
- **Be honest about status.** If something is 80% done, mark it `in_progress`, not `complete`.
- **The session log is append-only.** Never edit previous session entries — they're historical record.
- **If the user hasn't committed recent work,** stage and commit it before the checkpoint commit, or include it in the same commit.
