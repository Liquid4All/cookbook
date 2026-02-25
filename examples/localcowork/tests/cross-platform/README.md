# Cross-Platform Tests

Validates that LocalCowork builds, installs, and runs correctly on macOS, Windows, and Linux.

## Running Tests

```bash
# Run all cross-platform tests
npx vitest run tests/cross-platform/

# Run individual suites
npx vitest run tests/cross-platform/smoke-runner.test.ts
npx vitest run tests/cross-platform/smoke-runner-fs.test.ts
npx vitest run tests/cross-platform/build-check.test.ts
npx vitest run tests/cross-platform/benchmarks.test.ts
```

## Test Suites

### smoke-runner.test.ts (13 test groups)

| # | Test Group | What it validates |
|---|-----------|-------------------|
| 1 | Platform Detection | OS, arch, hostname, memory detection |
| 2 | Filesystem Paths | Forward/backslash normalization, path.join, path.resolve |
| 3 | Temp Directory | Temp dir existence, subdirectory creation/removal |
| 4 | File Permissions | Read/write, stat, Unix file mode (skipped on Windows) |
| 5 | Process Listing | NodeSystemBridge.listProcesses() returns valid data |
| 6 | System Info | NodeSystemBridge.getSystemInfo() returns correct platform |
| 7 | Clipboard | MockClipboardBridge read/write, history tracking |

### smoke-runner-fs.test.ts (tests 8-13: filesystem edge cases)

| # | Test Group | What it validates |
|---|-----------|-------------------|
| 8 | File Watcher | fs.watch API availability, directory change detection |
| 9 | Unicode Paths | Files with accented, CJK, and emoji characters in names |
| 10 | Long Paths | Moderately long paths, near-component-limit filenames |
| 11 | Hidden Files | Dotfile creation (Unix) / attrib (Windows), detection |
| 12 | Line Endings | LF/CRLF preservation, line ending detection |
| 13 | Case Sensitivity | Filesystem case sensitivity probe, behavior verification |

### build-check.test.ts (6 test groups)

| # | Test Group | What it validates |
|---|-----------|-------------------|
| 1 | TypeScript Compile | tsconfig.json valid, tsc available |
| 2 | Vite Build | vite available, config exists |
| 3 | Cargo Check | rustc available (optional), Cargo.toml exists |
| 4 | Server Health | All TS MCP servers have index.ts entry points |
| 5 | Dependency Check | node_modules, package-lock.json, key packages installed |
| 6 | Python Check | python3 >= 3.11 available |

### benchmarks.test.ts (5 test groups)

| # | Test Group | Threshold |
|---|-----------|-----------|
| 1 | File List Speed | list_dir on 100 files < 500ms |
| 2 | File Read Speed | read_file on 1MB file < 200ms |
| 3 | CSV Write Speed | write_csv with 1000 rows < 500ms |
| 4 | SQLite Query Speed | query_sqlite with 1000 rows < 500ms |
| 5 | Search Speed | search_files across 50 files < 1000ms |

## Platform-Specific Notes

- **macOS**: APFS is case-insensitive by default. FSEvents powers fs.watch.
- **Windows**: 260-char MAX_PATH limit applies unless long path support is enabled. Hidden files use the attrib command. Line endings default to CRLF.
- **Linux**: Case-sensitive ext4/btrfs. inotify powers fs.watch. No hidden file attribute (dotfile convention only).

## Helpers: platform-helpers.ts

Reusable utility functions for writing new cross-platform tests:

- `getPlatformTempDir()` -- Platform temp directory
- `normalizePath(p)` -- Normalize path separators
- `getExpectedLineEnding()` -- Platform line ending
- `isCaseSensitiveFS(dir?)` -- Probe case sensitivity
- `getMaxPathLength()` -- OS max path length
- `createHiddenFile(dir, name, content)` -- Create hidden file (cross-platform)
- `isHiddenFile(path)` -- Check if file is hidden
- `createTestTempDir(prefix)` -- Create isolated temp dir for tests
- `cleanupTestDir(path)` -- Remove test temp dir

## Adding New Tests

1. Add test cases to the appropriate file (smoke-runner, build-check, or benchmarks).
2. Use `it.skipIf(platform === 'win32')` for platform-specific skips.
3. Use helpers from `platform-helpers.ts` for temp dirs and cleanup.
4. Keep each test self-contained: create fixtures in beforeAll, clean up in afterAll.

## CI Template

See `ci-template.yml` for a GitHub Actions workflow that runs these tests on macOS, Windows, and Linux. Copy to `.github/workflows/cross-platform.yml` to activate.
