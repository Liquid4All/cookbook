# LFM2-24B-A2B Demo: 21 Tools That Work Every Time

## Why 21 Tools?

The full LocalCowork agent has 67 tools across 13 servers. LFM2-24B-A2B scored 80%
single-step accuracy across all 67 — impressive for a local model. But for a live demo,
80% means 1-in-5 wrong picks. That's fine in a human-in-the-loop agent (correction is
sub-second), but it's not "works like a charm, every time."

This curated set picks the **21 tools from 6 servers that score 80%+ individually and
participate in passing multi-step chains**. These are the tools where LFM2 consistently
dispatches correctly on the first try.

## Tool Set (21 tools, 6 servers)

| Server | Tools | Count | Accuracy |
|--------|-------|-------|----------|
| **security** | scan_for_pii, scan_for_secrets, find_duplicates, propose_cleanup, encrypt_file | 5 | 90% |
| **audit** | get_tool_log, generate_audit_report, export_audit_pdf | 3 | 100% |
| **document** | extract_text, diff_documents, create_pdf, create_docx | 4 | 83% |
| **filesystem** | list_dir, read_file, search_files | 3 | 80% |
| **system** | take_screenshot, get_system_info, get_disk_usage | 3 | 80% |
| **clipboard** | get_clipboard, set_clipboard, clipboard_history | 3 | 80% |

**Config:** `_models/config.yaml` has `enabled_servers` (6 servers) and `enabled_tools`
(21 tools). The app logs confirm: `running_servers=6 total_tools=21`.

---

## Working Folder Pattern (Claude Cowork Style)

Security scan presets use the **working folder** — a directory the user selects in the
file browser panel. This mirrors Claude Cowork's "project directory" pattern: the
**product** provides context, not the model. A 20B local model doesn't reliably ask
clarifying questions, so the app ensures the path is always provided.

**How it works:**
1. In the file browser (left panel), click the 📂 button on any folder to set it
   as the working folder. A blue badge shows the active folder name.
2. Security scan presets ("Scan for leaked secrets", "Find personal data") resolve
   `{cwd}` to the working folder path. These presets are **disabled** until a
   working folder is set.
3. Non-path presets (clipboard, system info, Downloads) work without a working folder.

**Demo setup:**
1. Navigate to `tests/fixtures/uc3/sample_files` in the file browser
2. Click the 📂 button next to `sample_files` to set it as working folder
3. Click "Scan for leaked secrets" → sends full absolute path → model executes

**Why this is reliable:**
- The model always receives a complete absolute path — no guessing, no asking
- Scanning an empty Desktop is impossible — the user must explicitly choose a folder
- Works with any model size (no instruction-following requirements)

---

## Demo 1: Security Scan + Audit Trail

**Story:** Every developer has `.env` files with API keys scattered across old projects.
You'd never upload your filesystem to GPT-4 to find them. This scan runs entirely on
your machine — no secrets ever leave the device.

| Step | Prompt | Expected Tool |
|------|--------|---------------|
| 1 | Set working folder → click "Scan for leaked secrets" | `security.scan_for_secrets` |
| 2 | "Show me the audit log of what you just did" | `audit.get_tool_log` |
| 3 | "Generate an audit report" | `audit.generate_audit_report` |
| 4 | "Export the report as a PDF" | `audit.export_audit_pdf` |

**Servers:** security (90%) + audit (100%) — highest-accuracy categories.
**Fixture:** `tests/fixtures/uc3/sample_files/has_api_key.env` (fake AWS + Stripe keys).
**Chain ID:** ms-simple-009 (PASSED in benchmark).

### Fallback prompts (if needed)

- Step 1 alt: "Check this folder for leaked secrets and passwords: tests/fixtures/uc3/sample_files"
- Step 1 direct: "Scan tests/fixtures/uc3/sample_files for exposed API keys" (skips interactive step)
- Step 2 alt: "What tool calls were just made?"
- Step 3 alt: "Create a summary report of the security findings"

---

## Demo 2: Contract Diff + PDF Report

**Story:** A freelancer gets a revised NDA. They need to know what changed. These are
confidential documents that should never leave the machine.

| Step | Prompt | Expected Tool |
|------|--------|---------------|
| 1 | "Extract the text from tests/fixtures/uc2/original_contract.txt" | `document.extract_text` |
| 2 | "Now compare it against tests/fixtures/uc2/revised_contract.txt" | `document.diff_documents` |
| 3 | "Create a PDF summary of the changes" | `document.create_pdf` |

**Servers:** document (83%) — strong single-step, proven chain.
**Fixture:** `tests/fixtures/uc2/original_contract.txt` + `revised_contract.txt`.
**Chain ID:** ms-simple-012 (PASSED).

### Fallback prompts

- Step 1 alt: "Read the original contract at tests/fixtures/uc2/original_contract.txt"
- Step 2 alt: "Show me the differences between the original and revised versions"
- Step 3 alt: "Generate a PDF report of the contract differences"

---

## Demo 3: Screenshot to Clipboard

**Story:** You're debugging and hit a cryptic error dialog. You screenshot it. With
LocalCowork, the text is on your clipboard in seconds. No screenshot ever leaves
your machine.

| Step | Prompt | Expected Tool |
|------|--------|---------------|
| 1 | "Take a screenshot of my screen" | `system.take_screenshot` |
| 2 | "Copy the text 'LFM2-24B demo complete' to my clipboard" | `clipboard.set_clipboard` |
| 3 | "Show me my clipboard history" | `clipboard.clipboard_history` |

**Servers:** system (80%) + clipboard (80%).
**Chain ID:** ms-simple-011 (PASSED).

> Note: OCR is not in the curated 20-tool set (75% accuracy). For the demo, step 2
> uses `clipboard.set_clipboard` directly instead of chaining through OCR extraction.
> If OCR is added back (server + tools), the full screenshot→OCR→clipboard chain works.

### Fallback prompts

- Step 1 alt: "Capture a screenshot right now"
- Step 2 alt: "Put this text on my clipboard: LFM2-24B demo complete"
- Step 3 alt: "What's been on my clipboard recently?"

---

## Demo 4: Security Deep Dive (PII + Cleanup + Encrypt)

**Story:** An HR team needs to find and secure employee data scattered across shared
folders. Scan for PII, find duplicates, get a cleanup plan, and encrypt the sensitive files.

| Step | Prompt | Expected Tool |
|------|--------|---------------|
| 1 | Set working folder → click "Find personal data" | `security.scan_for_pii` |
| 2 | "Check for duplicate files in that folder" | `security.find_duplicates` |
| 3 | "Suggest a cleanup plan for the findings" | `security.propose_cleanup` |
| 4 | "Encrypt the file that contains SSN data" | `security.encrypt_file` |

**Servers:** security (90%) — 4-step single-server chain.
**Fixture:** `tests/fixtures/uc3/sample_files/has_ssn.txt` (fake SSN data).
**Chain ID:** ms-medium-007 (PASSED).

### Fallback prompts

- Step 1 alt: "Find files containing social security numbers in tests/fixtures/uc3/sample_files" (skips interactive step)
- Step 4 alt: "Encrypt tests/fixtures/uc3/sample_files/has_ssn.txt"

---

## Bonus: File Browsing (warm-up)

Quick 3-step warm-up to show the app working before the main demos.

| Step | Prompt | Expected Tool |
|------|--------|---------------|
| 1 | "List what's in my Downloads folder" | `filesystem.list_dir` |
| 2 | "Search for any .txt files in tests/fixtures" | `filesystem.search_files` |
| 3 | "Read the first receipt file" | `filesystem.read_file` |

**Server:** filesystem (80%).
**Chain ID:** ms-simple-001 (PASSED).

---

## Demo Video Playbook

### Prerequisites

```bash
# Terminal 1 — Model server
llama-server \
  --model _models/lfm2-24b-a2b/lfm2-24b-a2b-Q4_K_M.gguf \
  --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --flash-attn

# Verify
curl -s http://localhost:8080/health
# {"status":"ok"}

# Terminal 2 — Desktop app
cd ~/Projects/localCoWork
cargo tauri dev

# Terminal 3 — Tail logs (optional)
tail -f ~/Library/Application\ Support/com.localcowork.app/agent.log
```

**Verify in logs:**
```
filtered MCP servers by enabled_servers allowlist before=15 after=6
filtered tool registry by enabled_tools allowlist before=40 after=21
MCP client initialized running_servers=6 total_tools=21
```

### Recommended Demo Order

1. **File browsing** (warm-up) — shows the app working, low stakes
2. **Security scan + audit** (strongest: 90%+100%) — guaranteed clean demo
3. **Contract diff + PDF** (cross-capability) — document analysis → generation
4. **Screenshot → clipboard** (visual, instant) — the "magic moment"
5. **Security deep dive** (4-step chain) — shows multi-step reliability

### Tips for Recording

1. **Start a fresh chat session** before each workflow
2. **Use relative paths** from project root: `tests/fixtures/uc3/sample_files`
3. **Show the confirmation dialog** — the human-in-the-loop UX is the narrative
4. **Show latency** — tool selection completes in <500ms per step
5. **Show the tool trace** — the ToolTrace panel shows which tool was selected
6. **If a step picks the wrong tool** — correct and move on. The blog's point
   is that correction is cheap at this speed

---

## Hardware Requirements

- Apple M4 Max, 36 GB unified memory (blog spec)
- Also works: Apple M4 / M2 Pro, 16+ GB unified memory
- macOS 14+ (Sonoma or later)
- ~14.5 GB VRAM for Q4_K_M quantization

## How to Reproduce Benchmarks

```bash
# Full 67-tool benchmark (blog numbers)
npx tsx tests/model-behavior/benchmark-lfm.ts --greedy --endpoint http://localhost:8080

# Focused 21-tool benchmark (demo set)
./scripts/benchmark-demo.sh --endpoint http://localhost:8080

# Results in tests/model-behavior/.results/
```
