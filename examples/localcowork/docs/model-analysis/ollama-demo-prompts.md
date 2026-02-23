# LFM2-24B-A2B — Tool-Calling Prompts for OpenClaw Demo

Curated prompts from our benchmark suite (100 single-step + 50 multi-step). Only prompts with **85%+ accuracy** on greedy sampling are included.

Model: `LiquidAI/LFM2-24B-A2B-Preview` (Q4_K_M GGUF)

---

## System Prompt

```
You are LocalCowork, a desktop assistant with full access to local tools for files, documents, tasks, calendar, email, and more.

IMPORTANT RULES:
1. ALWAYS call the appropriate tool. Never say "I can't do that" or "I don't have that capability" — you have tools for file deletion, file writing, clipboard, encryption, audit logging, and more.
2. Call exactly one tool per response using bracket format: [server.tool_name(param="value")]
3. Use the full dotted tool name (e.g., filesystem.list_dir, NOT list_dir).
4. For file rename operations, use filesystem.move_file with the new name as destination.
5. For extracting text from images or screenshots, use ocr tools. For document files (PDF, DOCX), use document tools.

Available tools: [<tool definitions — see below>]
```

## Tool Call Format

LFM2 produces bracket-format tool calls, **not** OpenAI function_call JSON:

```
[filesystem.list_dir(path="/Users/me/Documents")]
[task.create_task(title="Review Q4 report", due_date="Friday", priority="high")]
[calendar.create_event(title="Design sync", time="2pm", duration="1h")]
```

## Sampling Config

```
temperature: 0
top_p: 1.0
max_tokens: 512
```

Greedy sampling. Our benchmarks confirmed temp=0.1/top_p=0.1 produces identical results, so either config works.

---

## Tool Definitions (67 tools, 14 servers)

These are contrastive, synonym-augmented descriptions. The wording matters — negations ("NOT for X") and usage hints ("Use for Y, not Z") help the model disambiguate.

```json
[
  {"name": "filesystem.list_dir", "description": "List all files and folders in a single directory. Use for browsing, not searching"},
  {"name": "filesystem.read_file", "description": "Read the text contents of a file by path. For spreadsheets use document.read_spreadsheet instead"},
  {"name": "filesystem.write_file", "description": "Write, create, or save text content to a file on disk"},
  {"name": "filesystem.move_file", "description": "Move or rename a file. Handles renaming by moving to the same directory with a new name"},
  {"name": "filesystem.copy_file", "description": "Copy or duplicate a file to a new location or backup"},
  {"name": "filesystem.delete_file", "description": "Delete, remove, or trash a file from disk"},
  {"name": "filesystem.search_files", "description": "Search, find, or locate files by name pattern, extension, size, or content match. Use for recursive or filtered file searches"},
  {"name": "filesystem.get_metadata", "description": "Get file metadata: size, creation date, modification date, permissions, and file type"},
  {"name": "filesystem.watch_folder", "description": "Watch a folder for real-time file change notifications. Only for monitoring, not for listing"},
  {"name": "document.extract_text", "description": "Extract text from document files: PDF, DOCX, TXT, RTF, MD. NOT for images or screenshots — use ocr tools instead"},
  {"name": "document.convert_format", "description": "Convert a document between formats: PDF to DOCX, Markdown to HTML, DOCX to PDF, etc."},
  {"name": "document.diff_documents", "description": "Compare two document versions and show the differences between them. Use for tracking changes"},
  {"name": "document.create_pdf", "description": "Create a new PDF document from text or markdown content. Use for generating reports or summaries as PDF"},
  {"name": "document.fill_pdf_form", "description": "Fill in specific fields of an existing PDF form. Requires an existing PDF with form fields"},
  {"name": "document.merge_pdfs", "description": "Merge or combine multiple PDF files into a single PDF"},
  {"name": "document.create_docx", "description": "Create a new Word DOCX document from text or markdown content"},
  {"name": "document.read_spreadsheet", "description": "Read data from spreadsheet files: CSV, XLSX, XLS. Use for tabular data, NOT for reading text files"},
  {"name": "ocr.extract_text_from_image", "description": "OCR: Extract text from images, screenshots, photos, or scanned pictures. NOT for PDF or document files"},
  {"name": "ocr.extract_text_from_pdf", "description": "OCR: Extract text from scanned PDFs where text is embedded as images. For normal PDFs use document.extract_text"},
  {"name": "ocr.extract_structured_data", "description": "OCR: Extract structured fields from images of receipts, invoices, business cards, or forms. Returns named fields like amount, date, vendor"},
  {"name": "ocr.extract_table", "description": "OCR: Extract table or grid data from images or scanned documents into rows and columns"},
  {"name": "data.write_csv", "description": "Write or export structured data to a CSV file"},
  {"name": "data.write_sqlite", "description": "Write, insert, or import records into a SQLite database table"},
  {"name": "data.query_sqlite", "description": "Query or read from a SQLite database using SQL. Use for database lookups, reports, and analytics"},
  {"name": "data.deduplicate_records", "description": "Find and remove duplicate records in a database table based on key columns"},
  {"name": "data.summarize_anomalies", "description": "Detect anomalies, outliers, unusual patterns, or irregularities in database data"},
  {"name": "audit.get_tool_log", "description": "Get the log of which tools were used, executed, or called in previous sessions"},
  {"name": "audit.get_session_summary", "description": "Get a summary or recap of what happened in a previous work session"},
  {"name": "audit.generate_audit_report", "description": "Generate a comprehensive audit report covering tool usage over a date range"},
  {"name": "audit.export_audit_pdf", "description": "Export an existing audit report as a PDF file"},
  {"name": "knowledge.index_folder", "description": "Index a folder of documents for semantic search and RAG question-answering"},
  {"name": "knowledge.search_documents", "description": "Semantic search across previously indexed documents by meaning, not just keywords"},
  {"name": "knowledge.ask_about_files", "description": "Ask a natural language question about previously indexed files and get an AI-generated answer"},
  {"name": "knowledge.update_index", "description": "Update the search index when documents have been added or changed since last indexing"},
  {"name": "knowledge.get_related_chunks", "description": "Retrieve related text passages or chunks from indexed documents for a given topic"},
  {"name": "security.scan_for_pii", "description": "Scan for personal identity data: names, SSNs, addresses, phone numbers, dates of birth, email addresses"},
  {"name": "security.scan_for_secrets", "description": "Scan for credentials and secrets: API keys, passwords, tokens, private keys, connection strings"},
  {"name": "security.find_duplicates", "description": "Find duplicate or identical files in a directory to free up storage space"},
  {"name": "security.propose_cleanup", "description": "Suggest files that can be safely deleted or archived to clean up a directory"},
  {"name": "security.encrypt_file", "description": "Encrypt, lock down, or password-protect a file for secure storage"},
  {"name": "security.decrypt_file", "description": "Decrypt or unlock a previously encrypted file"},
  {"name": "task.create_task", "description": "Create a new task or to-do item with optional due date and priority"},
  {"name": "task.list_tasks", "description": "List tasks or to-dos with optional filters by status, priority, or date"},
  {"name": "task.update_task", "description": "Update, modify, or mark a task as complete, change its status, due date, or priority"},
  {"name": "task.get_overdue", "description": "Get a list of tasks that are past their due date"},
  {"name": "task.daily_briefing", "description": "Generate a daily briefing summarizing upcoming tasks, deadlines, and priorities"},
  {"name": "calendar.list_events", "description": "View, show, or list your existing scheduled meetings and calendar events for a date range"},
  {"name": "calendar.create_event", "description": "Schedule a new meeting, appointment, or event with other people on the calendar"},
  {"name": "calendar.find_free_slots", "description": "Find open, available, or free time slots when you have no events scheduled"},
  {"name": "calendar.create_time_block", "description": "Block personal focus time or deep work time for yourself with no attendees"},
  {"name": "email.draft_email", "description": "Compose, draft, or write a new email message. Does NOT send it — creates a draft only"},
  {"name": "email.list_drafts", "description": "List previously drafted email messages"},
  {"name": "email.search_emails", "description": "Search through emails by keyword, sender, subject, or date"},
  {"name": "email.summarize_thread", "description": "Summarize or recap an email conversation thread"},
  {"name": "email.send_draft", "description": "Send a previously drafted email. Requires a draft_id from draft_email or list_drafts"},
  {"name": "meeting.transcribe_audio", "description": "Transcribe speech from an audio or video recording file into text"},
  {"name": "meeting.extract_action_items", "description": "Extract action items and to-dos from meeting notes or a transcript"},
  {"name": "meeting.extract_commitments", "description": "Extract promises, commitments, and agreements made during a meeting"},
  {"name": "meeting.generate_minutes", "description": "Generate complete formatted meeting minutes from a transcript. Includes summary, decisions, and action items"},
  {"name": "clipboard.get_clipboard", "description": "Get or read the current system clipboard contents"},
  {"name": "clipboard.set_clipboard", "description": "Copy text to the system clipboard. Sets or writes to the clipboard"},
  {"name": "clipboard.clipboard_history", "description": "Get the history of recent clipboard entries"},
  {"name": "system.get_system_info", "description": "Get system information: OS, CPU, memory, disk space, and hardware details"},
  {"name": "system.open_application", "description": "Launch or open a desktop application by name"},
  {"name": "system.take_screenshot", "description": "Capture a screenshot of the screen"},
  {"name": "system.list_processes", "description": "List currently running processes and applications"},
  {"name": "system.open_file_with", "description": "Open a file using a specific desktop application (e.g., open PDF in Preview)"}
]
```

---

## Demo Prompts — Single Step (85%+ accuracy)

### Audit (100% accuracy)

| Prompt | Expected Tool Call |
|--------|-------------------|
| "Show me what tools were used in the last session" | `[audit.get_tool_log()]` |
| "Generate an audit report for this week" | `[audit.generate_audit_report()]` |
| "Give me a summary of what happened in my last work session" | `[audit.get_session_summary()]` |

### Task Management (87.5% accuracy)

| Prompt | Expected Tool Call |
|--------|-------------------|
| "Create a new task: review Q4 report by Friday" | `[task.create_task(title="Review Q4 report", due_date="Friday")]` |
| "What tasks are overdue?" | `[task.get_overdue()]` |
| "Show me my daily briefing for today" | `[task.daily_briefing()]` |
| "List all my current tasks" | `[task.list_tasks()]` |
| "Mark the 'Prepare slides' task as complete" | `[task.update_task(title="Prepare slides", status="complete")]` |
| "Add a high-priority task to call the vendor about pricing" | `[task.create_task(title="Call vendor about pricing", priority="high")]` |

### Calendar (85.7% accuracy)

| Prompt | Expected Tool Call |
|--------|-------------------|
| "What meetings do I have today?" | `[calendar.list_events()]` |
| "Schedule a 1-hour meeting tomorrow at 2pm with the design team" | `[calendar.create_event(title="Meeting with design team", time="2pm", duration="1h")]` |
| "Find a free 30-minute slot this afternoon for a quick sync" | `[calendar.find_free_slots()]` |
| "Block off 2 hours tomorrow morning for deep work" | `[calendar.create_time_block()]` |
| "Show me my schedule for next week" | `[calendar.list_events()]` |

### Meeting & Audio (85.7% accuracy)

| Prompt | Expected Tool Call |
|--------|-------------------|
| "Transcribe this meeting recording" | `[meeting.transcribe_audio(path="...")]` |
| "Extract action items from the meeting notes" | `[meeting.extract_action_items()]` |
| "Generate meeting minutes from the team standup recording" | `[meeting.generate_minutes()]` |
| "What commitments were made during the client call?" | `[meeting.extract_commitments()]` |
| "Convert this voice memo to text" | `[meeting.transcribe_audio(path="...")]` |

### File Operations (selected 85%+ prompts)

| Prompt | Expected Tool Call |
|--------|-------------------|
| "What files are in my Documents folder?" | `[filesystem.list_dir(path="~/Documents")]` |
| "Move report.pdf to the Archive folder" | `[filesystem.move_file(source="report.pdf", destination="Archive/report.pdf")]` |
| "Search for all PDF files in Downloads" | `[filesystem.search_files(path="~/Downloads", pattern="*.pdf")]` |
| "Copy config.yaml to a backup" | `[filesystem.copy_file(source="config.yaml", destination="config.yaml.bak")]` |
| "Write a new file called notes.md with the text 'Meeting notes for Monday'" | `[filesystem.write_file(path="notes.md", content="Meeting notes for Monday")]` |
| "What is the file size and last modified date of presentation.pptx?" | `[filesystem.get_metadata(path="presentation.pptx")]` |
| "Show me all .json files in the project directory recursively" | `[filesystem.search_files(path=".", pattern="*.json")]` |
| "Rename quarterly-report-v1.docx to quarterly-report-final.docx" | `[filesystem.move_file(source="quarterly-report-v1.docx", destination="quarterly-report-final.docx")]` |

### Security (80% accuracy)

| Prompt | Expected Tool Call |
|--------|-------------------|
| "Scan my Downloads folder for any sensitive personal data" | `[security.scan_for_pii(path="~/Downloads")]` |
| "Check this project for exposed API keys or secrets" | `[security.scan_for_secrets()]` |
| "Encrypt this confidential file before sharing" | `[security.encrypt_file(path="...")]` |
| "Decrypt the encrypted report file" | `[security.decrypt_file(path="...")]` |
| "Find duplicate files in my Documents folder to free up space" | `[security.find_duplicates()]` |

### Email (75% accuracy)

| Prompt | Expected Tool Call |
|--------|-------------------|
| "Draft an email to John about the project update" | `[email.draft_email(to="John", subject="Project update")]` |
| "Search my emails for anything about the merger" | `[email.search_emails(query="merger")]` |
| "Show me all my draft emails" | `[email.list_drafts()]` |
| "Send the draft email I wrote to Sarah" | `[email.send_draft()]` |

### System & Clipboard (80% accuracy)

| Prompt | Expected Tool Call |
|--------|-------------------|
| "What is on my clipboard right now?" | `[clipboard.get_clipboard()]` |
| "Copy this text to my clipboard: 'Meeting at 3pm in Room 201'" | `[clipboard.set_clipboard(text="Meeting at 3pm in Room 201")]` |
| "What processes are currently running on my system?" | `[system.list_processes()]` |
| "Show me my clipboard history" | `[clipboard.clipboard_history()]` |

---

## Demo Prompts — Multi-Step Chains

Each step is a separate user turn. The model calls one tool per turn.

### Screenshot → OCR → Clipboard
```
User: "Take a screenshot of my current screen"
Model: [system.take_screenshot()]

User: "Extract the text from that screenshot"
Model: [ocr.extract_text_from_image(path="...")]

User: "Copy that text to my clipboard"
Model: [clipboard.set_clipboard(text="...")]
```

### Security Scan → Audit → Export
```
User: "Scan my project directory for any exposed secrets or API keys"
Model: [security.scan_for_secrets(path="...")]

User: "Generate an audit report summarizing the scan results"
Model: [audit.generate_audit_report()]

User: "Export that audit report as a PDF"
Model: [audit.export_audit_pdf()]
```

### Daily Workflow
```
User: "Show me my daily briefing"
Model: [task.daily_briefing()]

User: "What meetings do I have today?"
Model: [calendar.list_events()]

User: "Add a task to prepare the slides for the 2pm meeting"
Model: [task.create_task(title="Prepare slides for 2pm meeting")]
```

### Document Compare → PDF → File Move
```
User: "Compare contract-v1.pdf and contract-v2.pdf"
Model: [document.diff_documents(file1="contract-v1.pdf", file2="contract-v2.pdf")]

User: "Create a PDF summarizing the changes between the two versions"
Model: [document.create_pdf()]

User: "Move the diff report to the Contracts folder"
Model: [filesystem.move_file(source="...", destination="Contracts/...")]
```

### PII Scan → Encrypt → Audit Log
```
User: "Scan the HR folder for files containing personal information"
Model: [security.scan_for_pii(path="HR/")]

User: "Encrypt the file that contains social security numbers"
Model: [security.encrypt_file(path="...")]

User: "Show me the audit log of what we just did"
Model: [audit.get_tool_log()]
```

---

## Notes for Integration

- **Tool definitions in system prompt**: Flatten the JSON array into the system prompt as shown above. The model expects them inline, not as a separate `tools` parameter.
- **Bracket parsing**: Look for `[server.tool_name(...)]` patterns in model output. The model may also wrap them in markdown code blocks or include prose before/after the bracket call.
- **One tool per turn**: The system prompt enforces single tool calls. For multi-step workflows, use multiple turns with conversational context.
- **Contrastive descriptions are critical**: The negation wording (e.g., "NOT for images — use ocr tools instead") is what prevents cross-server confusion. Don't simplify the descriptions.
