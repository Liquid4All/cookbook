#!/usr/bin/env bash
# Test the fine-tuned LFM2.5-1.2B-Router model via llama-server (v2)
#
# Covers all 15 servers, variable K sizes, and server-prefix drills.
# Usage: ./scripts/test-router-ft.sh [port] [--k15|--k25|--k35|--drills|--all]
#   Default port: 8085 (temporary test port)
#   Default suite: --all (run all test groups)

PORT="${1:-8085}"
SUITE="${2:---all}"
BASE_URL="http://localhost:${PORT}/v1/chat/completions"

# ---------------------------------------------------------------------------
# System prompts with different K values
# ---------------------------------------------------------------------------

# K=15: Production configuration (matches RAG pre-filter)
SYSTEM_K15='You are LocalCowork, a desktop AI assistant that runs entirely on-device. You have access to the following tools. ALWAYS call exactly one tool using bracket syntax: [server.tool(param="value")]. NEVER ask questions. NEVER say you cannot help. ALWAYS select the most appropriate tool.

Available tools:
1. filesystem.list_dir — List files and directories at a given path
2. filesystem.read_file — Read the contents of a file
3. filesystem.write_file — Write content to a file
4. document.extract_text — Extract text from PDF, DOCX, or other document formats
5. knowledge.search_documents — Search the local knowledge base / RAG index
6. data.query_sqlite — Query a SQLite database
7. security.scan_for_pii — Scan text for personally identifiable information
8. security.encrypt_file — Encrypt a file with a passphrase
9. task.create_task — Create a new task with optional due date and priority
10. task.get_overdue — Get overdue tasks
11. calendar.list_events — List calendar events in a date range
12. calendar.create_event — Create a new calendar event
13. email.search_emails — Search email messages by keyword or sender
14. email.draft_email — Draft a new email message
15. meeting.transcribe_audio — Transcribe an audio recording'

# K=25: Expanded candidate set (robustness test)
SYSTEM_K25='You are LocalCowork, a desktop AI assistant that runs entirely on-device. You have access to the following tools. ALWAYS call exactly one tool using bracket syntax: [server.tool(param="value")]. NEVER ask questions. NEVER say you cannot help. ALWAYS select the most appropriate tool.

Available tools:
1. filesystem.list_dir — List files and directories at a given path
2. filesystem.read_file — Read the contents of a file
3. filesystem.write_file — Write content to a file
4. filesystem.move_file — Move or rename a file or directory
5. filesystem.search_files — Search for files by name or content
6. document.extract_text — Extract text from PDF, DOCX, or other document formats
7. document.convert_format — Convert a document between formats
8. ocr.extract_text_from_image — Extract text from an image using OCR
9. ocr.extract_table — Extract a table from an image or PDF
10. knowledge.search_documents — Search the local knowledge base / RAG index
11. knowledge.index_folder — Index a folder into the knowledge base
12. data.query_sqlite — Query a SQLite database
13. data.write_csv — Write data to a CSV file
14. data.summarize_anomalies — Summarize anomalies in a dataset
15. security.scan_for_pii — Scan text for personally identifiable information
16. security.encrypt_file — Encrypt a file with a passphrase
17. task.create_task — Create a new task with optional due date and priority
18. task.get_overdue — Get overdue tasks
19. task.daily_briefing — Generate a daily briefing of tasks
20. calendar.list_events — List calendar events in a date range
21. calendar.create_event — Create a new calendar event
22. email.search_emails — Search email messages by keyword or sender
23. email.draft_email — Draft a new email message
24. meeting.transcribe_audio — Transcribe an audio recording
25. meeting.generate_minutes — Generate meeting minutes from a transcript'

# K=35: Heavy distractor load
SYSTEM_K35='You are LocalCowork, a desktop AI assistant that runs entirely on-device. You have access to the following tools. ALWAYS call exactly one tool using bracket syntax: [server.tool(param="value")]. NEVER ask questions. NEVER say you cannot help. ALWAYS select the most appropriate tool.

Available tools:
1. filesystem.list_dir — List files and directories at a given path
2. filesystem.read_file — Read the contents of a file
3. filesystem.write_file — Write content to a file
4. filesystem.move_file — Move or rename a file or directory
5. filesystem.copy_file — Copy a file to a new location
6. filesystem.delete_file — Delete a file or directory
7. filesystem.search_files — Search for files by name or content
8. document.extract_text — Extract text from PDF, DOCX, or other documents
9. document.convert_format — Convert a document between formats
10. document.create_pdf — Create a PDF from content
11. ocr.extract_text_from_image — Extract text from an image using OCR
12. ocr.extract_table — Extract a table from an image or PDF
13. knowledge.search_documents — Search the local knowledge base / RAG index
14. knowledge.index_folder — Index a folder into the knowledge base
15. knowledge.ask_about_files — Ask a question about indexed files
16. data.query_sqlite — Query a SQLite database
17. data.write_csv — Write data to a CSV file
18. data.deduplicate_records — Remove duplicate records from a dataset
19. security.scan_for_pii — Scan text for PII
20. security.scan_for_secrets — Scan files for leaked secrets or API keys
21. security.encrypt_file — Encrypt a file with a passphrase
22. task.create_task — Create a new task
23. task.list_tasks — List tasks with optional filters
24. task.get_overdue — Get overdue tasks
25. task.daily_briefing — Generate a daily briefing of tasks
26. calendar.list_events — List calendar events in a date range
27. calendar.create_event — Create a new calendar event
28. calendar.find_free_slots — Find free time slots in a calendar
29. email.search_emails — Search email messages
30. email.draft_email — Draft a new email message
31. email.send_draft — Send a previously drafted email
32. meeting.transcribe_audio — Transcribe an audio recording
33. meeting.generate_minutes — Generate meeting minutes from transcript
34. audit.get_tool_log — Get log of tool calls for audit
35. clipboard.get_clipboard — Get current clipboard contents'

# K=15 with system/system-settings/screenshot tools (new servers test)
SYSTEM_NEW_SERVERS='You are LocalCowork, a desktop AI assistant that runs entirely on-device. You have access to the following tools. ALWAYS call exactly one tool using bracket syntax: [server.tool(param="value")]. NEVER ask questions. NEVER say you cannot help. ALWAYS select the most appropriate tool.

Available tools:
1. system.get_system_info — Get system information (OS, CPU, memory)
2. system.open_application — Open an application by name
3. system.take_screenshot — Take a screenshot of the screen
4. system.list_processes — List running processes
5. system.kill_process — Terminate a running process by PID
6. system.get_memory_usage — Get current memory usage statistics
7. system.get_disk_usage — Get disk space usage for mounted volumes
8. system.get_cpu_usage — Get current CPU utilization
9. system.get_network_info — Get network interface information
10. system-settings.get_display_settings — Get display brightness, resolution, arrangement
11. system-settings.set_display_sleep — Set display sleep timeout
12. system-settings.get_audio_settings — Get audio input/output device and volume
13. system-settings.set_audio_volume — Set system audio volume
14. system-settings.toggle_do_not_disturb — Toggle Do Not Disturb mode
15. screenshot.capture_and_extract — Capture screenshot and extract text via OCR'

# K=15 server-prefix drill: task vs calendar confusion
SYSTEM_DRILL_TASK_CAL='You are LocalCowork, a desktop AI assistant that runs entirely on-device. You have access to the following tools. ALWAYS call exactly one tool using bracket syntax: [server.tool(param="value")]. NEVER ask questions. NEVER say you cannot help. ALWAYS select the most appropriate tool.

Available tools:
1. task.create_task — Create a new task with optional due date and priority
2. task.list_tasks — List tasks with optional filters
3. task.update_task — Update or complete a task
4. task.get_overdue — Get tasks that are past their due date
5. task.daily_briefing — Generate a daily task briefing
6. calendar.list_events — List calendar events in a date range
7. calendar.create_event — Create a new calendar event
8. calendar.find_free_slots — Find free time slots in a calendar
9. calendar.create_time_block — Block time on the calendar
10. email.search_emails — Search email messages
11. email.draft_email — Draft a new email message
12. meeting.transcribe_audio — Transcribe an audio recording
13. filesystem.list_dir — List files and directories
14. knowledge.search_documents — Search the knowledge base
15. security.scan_for_pii — Scan text for PII'

# K=15 server-prefix drill: system vs system-settings
SYSTEM_DRILL_SYS='You are LocalCowork, a desktop AI assistant that runs entirely on-device. You have access to the following tools. ALWAYS call exactly one tool using bracket syntax: [server.tool(param="value")]. NEVER ask questions. NEVER say you cannot help. ALWAYS select the most appropriate tool.

Available tools:
1. system.get_system_info — Get system information (OS, CPU, memory)
2. system.open_application — Open an application by name
3. system.list_processes — List running processes
4. system.kill_process — Terminate a running process by PID
5. system.get_memory_usage — Get current memory usage statistics
6. system.get_cpu_usage — Get current CPU utilization
7. system-settings.get_display_settings — Get display settings
8. system-settings.set_display_sleep — Set display sleep timeout
9. system-settings.get_audio_settings — Get audio settings
10. system-settings.set_audio_volume — Set system audio volume
11. system-settings.get_power_settings — Get power/energy saver settings
12. system-settings.toggle_do_not_disturb — Toggle Do Not Disturb mode
13. screenshot.capture_and_extract — Capture screenshot and OCR text
14. screenshot.extract_ui_elements — Identify UI elements in a screenshot
15. screenshot.suggest_actions — Suggest actions based on screen content'

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

echo "=== Testing Fine-Tuned Router v2 on port ${PORT} ==="
echo "Suite: ${SUITE}"
echo ""

PASS_COUNT=0
FAIL_COUNT=0
TOTAL_TESTS=0
CURRENT_GROUP=""

start_group() {
    CURRENT_GROUP="$1"
    echo "========================================"
    echo "GROUP: ${CURRENT_GROUP}"
    echo "========================================"
    echo ""
}

run_test() {
    local test_name="$1"
    local user_prompt="$2"
    local expected="$3"
    local system_prompt="${4:-$SYSTEM_K15}"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo "--- Test ${TOTAL_TESTS}: ${test_name} ---"
    echo "User: ${user_prompt}"
    echo "Expected: ${expected}"

    local response
    response=$(curl -s "${BASE_URL}" \
        -H "Content-Type: application/json" \
        -d @- <<PAYLOAD
{
    "model": "lfm25-router-ft",
    "messages": [
        {
            "role": "system",
            "content": $(echo "${system_prompt}" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")
        },
        {
            "role": "user",
            "content": $(echo "${user_prompt}" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")
        }
    ],
    "temperature": 0.1,
    "max_tokens": 256
}
PAYLOAD
    )

    local content
    content=$(echo "${response}" | python3 -c "import sys, json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null)

    if [ -z "${content}" ]; then
        echo "ERROR: No response from model"
        echo "Raw: ${response}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        echo "Model: ${content}"
        if echo "${content}" | grep -q "${expected}"; then
            echo "PASS ✓"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo "FAIL ✗"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    fi
    echo ""
}

# ============================================================================
# GROUP 1: K=15 — Core servers (production configuration)
# ============================================================================
if [[ "${SUITE}" == "--all" || "${SUITE}" == "--k15" ]]; then

start_group "K=15 Core Servers"

# --- filesystem (3 tests) ---
run_test "filesystem: list_dir" \
    "Show me what files are in my Documents folder" \
    "filesystem.list_dir"

run_test "filesystem: read_file" \
    "Read the contents of README.md" \
    "filesystem.read_file"

run_test "filesystem: write_file" \
    "Create a new file called notes.txt with the meeting summary" \
    "filesystem.write_file"

# --- document (1 test) ---
run_test "document: extract_text" \
    "Extract the text from this PDF: invoice.pdf" \
    "document.extract_text"

# --- knowledge (1 test) ---
run_test "knowledge: search_documents" \
    "Find information about API authentication in my notes" \
    "knowledge.search_documents"

# --- data (1 test) ---
run_test "data: query_sqlite" \
    "What are the total sales from the analytics database?" \
    "data.query_sqlite"

# --- security (2 tests) ---
run_test "security: scan_for_pii" \
    "Check this document for personal information: John Smith 555-01-2345" \
    "security.scan_for_pii"

run_test "security: encrypt_file" \
    "Encrypt the file secrets.txt with a passphrase" \
    "security.encrypt_file"

# --- task (2 tests) ---
run_test "task: create_task" \
    "Create a task to review the quarterly report by Friday" \
    "task.create_task"

run_test "task: get_overdue" \
    "Show me tasks that are past their due date" \
    "task.get_overdue"

# --- calendar (2 tests) ---
run_test "calendar: list_events" \
    "What meetings do I have this week?" \
    "calendar.list_events"

run_test "calendar: create_event" \
    "Schedule a team standup for tomorrow at 9am" \
    "calendar.create_event"

# --- email (2 tests) ---
run_test "email: search_emails" \
    "Find emails from Sarah about the project update" \
    "email.search_emails"

run_test "email: draft_email" \
    "Draft an email to the team about the deadline change" \
    "email.draft_email"

# --- meeting (1 test) ---
run_test "meeting: transcribe_audio" \
    "Transcribe the recording from today's standup meeting" \
    "meeting.transcribe_audio"

# --- Terse prompts (3 tests) ---
run_test "terse: list files" \
    "list my files" \
    "filesystem.list_dir"

run_test "terse: search email" \
    "emails from Bob" \
    "email.search_emails"

run_test "terse: new task" \
    "add task buy groceries" \
    "task.create_task"

fi  # --k15

# ============================================================================
# GROUP 2: K=25 — Expanded candidate set
# ============================================================================
if [[ "${SUITE}" == "--all" || "${SUITE}" == "--k25" ]]; then

start_group "K=25 Expanded Set"

run_test "K25: query_sqlite" \
    "Run a query on the users database to find inactive accounts" \
    "data.query_sqlite" \
    "${SYSTEM_K25}"

run_test "K25: ocr.extract_text_from_image" \
    "Extract the text from this screenshot of a receipt" \
    "ocr.extract_text_from_image" \
    "${SYSTEM_K25}"

run_test "K25: summarize_anomalies" \
    "Find unusual patterns in the server metrics dataset" \
    "data.summarize_anomalies" \
    "${SYSTEM_K25}"

run_test "K25: index_folder" \
    "Index all the documents in the project wiki folder" \
    "knowledge.index_folder" \
    "${SYSTEM_K25}"

run_test "K25: generate_minutes" \
    "Create meeting minutes from the standup transcript" \
    "meeting.generate_minutes" \
    "${SYSTEM_K25}"

run_test "K25: convert_format" \
    "Convert report.docx to PDF format" \
    "document.convert_format" \
    "${SYSTEM_K25}"

fi  # --k25

# ============================================================================
# GROUP 3: K=35 — Heavy distractor load
# ============================================================================
if [[ "${SUITE}" == "--all" || "${SUITE}" == "--k35" ]]; then

start_group "K=35 Heavy Distractor"

run_test "K35: scan_for_secrets" \
    "Scan my source code for leaked API keys" \
    "security.scan_for_secrets" \
    "${SYSTEM_K35}"

run_test "K35: deduplicate_records" \
    "Remove duplicate entries from the contacts CSV" \
    "data.deduplicate_records" \
    "${SYSTEM_K35}"

run_test "K35: find_free_slots" \
    "When am I free next Tuesday afternoon?" \
    "calendar.find_free_slots" \
    "${SYSTEM_K35}"

run_test "K35: send_draft" \
    "Send the email draft I wrote to the team" \
    "email.send_draft" \
    "${SYSTEM_K35}"

run_test "K35: get_tool_log" \
    "Show me the audit log of recent tool calls" \
    "audit.get_tool_log" \
    "${SYSTEM_K35}"

run_test "K35: clipboard.get_clipboard" \
    "What is on my clipboard right now?" \
    "clipboard.get_clipboard" \
    "${SYSTEM_K35}"

fi  # --k35

# ============================================================================
# GROUP 4: New servers (system, system-settings, screenshot)
# ============================================================================
if [[ "${SUITE}" == "--all" || "${SUITE}" == "--new" ]]; then

start_group "New Servers (system, system-settings, screenshot)"

# --- system (3 tests) ---
run_test "system: get_system_info" \
    "What operating system am I running?" \
    "system.get_system_info" \
    "${SYSTEM_NEW_SERVERS}"

run_test "system: get_memory_usage" \
    "How much RAM is currently being used?" \
    "system.get_memory_usage" \
    "${SYSTEM_NEW_SERVERS}"

run_test "system: list_processes" \
    "Show me all running processes on my machine" \
    "system.list_processes" \
    "${SYSTEM_NEW_SERVERS}"

# --- system-settings (3 tests) ---
run_test "system-settings: get_display_settings" \
    "What are my display brightness and resolution settings?" \
    "system-settings.get_display_settings" \
    "${SYSTEM_NEW_SERVERS}"

run_test "system-settings: set_audio_volume" \
    "Set my system volume to 50 percent" \
    "system-settings.set_audio_volume" \
    "${SYSTEM_NEW_SERVERS}"

run_test "system-settings: toggle_do_not_disturb" \
    "Turn on do not disturb mode" \
    "system-settings.toggle_do_not_disturb" \
    "${SYSTEM_NEW_SERVERS}"

# --- screenshot (1 test) ---
run_test "screenshot: capture_and_extract" \
    "Take a screenshot and read the text on screen" \
    "screenshot.capture_and_extract" \
    "${SYSTEM_NEW_SERVERS}"

fi  # --new

# ============================================================================
# GROUP 5: Server-prefix drills (the v1 failure pattern)
# ============================================================================
if [[ "${SUITE}" == "--all" || "${SUITE}" == "--drills" ]]; then

start_group "Server-Prefix Drills"

# --- task vs calendar (the 3 v1 failures) ---
run_test "DRILL task.get_overdue (NOT calendar)" \
    "Which of my tasks are overdue?" \
    "task.get_overdue" \
    "${SYSTEM_DRILL_TASK_CAL}"

run_test "DRILL task.daily_briefing (NOT calendar)" \
    "Give me my daily task briefing" \
    "task.daily_briefing" \
    "${SYSTEM_DRILL_TASK_CAL}"

run_test "DRILL calendar.list_events (NOT task)" \
    "What calendar events do I have tomorrow?" \
    "calendar.list_events" \
    "${SYSTEM_DRILL_TASK_CAL}"

run_test "DRILL calendar.create_time_block (NOT task)" \
    "Block 2 hours on my calendar for deep work" \
    "calendar.create_time_block" \
    "${SYSTEM_DRILL_TASK_CAL}"

run_test "DRILL task.update_task (NOT calendar)" \
    "Mark the code review task as complete" \
    "task.update_task" \
    "${SYSTEM_DRILL_TASK_CAL}"

# --- system vs system-settings ---
run_test "DRILL system.get_system_info (NOT system-settings)" \
    "What CPU and OS is this machine?" \
    "system.get_system_info" \
    "${SYSTEM_DRILL_SYS}"

run_test "DRILL system-settings.get_display_settings (NOT system)" \
    "What is my screen brightness set to?" \
    "system-settings.get_display_settings" \
    "${SYSTEM_DRILL_SYS}"

run_test "DRILL system.get_cpu_usage (NOT system-settings)" \
    "What percentage of CPU is being used right now?" \
    "system.get_cpu_usage" \
    "${SYSTEM_DRILL_SYS}"

run_test "DRILL system-settings.set_audio_volume (NOT system)" \
    "Change the volume to 75%" \
    "system-settings.set_audio_volume" \
    "${SYSTEM_DRILL_SYS}"

run_test "DRILL system.kill_process (NOT system-settings)" \
    "Kill process 12345" \
    "system.kill_process" \
    "${SYSTEM_DRILL_SYS}"

# --- screenshot vs system.take_screenshot ---
run_test "DRILL screenshot.capture_and_extract (NOT system.take_screenshot)" \
    "Screenshot my screen and read what it says" \
    "screenshot.capture_and_extract" \
    "${SYSTEM_DRILL_SYS}"

run_test "DRILL screenshot.extract_ui_elements (NOT system)" \
    "Identify the buttons and text fields on screen" \
    "screenshot.extract_ui_elements" \
    "${SYSTEM_DRILL_SYS}"

fi  # --drills

# ============================================================================
# Results
# ============================================================================

echo ""
echo "========================================"
echo "RESULTS"
echo "========================================"
echo "Passed: ${PASS_COUNT} / ${TOTAL_TESTS}"
echo "Failed: ${FAIL_COUNT} / ${TOTAL_TESTS}"
ACCURACY=$(python3 -c "
p, t = ${PASS_COUNT}, ${TOTAL_TESTS}
print(f'{p/t*100:.1f}%' if t > 0 else 'N/A')
" 2>/dev/null || echo "N/A")
echo "Accuracy: ${ACCURACY}"
echo ""

# Exit with non-zero if any test failed
if [ "${FAIL_COUNT}" -gt 0 ]; then
    echo "SOME TESTS FAILED"
    exit 1
else
    echo "ALL TESTS PASSED"
    exit 0
fi
