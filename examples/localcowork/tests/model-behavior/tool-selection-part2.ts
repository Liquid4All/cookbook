/**
 * Tool Selection Tests — Part 2: OCR, Security, Task, Calendar.
 *
 * 33 tests covering ocr, security, task, and calendar servers.
 */

import type { ToolSelectionTest } from './types';

/** OCR & Vision — 8 tests covering the ocr server. */
export const ocrVisionTests: readonly ToolSelectionTest[] = [
  {
    id: 'ts-ocr-001',
    category: 'ocr-vision',
    prompt: 'Extract the text from this screenshot',
    context: ['I took a screenshot at /Users/me/Desktop/screenshot.png'],
    expectedTools: ['ocr.extract_text_from_image'],
    expectedParamKeys: { 'ocr.extract_text_from_image': ['path'] },
    difficulty: 'easy',
  },
  {
    id: 'ts-ocr-002',
    category: 'ocr-vision',
    prompt: 'Read the table in this scanned document',
    context: ['The scanned page is at /Users/me/scan-page3.png'],
    expectedTools: ['ocr.extract_table'],
    expectedParamKeys: { 'ocr.extract_table': ['path'] },
    difficulty: 'easy',
  },
  {
    id: 'ts-ocr-003',
    category: 'ocr-vision',
    prompt: 'OCR the receipt image and extract the total amount and date',
    context: ['Receipt image at /Users/me/receipts/grocery-receipt.jpg'],
    expectedTools: ['ocr.extract_structured_data'],
    difficulty: 'medium',
  },
  {
    id: 'ts-ocr-004',
    category: 'ocr-vision',
    prompt: 'Extract text from every page of this scanned PDF',
    context: ['The scanned PDF is at /Users/me/old-manual.pdf'],
    expectedTools: ['ocr.extract_text_from_pdf'],
    expectedParamKeys: { 'ocr.extract_text_from_pdf': ['path'] },
    difficulty: 'easy',
  },
  {
    id: 'ts-ocr-005',
    category: 'ocr-vision',
    prompt: 'Read the text in this photo of a whiteboard',
    context: ['Photo is at /Users/me/Photos/whiteboard-meeting.jpg'],
    expectedTools: ['ocr.extract_text_from_image'],
    difficulty: 'easy',
  },
  {
    id: 'ts-ocr-006',
    category: 'ocr-vision',
    prompt: 'Extract the vendor name, date, and line items from this invoice scan',
    context: ['Scanned invoice at /Users/me/invoices/scan-001.png'],
    expectedTools: ['ocr.extract_structured_data'],
    difficulty: 'medium',
  },
  {
    id: 'ts-ocr-007',
    category: 'ocr-vision',
    prompt: 'Pull the data table from page 5 of this scanned report',
    context: ['Report PDF at /Users/me/reports/annual-2024.pdf'],
    expectedTools: ['ocr.extract_table'],
    expectedParamKeys: { 'ocr.extract_table': ['path', 'page'] },
    difficulty: 'medium',
  },
  {
    id: 'ts-ocr-008',
    category: 'ocr-vision',
    prompt: 'I have a scan of a business card — extract the name, phone, and email',
    context: ['Business card image at /Users/me/cards/john-doe.jpg'],
    expectedTools: ['ocr.extract_structured_data'],
    difficulty: 'medium',
  },
];

/** Security & Privacy — 10 tests covering the security server. */
export const securityPrivacyTests: readonly ToolSelectionTest[] = [
  {
    id: 'ts-sec-001',
    category: 'security-privacy',
    prompt: 'Scan my Downloads folder for any sensitive personal data',
    expectedTools: ['security.scan_for_pii'],
    expectedParamKeys: { 'security.scan_for_pii': ['path'] },
    difficulty: 'easy',
  },
  {
    id: 'ts-sec-002',
    category: 'security-privacy',
    prompt: 'Check this project for exposed API keys or secrets',
    expectedTools: ['security.scan_for_secrets'],
    difficulty: 'easy',
  },
  {
    id: 'ts-sec-003',
    category: 'security-privacy',
    prompt: 'Encrypt this confidential file before sharing',
    context: ['File at /Users/me/Documents/financials.xlsx'],
    expectedTools: ['security.encrypt_file'],
    expectedParamKeys: { 'security.encrypt_file': ['path'] },
    difficulty: 'easy',
  },
  {
    id: 'ts-sec-004',
    category: 'security-privacy',
    prompt: 'Decrypt the encrypted report file',
    context: ['Encrypted file at /Users/me/Documents/financials.xlsx.enc'],
    expectedTools: ['security.decrypt_file'],
    expectedParamKeys: { 'security.decrypt_file': ['path'] },
    difficulty: 'easy',
  },
  {
    id: 'ts-sec-005',
    category: 'security-privacy',
    prompt: 'Find duplicate files in my Documents folder to free up space',
    expectedTools: ['security.find_duplicates'],
    difficulty: 'easy',
  },
  {
    id: 'ts-sec-006',
    category: 'security-privacy',
    prompt: 'Suggest files I can safely delete to clean up my home directory',
    expectedTools: ['security.propose_cleanup'],
    difficulty: 'medium',
  },
  {
    id: 'ts-sec-007',
    category: 'security-privacy',
    prompt: 'Are there any social security numbers or credit card numbers in my tax folder?',
    expectedTools: ['security.scan_for_pii'],
    difficulty: 'medium',
  },
  {
    id: 'ts-sec-008',
    category: 'security-privacy',
    prompt: 'Check if any .env files in my projects contain hardcoded credentials',
    expectedTools: ['security.scan_for_secrets'],
    difficulty: 'medium',
  },
  {
    id: 'ts-sec-009',
    category: 'security-privacy',
    prompt: 'Lock down the salary spreadsheet with encryption',
    context: ['Spreadsheet at /Users/me/HR/salaries-2024.xlsx'],
    expectedTools: ['security.encrypt_file'],
    difficulty: 'easy',
  },
  {
    id: 'ts-sec-010',
    category: 'security-privacy',
    prompt: 'I have three copies of the same photo — find and show me the duplicates',
    expectedTools: ['security.find_duplicates'],
    difficulty: 'medium',
  },
];

/** Task Management — 8 tests covering the task server. */
export const taskManagementTests: readonly ToolSelectionTest[] = [
  {
    id: 'ts-task-001',
    category: 'task-management',
    prompt: 'Create a new task: review Q4 report by Friday',
    expectedTools: ['task.create_task'],
    difficulty: 'easy',
  },
  {
    id: 'ts-task-002',
    category: 'task-management',
    prompt: 'What tasks are overdue?',
    expectedTools: ['task.get_overdue'],
    difficulty: 'easy',
  },
  {
    id: 'ts-task-003',
    category: 'task-management',
    prompt: 'Show me my daily briefing for today',
    expectedTools: ['task.daily_briefing'],
    difficulty: 'easy',
  },
  {
    id: 'ts-task-004',
    category: 'task-management',
    prompt: 'List all my current tasks',
    expectedTools: ['task.list_tasks'],
    difficulty: 'easy',
  },
  {
    id: 'ts-task-005',
    category: 'task-management',
    prompt: 'Mark the "Prepare slides" task as complete',
    expectedTools: ['task.update_task'],
    difficulty: 'easy',
  },
  {
    id: 'ts-task-006',
    category: 'task-management',
    prompt: 'Add a high-priority task to call the vendor about pricing',
    expectedTools: ['task.create_task'],
    difficulty: 'easy',
  },
  {
    id: 'ts-task-007',
    category: 'task-management',
    prompt: 'Change the due date on my "Submit expenses" task to next Monday',
    expectedTools: ['task.update_task'],
    difficulty: 'medium',
  },
  {
    id: 'ts-task-008',
    category: 'task-management',
    prompt: 'Show me everything I need to do this week sorted by priority',
    expectedTools: ['task.list_tasks'],
    difficulty: 'medium',
  },
];

/** Calendar — 7 tests covering the calendar server. */
export const calendarTests: readonly ToolSelectionTest[] = [
  {
    id: 'ts-cal-001',
    category: 'calendar',
    prompt: 'What meetings do I have today?',
    expectedTools: ['calendar.list_events'],
    difficulty: 'easy',
  },
  {
    id: 'ts-cal-002',
    category: 'calendar',
    prompt: 'Schedule a 1-hour meeting tomorrow at 2pm with the design team',
    expectedTools: ['calendar.create_event'],
    difficulty: 'easy',
  },
  {
    id: 'ts-cal-003',
    category: 'calendar',
    prompt: 'Find a free 30-minute slot this afternoon for a quick sync',
    expectedTools: ['calendar.find_free_slots'],
    difficulty: 'easy',
  },
  {
    id: 'ts-cal-004',
    category: 'calendar',
    prompt: 'Block off 2 hours tomorrow morning for deep work',
    expectedTools: ['calendar.create_time_block'],
    difficulty: 'easy',
  },
  {
    id: 'ts-cal-005',
    category: 'calendar',
    prompt: 'Show me my schedule for next week',
    expectedTools: ['calendar.list_events'],
    difficulty: 'easy',
  },
  {
    id: 'ts-cal-006',
    category: 'calendar',
    prompt: 'When am I free on Thursday for a 1-hour client call?',
    expectedTools: ['calendar.find_free_slots'],
    difficulty: 'medium',
  },
  {
    id: 'ts-cal-007',
    category: 'calendar',
    prompt: 'Create a recurring daily standup at 9:15am',
    expectedTools: ['calendar.create_event'],
    difficulty: 'medium',
  },
];
