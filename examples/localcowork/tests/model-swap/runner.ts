/**
 * Model Swap Runner — WS-6C
 *
 * Loads test definitions from tests/model-behavior/definitions/,
 * sends each prompt to a model via OpenAI-compatible chat completions API,
 * parses the response for tool calls, and compares against expected results.
 *
 * When no live model endpoint is available (LOCALCOWORK_MODEL_ENDPOINT not set),
 * runs structural validation only.
 */

import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

export interface ModelSwapConfig {
  readonly modelName: string;
  readonly endpoint: string;
  readonly apiKey?: string;
  readonly temperature?: number;
  readonly systemPrompt: string;
}

export interface ToolSelectionTest {
  readonly id: string;
  readonly prompt: string;
  readonly expectedTool: string;
  readonly expectedParams: Record<string, unknown>;
  readonly tags: readonly string[];
}

export interface MultiStepTest {
  readonly id: string;
  readonly name: string;
  readonly prompt: string;
  readonly expectedChain: readonly string[];
  readonly tags: readonly string[];
}

export interface EdgeCaseTest {
  readonly id: string;
  readonly name: string;
  readonly prompt: string;
  readonly expectedBehavior: string;
  readonly expectedTool: string | null;
  readonly reason: string;
  readonly tags: readonly string[];
}

export interface TestSuite<T> {
  readonly suite: string;
  readonly description: string;
  readonly version: string;
  readonly tests: readonly T[];
}

export interface TestCaseResult {
  readonly id: string;
  readonly prompt: string;
  readonly expected: string | null;
  readonly actual: string | null;
  readonly passed: boolean;
  readonly details?: string;
}

export interface SuiteResult {
  readonly total: number;
  readonly passed: number;
  readonly accuracy: number;
  readonly results: readonly TestCaseResult[];
}

export interface SwapResult {
  readonly modelName: string;
  readonly timestamp: string;
  readonly toolSelection: SuiteResult;
  readonly multiStep: SuiteResult;
  readonly edgeCases: SuiteResult;
  readonly overall: { readonly total: number; readonly passed: number; readonly accuracy: number };
}

// ---------------------------------------------------------------------------
// OpenAI-compatible API types (minimal subset)
// ---------------------------------------------------------------------------

interface ChatMessage {
  readonly role: "system" | "user" | "assistant";
  readonly content: string;
}

interface ToolCallFunction {
  readonly name: string;
  readonly arguments: string;
}

interface ToolCall {
  readonly id: string;
  readonly type: "function";
  readonly function: ToolCallFunction;
}

interface ChatChoice {
  readonly message: {
    readonly role: string;
    readonly content: string | null;
    readonly tool_calls?: readonly ToolCall[];
  };
}

interface ChatCompletionResponse {
  readonly choices: readonly ChatChoice[];
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

const currentDir = typeof __dirname !== "undefined"
  ? __dirname
  : dirname(fileURLToPath(import.meta.url));

function definitionsDir(): string {
  return resolve(currentDir, "..", "model-behavior", "definitions");
}

// ---------------------------------------------------------------------------
// Test definition loaders
// ---------------------------------------------------------------------------

export function loadToolSelectionTests(): TestSuite<ToolSelectionTest> {
  const filePath = resolve(definitionsDir(), "tool-selection.json");
  const raw = readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as TestSuite<ToolSelectionTest>;
}

export function loadMultiStepTests(): TestSuite<MultiStepTest> {
  const filePath = resolve(definitionsDir(), "multi-step.json");
  const raw = readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as TestSuite<MultiStepTest>;
}

export function loadEdgeCaseTests(): TestSuite<EdgeCaseTest> {
  const filePath = resolve(definitionsDir(), "edge-cases.json");
  const raw = readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as TestSuite<EdgeCaseTest>;
}

// ---------------------------------------------------------------------------
// Model API caller
// ---------------------------------------------------------------------------

async function callModel(
  config: ModelSwapConfig,
  userPrompt: string,
): Promise<ChatCompletionResponse> {
  const url = config.endpoint.replace(/\/$/, "") + "/chat/completions";

  const messages: readonly ChatMessage[] = [
    { role: "system", content: config.systemPrompt },
    { role: "user", content: userPrompt },
  ];

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (config.apiKey) {
    headers["Authorization"] = "Bearer " + config.apiKey;
  }

  const body = JSON.stringify({
    model: config.modelName,
    messages,
    temperature: config.temperature ?? 0.1,
    tools: buildToolDefinitions(),
  });

  const response = await fetch(url, { method: "POST", headers, body });
  if (!response.ok) {
    const text = await response.text();
    throw new Error("Model API error " + String(response.status) + ": " + text);
  }
  return (await response.json()) as ChatCompletionResponse;
}

// ---------------------------------------------------------------------------
// Minimal tool definitions for the OpenAI tools parameter
// ---------------------------------------------------------------------------

function buildToolDefinitions(): readonly Record<string, unknown>[] {
  const toolNames = [
    "filesystem.list_dir", "filesystem.read_file", "filesystem.write_file",
    "filesystem.move_file", "filesystem.copy_file", "filesystem.delete_file",
    "filesystem.search_files", "filesystem.get_metadata", "filesystem.watch_folder",
    "document.extract_text", "document.convert_format", "document.diff_documents",
    "document.create_pdf", "document.fill_pdf_form", "document.merge_pdfs",
    "document.create_docx", "document.read_spreadsheet",
    "ocr.extract_text_from_image", "ocr.extract_text_from_pdf",
    "ocr.extract_structured_data", "ocr.extract_table",
    "data.write_csv", "data.write_sqlite", "data.query_sqlite",
    "data.deduplicate_records", "data.summarize_anomalies",
    "audit.get_tool_log", "audit.get_session_summary",
    "audit.generate_audit_report", "audit.export_audit_pdf",
    "knowledge.index_folder", "knowledge.search_documents",
    "knowledge.ask_about_files", "knowledge.update_index",
    "knowledge.get_related_chunks",
    "security.scan_for_pii", "security.scan_for_secrets",
    "security.find_duplicates", "security.propose_cleanup",
    "security.encrypt_file", "security.decrypt_file",
    "task.create_task", "task.list_tasks", "task.update_task",
    "task.get_overdue", "task.daily_briefing",
    "calendar.list_events", "calendar.create_event",
    "calendar.find_free_slots", "calendar.create_time_block",
    "email.draft_email", "email.list_drafts", "email.search_emails",
    "email.summarize_thread", "email.send_draft",
    "meeting.transcribe_audio", "meeting.extract_action_items",
    "meeting.extract_commitments", "meeting.generate_minutes",
    "clipboard.get_clipboard", "clipboard.set_clipboard",
    "clipboard.clipboard_history",
    "system.get_system_info", "system.open_application",
    "system.take_screenshot", "system.list_processes",
    "system.open_file_with",
  ];

  return toolNames.map((name) => ({
    type: "function",
    function: {
      name: name.replace(".", "_"),
      description: "MCP tool: " + name,
      parameters: { type: "object", properties: {}, additionalProperties: true },
    },
    _mcpName: name,
  }));
}

// ---------------------------------------------------------------------------
// Extract first tool call name from a model response
// ---------------------------------------------------------------------------

function extractToolName(response: ChatCompletionResponse): string | null {
  const choice = response.choices[0];
  if (!choice) return null;

  const toolCalls = choice.message.tool_calls;
  if (toolCalls && toolCalls.length > 0) {
    const fnName = toolCalls[0].function.name;
    // Convert back from OpenAI function name (underscore) to MCP dotted name
    return fnName.replace("_", ".");
  }
  return null;
}

function extractAllToolNames(response: ChatCompletionResponse): readonly string[] {
  const choice = response.choices[0];
  if (!choice) return [];

  const toolCalls = choice.message.tool_calls;
  if (!toolCalls) return [];

  return toolCalls.map((tc) => tc.function.name.replace("_", "."));
}

// ---------------------------------------------------------------------------
// Suite runners (live model)
// ---------------------------------------------------------------------------

async function runToolSelectionSuite(
  config: ModelSwapConfig,
  suite: TestSuite<ToolSelectionTest>,
): Promise<SuiteResult> {
  const results: TestCaseResult[] = [];

  for (const test of suite.tests) {
    try {
      const response = await callModel(config, test.prompt);
      const actual = extractToolName(response);
      const passed = actual === test.expectedTool;
      results.push({
        id: test.id,
        prompt: test.prompt,
        expected: test.expectedTool,
        actual,
        passed,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      results.push({
        id: test.id,
        prompt: test.prompt,
        expected: test.expectedTool,
        actual: null,
        passed: false,
        details: "Error: " + message,
      });
    }
  }

  const passed = results.filter((r) => r.passed).length;
  return {
    total: results.length,
    passed,
    accuracy: results.length > 0 ? passed / results.length : 0,
    results,
  };
}

async function runMultiStepSuite(
  config: ModelSwapConfig,
  suite: TestSuite<MultiStepTest>,
): Promise<SuiteResult> {
  const results: TestCaseResult[] = [];

  for (const test of suite.tests) {
    try {
      const response = await callModel(config, test.prompt);
      const actualChain = extractAllToolNames(response);
      // Check if the first tool in the expected chain matches the first actual tool
      // Full chain validation requires multi-turn; here we validate the first step
      const firstExpected = test.expectedChain[0] ?? null;
      const firstActual = actualChain[0] ?? null;
      const passed = firstExpected === firstActual;
      results.push({
        id: test.id,
        prompt: test.prompt,
        expected: firstExpected,
        actual: firstActual,
        passed,
        details: "Expected chain: [" + test.expectedChain.join(", ") + "] Got first: " + String(firstActual),
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      results.push({
        id: test.id,
        prompt: test.prompt,
        expected: test.expectedChain[0] ?? null,
        actual: null,
        passed: false,
        details: "Error: " + message,
      });
    }
  }

  const passed = results.filter((r) => r.passed).length;
  return {
    total: results.length,
    passed,
    accuracy: results.length > 0 ? passed / results.length : 0,
    results,
  };
}

async function runEdgeCaseSuite(
  config: ModelSwapConfig,
  suite: TestSuite<EdgeCaseTest>,
): Promise<SuiteResult> {
  const results: TestCaseResult[] = [];

  for (const test of suite.tests) {
    try {
      const response = await callModel(config, test.prompt);
      const actual = extractToolName(response);

      let passed = false;
      if (test.expectedBehavior === "tool_call") {
        passed = actual === test.expectedTool;
      } else if (
        test.expectedBehavior === "ask_clarification" ||
        test.expectedBehavior === "explain_limitation" ||
        test.expectedBehavior === "conversational_response"
      ) {
        // For non-tool responses, the model should NOT call a tool
        passed = actual === null;
      }

      results.push({
        id: test.id,
        prompt: test.prompt,
        expected: test.expectedTool,
        actual,
        passed,
        details: "Behavior: " + test.expectedBehavior + " | Reason: " + test.reason,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      results.push({
        id: test.id,
        prompt: test.prompt,
        expected: test.expectedTool,
        actual: null,
        passed: false,
        details: "Error: " + message,
      });
    }
  }

  const passed = results.filter((r) => r.passed).length;
  return {
    total: results.length,
    passed,
    accuracy: results.length > 0 ? passed / results.length : 0,
    results,
  };
}

// ---------------------------------------------------------------------------
// Main runner
// ---------------------------------------------------------------------------

export async function runModelSwap(config: ModelSwapConfig): Promise<SwapResult> {
  const toolSelectionSuite = loadToolSelectionTests();
  const multiStepSuite = loadMultiStepTests();
  const edgeCaseSuite = loadEdgeCaseTests();

  const toolSelection = await runToolSelectionSuite(config, toolSelectionSuite);
  const multiStep = await runMultiStepSuite(config, multiStepSuite);
  const edgeCases = await runEdgeCaseSuite(config, edgeCaseSuite);

  const totalAll = toolSelection.total + multiStep.total + edgeCases.total;
  const passedAll = toolSelection.passed + multiStep.passed + edgeCases.passed;

  return {
    modelName: config.modelName,
    timestamp: new Date().toISOString(),
    toolSelection,
    multiStep,
    edgeCases,
    overall: {
      total: totalAll,
      passed: passedAll,
      accuracy: totalAll > 0 ? passedAll / totalAll : 0,
    },
  };
}

// ---------------------------------------------------------------------------
// Structural validation (no live model needed)
// ---------------------------------------------------------------------------

export function validateTestDefinitions(): {
  readonly valid: boolean;
  readonly errors: readonly string[];
  readonly counts: {
    readonly toolSelection: number;
    readonly multiStep: number;
    readonly edgeCases: number;
    readonly total: number;
  };
} {
  const errors: string[] = [];
  let tsCount = 0;
  let msCount = 0;
  let ecCount = 0;

  // Validate tool-selection
  try {
    const ts = loadToolSelectionTests();
    tsCount = ts.tests.length;
    if (tsCount === 0) errors.push("tool-selection suite has no tests");
    for (const test of ts.tests) {
      if (!test.id) errors.push("tool-selection test missing id");
      if (!test.prompt) errors.push("tool-selection test " + test.id + " missing prompt");
      if (!test.expectedTool) errors.push("tool-selection test " + test.id + " missing expectedTool");
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    errors.push("Failed to load tool-selection.json: " + msg);
  }

  // Validate multi-step
  try {
    const ms = loadMultiStepTests();
    msCount = ms.tests.length;
    if (msCount === 0) errors.push("multi-step suite has no tests");
    for (const test of ms.tests) {
      if (!test.id) errors.push("multi-step test missing id");
      if (!test.prompt) errors.push("multi-step test " + test.id + " missing prompt");
      if (!test.expectedChain || test.expectedChain.length === 0) {
        errors.push("multi-step test " + test.id + " missing expectedChain");
      }
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    errors.push("Failed to load multi-step.json: " + msg);
  }

  // Validate edge-cases
  try {
    const ec = loadEdgeCaseTests();
    ecCount = ec.tests.length;
    if (ecCount === 0) errors.push("edge-cases suite has no tests");
    for (const test of ec.tests) {
      if (!test.id) errors.push("edge-cases test missing id");
      if (test.prompt === undefined) errors.push("edge-cases test " + test.id + " missing prompt");
      if (!test.expectedBehavior) {
        errors.push("edge-cases test " + test.id + " missing expectedBehavior");
      }
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    errors.push("Failed to load edge-cases.json: " + msg);
  }

  return {
    valid: errors.length === 0,
    errors,
    counts: {
      toolSelection: tsCount,
      multiStep: msCount,
      edgeCases: ecCount,
      total: tsCount + msCount + ecCount,
    },
  };
}
