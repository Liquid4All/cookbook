/**
 * MCP Server Base Classes — TypeScript
 *
 * Shared foundation for all TypeScript MCP servers in LocalCowork.
 * Implements JSON-RPC 2.0 over stdio transport and tool registration.
 *
 * Usage:
 *   import { MCPServer, MCPTool, MCPResult, MCPError } from '../../_shared/ts/mcp-base';
 */

import { ZodSchema } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';

// ─── Types ──────────────────────────────────────────────────────────────────

/** Result returned by a tool execution */
export interface MCPResult<T = unknown> {
  success: boolean;
  data: T;
}

/** Structured error for MCP tool failures */
export class MCPError extends Error {
  code: number;

  constructor(code: number, message: string) {
    super(message);
    this.code = code;
    this.name = 'MCPError';
  }
}

/** Standard MCP error codes */
export const ErrorCodes = {
  PARSE_ERROR: -32700,
  INVALID_REQUEST: -32600,
  METHOD_NOT_FOUND: -32601,
  INVALID_PARAMS: -32602,
  INTERNAL_ERROR: -32603,
  // Custom codes for LocalCowork
  SANDBOX_VIOLATION: -32001,
  CONFIRMATION_REQUIRED: -32002,
  FILE_NOT_FOUND: -32003,
  PERMISSION_DENIED: -32004,
} as const;

/** Tool definition interface — every tool implements this */
export interface MCPTool<TParams = unknown> {
  /** Fully qualified tool name: server.tool_name */
  name: string;

  /** Human-readable description for the LLM */
  description: string;

  /** Zod schema for parameter validation */
  paramsSchema: ZodSchema<TParams>;

  /** Whether the Agent Core must get user confirmation before executing */
  confirmationRequired: boolean;

  /** Whether this action can be undone via the undo stack */
  undoSupported: boolean;

  /** Execute the tool with validated params */
  execute(params: TParams): Promise<MCPResult>;
}

// ─── JSON-RPC Types ─────────────────────────────────────────────────────────

interface JsonRpcRequest {
  jsonrpc: '2.0';
  id: string | number;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: '2.0';
  id: string | number;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

// ─── MCPServer ──────────────────────────────────────────────────────────────

export interface MCPServerConfig {
  name: string;
  version: string;
  tools: MCPTool[];
}

/**
 * Base MCP Server class.
 *
 * Registers tools, handles JSON-RPC over stdio, validates params,
 * and dispatches tool calls.
 *
 * Usage:
 *   const server = new MCPServer({ name: 'filesystem', version: '1.0.0', tools: [...] });
 *   server.start();
 */
export class MCPServer {
  private name: string;
  private version: string;
  private tools: Map<string, MCPTool>;

  constructor(config: MCPServerConfig) {
    this.name = config.name;
    this.version = config.version;
    this.tools = new Map();

    for (const tool of config.tools) {
      this.tools.set(tool.name, tool);
    }
  }

  /** Start the JSON-RPC listener on stdio */
  start(): void {
    process.stdin.setEncoding('utf-8');

    let buffer = '';

    process.stdin.on('data', (chunk: string) => {
      buffer += chunk;

      // Try to parse complete JSON-RPC messages (newline-delimited)
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.length === 0) continue;

        try {
          const request = JSON.parse(trimmed) as JsonRpcRequest;
          this.handleRequest(request).then((response) => {
            if (response) {
              process.stdout.write(JSON.stringify(response) + '\n');
            }
          });
        } catch {
          const error: JsonRpcResponse = {
            jsonrpc: '2.0',
            id: 0,
            error: { code: ErrorCodes.PARSE_ERROR, message: 'Invalid JSON' },
          };
          process.stdout.write(JSON.stringify(error) + '\n');
        }
      }
    });

    process.stdin.on('end', () => {
      process.exit(0);
    });

    // No proactive init message — the Rust client sends an `initialize`
    // request and we respond to it via handleRequest().
  }

  /**
   * Build the tool definitions array for the initialize/tools-list responses.
   *
   * Converts each registered tool's zod schema to JSON Schema using the
   * zod-to-json-schema library, producing full property/required/description
   * metadata that the LLM needs to call tools correctly.
   */
  private buildToolDefinitions(): Record<string, unknown>[] {
    return Array.from(this.tools.values()).map((tool) => ({
      name: tool.name,
      description: tool.description,
      inputSchema: zodToJsonSchema(tool.paramsSchema, {
        target: 'openApi3',
        $refStrategy: 'none',
      }),
      metadata: {
        confirmationRequired: tool.confirmationRequired,
        undoSupported: tool.undoSupported,
      },
    }));
  }

  /** Handle an incoming JSON-RPC request */
  private async handleRequest(request: JsonRpcRequest): Promise<JsonRpcResponse | null> {
    // MCP initialization handshake — Rust client sends this on server startup
    if (request.method === 'initialize') {
      return {
        jsonrpc: '2.0',
        id: request.id,
        result: {
          serverInfo: { name: this.name, version: this.version },
          tools: this.buildToolDefinitions(),
        },
      };
    }

    if (request.method === 'tools/call') {
      return this.handleToolCall(request);
    }

    if (request.method === 'tools/list') {
      return this.handleToolList(request);
    }

    if (request.method === 'ping') {
      return { jsonrpc: '2.0', id: request.id, result: { status: 'ok' } };
    }

    return {
      jsonrpc: '2.0',
      id: request.id,
      error: { code: ErrorCodes.METHOD_NOT_FOUND, message: `Unknown method: ${request.method}` },
    };
  }

  /** Handle a tools/call request */
  private async handleToolCall(request: JsonRpcRequest): Promise<JsonRpcResponse> {
    const { name, arguments: args } = request.params as { name: string; arguments: unknown };

    const tool = this.tools.get(name);
    if (!tool) {
      return {
        jsonrpc: '2.0',
        id: request.id,
        error: { code: ErrorCodes.METHOD_NOT_FOUND, message: `Unknown tool: ${name}` },
      };
    }

    // Validate params
    const parseResult = tool.paramsSchema.safeParse(args);
    if (!parseResult.success) {
      return {
        jsonrpc: '2.0',
        id: request.id,
        error: {
          code: ErrorCodes.INVALID_PARAMS,
          message: `Invalid parameters: ${parseResult.error.message}`,
        },
      };
    }

    // Execute tool
    try {
      const result = await tool.execute(parseResult.data);
      return {
        jsonrpc: '2.0',
        id: request.id,
        result: {
          content: [{ type: 'text', text: JSON.stringify(result.data) }],
        },
      };
    } catch (err) {
      if (err instanceof MCPError) {
        return {
          jsonrpc: '2.0',
          id: request.id,
          error: { code: err.code, message: err.message },
        };
      }
      return {
        jsonrpc: '2.0',
        id: request.id,
        error: {
          code: ErrorCodes.INTERNAL_ERROR,
          message: `Internal error: ${err instanceof Error ? err.message : 'Unknown'}`,
        },
      };
    }
  }

  /** Handle a tools/list request */
  private handleToolList(request: JsonRpcRequest): JsonRpcResponse {
    return { jsonrpc: '2.0', id: request.id, result: { tools: this.buildToolDefinitions() } };
  }
}
