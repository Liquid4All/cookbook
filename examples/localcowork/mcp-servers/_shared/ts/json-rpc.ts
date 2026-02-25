/**
 * JSON-RPC 2.0 Transport Utilities — TypeScript
 *
 * Low-level JSON-RPC message handling for MCP server communication.
 * Used by mcp-base.ts; typically not imported directly by tool implementations.
 */

// ─── JSON-RPC Types ─────────────────────────────────────────────────────────

export interface JsonRpcMessage {
  jsonrpc: '2.0';
}

export interface JsonRpcRequest extends JsonRpcMessage {
  id: string | number;
  method: string;
  params?: Record<string, unknown>;
}

export interface JsonRpcNotification extends JsonRpcMessage {
  method: string;
  params?: Record<string, unknown>;
}

export interface JsonRpcSuccessResponse extends JsonRpcMessage {
  id: string | number;
  result: unknown;
}

export interface JsonRpcErrorResponse extends JsonRpcMessage {
  id: string | number;
  error: {
    code: number;
    message: string;
    data?: unknown;
  };
}

export type JsonRpcResponse = JsonRpcSuccessResponse | JsonRpcErrorResponse;

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Create a success response */
export function successResponse(id: string | number, result: unknown): JsonRpcSuccessResponse {
  return { jsonrpc: '2.0', id, result };
}

/** Create an error response */
export function errorResponse(
  id: string | number,
  code: number,
  message: string,
  data?: unknown,
): JsonRpcErrorResponse {
  return { jsonrpc: '2.0', id, error: { code, message, data } };
}

/** Create a notification (no response expected) */
export function notification(method: string, params?: Record<string, unknown>): JsonRpcNotification {
  return { jsonrpc: '2.0', method, params };
}

/** Validate that a parsed message is a valid JSON-RPC request */
export function isValidRequest(msg: unknown): msg is JsonRpcRequest {
  if (typeof msg !== 'object' || msg === null) return false;
  const obj = msg as Record<string, unknown>;
  return (
    obj.jsonrpc === '2.0' &&
    typeof obj.method === 'string' &&
    (typeof obj.id === 'string' || typeof obj.id === 'number')
  );
}

/** Parse a raw string into a JsonRpcRequest, throwing on invalid input */
export function parseRequest(raw: string): JsonRpcRequest {
  const parsed = JSON.parse(raw);
  if (!isValidRequest(parsed)) {
    throw new Error('Invalid JSON-RPC request');
  }
  return parsed;
}
