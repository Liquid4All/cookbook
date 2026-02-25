/**
 * Clipboard MCP Server -- Entry Point
 *
 * Registers all clipboard tools and starts the JSON-RPC listener.
 * This server provides OS clipboard access via a Tauri bridge
 * abstraction (mock bridge used in standalone/test mode).
 *
 * Tools (3):
 *   clipboard.get_clipboard     -- read clipboard contents
 *   clipboard.set_clipboard     -- write to clipboard
 *   clipboard.clipboard_history -- recent clipboard entries
 */

import { MCPServer } from '../../_shared/ts/mcp-base';
import { getClipboard } from './tools/get_clipboard';
import { setClipboard } from './tools/set_clipboard';
import { clipboardHistory } from './tools/clipboard_history';

// ── Server Setup ─────────────────────────────────────────────────────────────

const server = new MCPServer({
  name: 'clipboard',
  version: '0.1.0',
  tools: [getClipboard, setClipboard, clipboardHistory],
});

// ── Graceful Shutdown ────────────────────────────────────────────────────────

process.on('SIGINT', () => {
  process.exit(0);
});

process.on('SIGTERM', () => {
  process.exit(0);
});

// ── Start ────────────────────────────────────────────────────────────────────

server.start();
