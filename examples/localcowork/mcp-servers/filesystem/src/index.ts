/**
 * Filesystem MCP Server — Entry Point
 *
 * Registers all filesystem tools and starts the JSON-RPC listener.
 * This server provides sandboxed file operations for LocalCowork.
 *
 * Tools (9):
 *   filesystem.list_dir       — list directory contents
 *   filesystem.read_file      — read file content
 *   filesystem.write_file     — write content to file (confirm)
 *   filesystem.move_file      — move/rename file (confirm, undo)
 *   filesystem.copy_file      — copy file (confirm)
 *   filesystem.delete_file    — delete to trash (confirm, undo)
 *   filesystem.search_files   — search by pattern
 *   filesystem.get_metadata   — file metadata
 *   filesystem.watch_folder   — directory watcher (confirm first time)
 */

import os from 'os';
import path from 'path';

import { MCPServer } from '../../_shared/ts/mcp-base';
import { initSandbox } from '../../_shared/ts/validation';
import { listDir } from './tools/list_dir';
import { readFile } from './tools/read_file';
import { writeFile } from './tools/write_file';
import { moveFile } from './tools/move_file';
import { copyFile } from './tools/copy_file';
import { deleteFile } from './tools/delete_file';
import { searchFiles } from './tools/search_files';
import { getMetadata } from './tools/get_metadata';
import { watchFolder, closeAllWatchers } from './tools/watch_folder';

// ─── Sandbox Initialization ─────────────────────────────────────────────────
// Allowed paths come from the Agent Core via environment variable.
// Default to home directory for development.

const allowedPaths = process.env.LOCALCOWORK_ALLOWED_PATHS
  ? process.env.LOCALCOWORK_ALLOWED_PATHS.split(path.delimiter)
  : [os.homedir()];

initSandbox(allowedPaths);

// ─── Server Setup ───────────────────────────────────────────────────────────

const server = new MCPServer({
  name: 'filesystem',
  version: '1.0.0',
  tools: [
    listDir,
    readFile,
    writeFile,
    moveFile,
    copyFile,
    deleteFile,
    searchFiles,
    getMetadata,
    watchFolder,
  ],
});

// ─── Graceful Shutdown ──────────────────────────────────────────────────────

process.on('SIGINT', () => {
  closeAllWatchers();
  process.exit(0);
});

process.on('SIGTERM', () => {
  closeAllWatchers();
  process.exit(0);
});

// ─── Start ──────────────────────────────────────────────────────────────────

server.start();
