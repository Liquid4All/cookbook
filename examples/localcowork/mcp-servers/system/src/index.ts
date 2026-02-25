/**
 * System MCP Server — Entry Point
 *
 * Registers all system tools and starts the JSON-RPC listener.
 * This server provides OS-level operations via a bridge abstraction.
 * In production, the bridge routes to Tauri IPC commands;
 * in dev/test, it uses Node.js APIs or mock stubs.
 *
 * Tools (10):
 *   system.get_system_info  — get hardware and OS info
 *   system.open_application — open an application by name (confirm)
 *   system.take_screenshot  — capture a screenshot
 *   system.list_processes   — list running processes
 *   system.open_file_with   — open a file with a specific app (confirm)
 *   system.get_memory_usage — get RAM and swap usage
 *   system.get_disk_usage   — get disk volume capacity and free space
 *   system.get_cpu_usage    — get CPU utilization per core
 *   system.get_network_info — get network interfaces and IPs
 *   system.kill_process     — terminate a process by PID (confirm)
 */

import { MCPServer } from '../../_shared/ts/mcp-base';
import { getSystemInfo } from './tools/get_system_info';
import { openApplication } from './tools/open_application';
import { takeScreenshot } from './tools/take_screenshot';
import { listProcesses } from './tools/list_processes';
import { openFileWith } from './tools/open_file_with';
import { getMemoryUsage } from './tools/get_memory_usage';
import { getDiskUsage } from './tools/get_disk_usage';
import { getCpuUsage } from './tools/get_cpu_usage';
import { getNetworkInfo } from './tools/get_network_info';
import { killProcess } from './tools/kill_process';

// ─── Server Setup ───────────────────────────────────────────────────────────

const server = new MCPServer({
  name: 'system',
  version: '0.2.0',
  tools: [
    getSystemInfo, openApplication, takeScreenshot, listProcesses, openFileWith,
    getMemoryUsage, getDiskUsage, getCpuUsage, getNetworkInfo, killProcess,
  ],
});

// ─── Graceful Shutdown ──────────────────────────────────────────────────────

process.on('SIGINT', () => {
  process.exit(0);
});

process.on('SIGTERM', () => {
  process.exit(0);
});

// ─── Start ──────────────────────────────────────────────────────────────────

server.start();
