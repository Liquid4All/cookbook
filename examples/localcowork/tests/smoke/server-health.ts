/**
 * ─────────────────────────────────────────────────────────────────────────────
 * LocalCowork — Server Health Check
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Attempts to start each implemented MCP server and send an `initialize`
 * JSON-RPC request. If the server responds with a valid capabilities object,
 * the health check passes.
 *
 * This test AUTO-DISCOVERS servers — no manual registration needed.
 *
 * Timeout: 5 seconds per server (these should start fast).
 *
 * Output format (consumed by smoke-test.sh):
 *   PASS server_name — N tools registered
 *   FAIL server_name — reason
 *   SKIP server_name — reason
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { spawn, ChildProcess } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TIMEOUT_MS = 5000;
const PROJECT_ROOT = path.resolve(__dirname, '../..');
const MCP_DIR = path.join(PROJECT_ROOT, 'mcp-servers');

interface HealthResult {
  server: string;
  status: 'PASS' | 'FAIL' | 'SKIP';
  detail: string;
}

/**
 * Detect how to start a server based on its language.
 */
function getStartCommand(serverName: string): { cmd: string; args: string[] } | null {
  const serverDir = path.join(MCP_DIR, serverName);

  // TypeScript server
  if (fs.existsSync(path.join(serverDir, 'src', 'index.ts'))) {
    return { cmd: 'npx', args: ['tsx', path.join(serverDir, 'src', 'index.ts')] };
  }

  // Python server — prefer explicit entry points over module invocation
  if (fs.existsSync(path.join(serverDir, 'src', 'main.py'))) {
    return { cmd: 'python', args: [path.join(serverDir, 'src', 'main.py')] };
  }
  if (fs.existsSync(path.join(serverDir, 'src', 'server.py'))) {
    return { cmd: 'python', args: [path.join(serverDir, 'src', 'server.py')] };
  }

  return null;
}

/**
 * Send a JSON-RPC initialize request and wait for a response.
 */
async function checkHealth(serverName: string): Promise<HealthResult> {
  const startCmd = getStartCommand(serverName);

  if (!startCmd) {
    return { server: serverName, status: 'SKIP', detail: 'no entry point found' };
  }

  return new Promise((resolve) => {
    let proc: ChildProcess | null = null;
    let resolved = false;
    let stdout = '';

    const cleanup = () => {
      if (proc && !proc.killed) {
        proc.kill('SIGTERM');
        // Force kill after 1 second if still alive
        setTimeout(() => {
          if (proc && !proc.killed) proc.kill('SIGKILL');
        }, 1000);
      }
    };

    const done = (result: HealthResult) => {
      if (resolved) return;
      resolved = true;
      cleanup();
      resolve(result);
    };

    // Timeout
    const timer = setTimeout(() => {
      done({ server: serverName, status: 'FAIL', detail: `timeout after ${TIMEOUT_MS}ms` });
    }, TIMEOUT_MS);

    try {
      proc = spawn(startCmd.cmd, startCmd.args, {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: PROJECT_ROOT,
        env: { ...process.env, NODE_ENV: 'test' },
      });

      proc.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();

        // Look for a JSON-RPC response
        try {
          // The response might have newlines; try to parse each line
          const lines = stdout.split('\n');
          for (const line of lines) {
            if (line.trim().startsWith('{')) {
              const response = JSON.parse(line.trim());
              if (response.result?.serverInfo || response.result?.server_info || response.result?.tools) {
                const toolCount = response.result.tools?.length ?? 'unknown';
                clearTimeout(timer);
                done({
                  server: serverName,
                  status: 'PASS',
                  detail: `${toolCount} tools registered`,
                });
              } else if (response.error) {
                clearTimeout(timer);
                done({
                  server: serverName,
                  status: 'FAIL',
                  detail: `initialize error: ${response.error.message}`,
                });
              }
            }
          }
        } catch {
          // Not valid JSON yet — keep accumulating
        }
      });

      proc.stderr?.on('data', (data: Buffer) => {
        // Ignore stderr unless it indicates a crash
        const msg = data.toString();
        if (msg.includes('Error') || msg.includes('FATAL') || msg.includes('Cannot find module')) {
          clearTimeout(timer);
          done({
            server: serverName,
            status: 'FAIL',
            detail: msg.trim().substring(0, 100),
          });
        }
      });

      proc.on('error', (err) => {
        clearTimeout(timer);
        done({ server: serverName, status: 'FAIL', detail: `spawn error: ${err.message}` });
      });

      proc.on('exit', (code) => {
        clearTimeout(timer);
        if (!resolved) {
          done({
            server: serverName,
            status: 'FAIL',
            detail: `exited with code ${code} before responding`,
          });
        }
      });

      // Send the initialize request via stdin
      const initRequest = JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        method: 'initialize',
        params: {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: { name: 'smoke-test', version: '1.0.0' },
        },
      });

      proc.stdin?.write(initRequest + '\n');

    } catch (err: unknown) {
      clearTimeout(timer);
      const message = err instanceof Error ? err.message : String(err);
      done({ server: serverName, status: 'FAIL', detail: `exception: ${message}` });
    }
  });
}

// ── Main ────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const requestedServers = process.argv.slice(2);

  // Find all server directories
  const allDirs = fs.readdirSync(MCP_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory() && d.name !== '_shared')
    .map(d => d.name);

  const servers = requestedServers.length > 0
    ? allDirs.filter(s => requestedServers.includes(s))
    : allDirs;

  // Run health checks sequentially (to avoid port/resource conflicts)
  for (const server of servers) {
    const result = await checkHealth(server);
    console.log(`${result.status} ${result.server} ${result.detail}`);
  }
}

main().catch(console.error);
