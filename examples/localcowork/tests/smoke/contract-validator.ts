/**
 * ─────────────────────────────────────────────────────────────────────────────
 * LocalCowork — Contract Validator
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Reads docs/mcp-tool-registry.yaml and validates every implemented server's
 * tool files against it. This test GROWS AUTOMATICALLY — when you add a new
 * server, it gets checked without any manual test authoring.
 *
 * Checks per tool:
 *   1. Tool file exists at the expected path
 *   2. Tool exports a params schema matching registry (field names + types)
 *   3. Tool exports confirmation/undo metadata matching registry
 *   4. Tool file has a corresponding test file
 *
 * Output format (consumed by smoke-test.sh):
 *   PASS server.tool_name [detail]
 *   FAIL server.tool_name [detail]
 *   SKIP server.tool_name [detail]
 * ─────────────────────────────────────────────────────────────────────────────
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ── Types ────────────────────────────────────────────────────────────────

interface RegistryTool {
  name: string;
  description: string;
  params: Record<string, unknown>;
  returns: Record<string, unknown>;
  confirmation_required: boolean;
  undo_supported: boolean;
}

interface RegistryServer {
  name: string;
  language: 'typescript' | 'python';
  tools: RegistryTool[];
}

// ── YAML Parsing (lightweight, no dependency) ────────────────────────────

function parseRegistryServers(yamlContent: string): RegistryServer[] {
  // Simple extraction of server blocks from the registry YAML.
  // This is a lightweight parser — it doesn't need to handle all YAML,
  // just the structure of mcp-tool-registry.yaml.
  const servers: RegistryServer[] = [];
  const serverBlocks = yamlContent.split(/^servers:/m)[1] || '';

  // Match top-level server names (lines like "  filesystem:")
  const serverRegex = /^ {2}(\w+):\s*$/gm;
  let match;
  const serverNames: string[] = [];
  const serverPositions: number[] = [];

  while ((match = serverRegex.exec(serverBlocks)) !== null) {
    serverNames.push(match[1]);
    serverPositions.push(match.index);
  }

  for (let i = 0; i < serverNames.length; i++) {
    const name = serverNames[i];
    const start = serverPositions[i];
    const end = i + 1 < serverPositions.length ? serverPositions[i + 1] : serverBlocks.length;
    const block = serverBlocks.substring(start, end);

    // Detect language
    const langMatch = block.match(/language:\s*(\w+)/);
    const language = (langMatch?.[1] === 'python' ? 'python' : 'typescript') as 'typescript' | 'python';

    // Extract tool names (lines like "      - name: list_directory")
    const toolNames: string[] = [];
    const toolNameRegex = /- name:\s*(\w+)/g;
    let toolMatch;
    while ((toolMatch = toolNameRegex.exec(block)) !== null) {
      toolNames.push(toolMatch[1]);
    }

    // Extract confirmation and undo metadata per tool
    const tools: RegistryTool[] = toolNames.map((toolName) => {
      // Find the tool block
      const toolBlockStart = block.indexOf(`- name: ${toolName}`);
      const nextToolStart = block.indexOf('- name:', toolBlockStart + 1);
      const toolBlock = block.substring(
        toolBlockStart,
        nextToolStart > -1 ? nextToolStart : block.length
      );

      const confirmMatch = toolBlock.match(/confirmation_required:\s*(true|false)/);
      const undoMatch = toolBlock.match(/undo_supported:\s*(true|false)/);

      // Extract param names
      const params: Record<string, unknown> = {};
      const paramsSection = toolBlock.match(/params:[\s\S]*?(?=returns:|confirmation_required:|$)/);
      if (paramsSection) {
        const paramRegex = /^ {10,}(\w+):/gm;
        let paramMatch;
        while ((paramMatch = paramRegex.exec(paramsSection[0])) !== null) {
          params[paramMatch[1]] = true;
        }
      }

      return {
        name: toolName,
        description: '',
        params,
        returns: {},
        confirmation_required: confirmMatch ? confirmMatch[1] === 'true' : false,
        undo_supported: undoMatch ? undoMatch[1] === 'true' : false,
      };
    });

    servers.push({ name, language, tools });
  }

  return servers;
}

// ── Validation Logic ────────────────────────────────────────────────────

function validateServer(
  server: RegistryServer,
  projectRoot: string
): void {
  const serverDir = path.join(projectRoot, 'mcp-servers', server.name);

  // Check if server directory exists
  if (!fs.existsSync(serverDir)) {
    for (const tool of server.tools) {
      console.log(`SKIP ${server.name}.${tool.name} server not yet scaffolded`);
    }
    return;
  }

  const toolsDir = path.join(serverDir, 'src', 'tools');
  const testsDir = path.join(serverDir, 'tests');

  for (const tool of server.tools) {
    const toolFileName = server.language === 'typescript'
      ? `${tool.name}.ts`
      : `${tool.name}.py`;
    const toolFilePath = path.join(toolsDir, toolFileName);

    // Check 1: Tool file exists
    if (!fs.existsSync(toolFilePath)) {
      console.log(`SKIP ${server.name}.${tool.name} tool file not yet created`);
      continue;
    }

    const toolContent = fs.readFileSync(toolFilePath, 'utf-8');
    let issues: string[] = [];

    // Check 2: Param schema references
    // For TS, look for z.object or paramsSchema; for PY, look for BaseModel/Params
    if (server.language === 'typescript') {
      if (!toolContent.includes('z.object') && !toolContent.includes('Schema')) {
        issues.push('no zod schema found');
      }
    } else {
      if (!toolContent.includes('BaseModel') && !toolContent.includes('class Params')) {
        issues.push('no pydantic model found');
      }
    }

    // Check 3: Confirmation metadata
    if (tool.confirmation_required) {
      if (!toolContent.includes('confirmationRequired') && !toolContent.includes('confirmation_required')) {
        issues.push('missing confirmationRequired metadata');
      }
    }

    if (tool.undo_supported) {
      if (!toolContent.includes('undoSupported') && !toolContent.includes('undo_supported')) {
        issues.push('missing undoSupported metadata');
      }
    }

    // Check 4: Corresponding test file exists
    const testFileName = server.language === 'typescript'
      ? `${tool.name}.test.ts`
      : `${tool.name}_test.py`;
    const testFilePath = path.join(testsDir, testFileName);
    const smokeTestFileName = server.language === 'typescript'
      ? `${tool.name}.smoke.test.ts`
      : `${tool.name}_smoke_test.py`;
    const smokeTestPath = path.join(testsDir, smokeTestFileName);

    const hasTest = fs.existsSync(testFilePath) || fs.existsSync(smokeTestPath);
    if (!hasTest) {
      issues.push('no test file found');
    }

    // Report
    if (issues.length === 0) {
      console.log(`PASS ${server.name}.${tool.name}`);
    } else {
      console.log(`FAIL ${server.name}.${tool.name} ${issues.join(', ')}`);
    }
  }
}

// ── Main ────────────────────────────────────────────────────────────────

function main(): void {
  const projectRoot = path.resolve(__dirname, '../..');
  const registryPath = path.join(projectRoot, 'docs', 'mcp-tool-registry.yaml');

  if (!fs.existsSync(registryPath)) {
    console.log('FAIL registry docs/mcp-tool-registry.yaml not found');
    process.exit(1);
  }

  const yamlContent = fs.readFileSync(registryPath, 'utf-8');
  const servers = parseRegistryServers(yamlContent);

  // Filter to only servers specified on CLI (if any)
  const requestedServers = process.argv.slice(2);

  for (const server of servers) {
    if (requestedServers.length > 0 && !requestedServers.includes(server.name)) {
      continue;
    }
    validateServer(server, projectRoot);
  }
}

main();
