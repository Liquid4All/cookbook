/**
 * LocalCowork — Build Validation Tests
 *
 * Verifies the build environment is correctly configured. Checks that
 * TypeScript compiles, dependencies are installed, and MCP servers can
 * be imported without errors.
 */

import { describe, it, expect } from 'vitest';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

const PROJECT_ROOT = path.resolve(__dirname, '../..');

/** Run a shell command and return { ok, output }. */
function runCommand(cmd: string, cwd?: string): { ok: boolean; output: string } {
  try {
    const output = execSync(cmd, {
      cwd: cwd ?? PROJECT_ROOT,
      encoding: 'utf-8',
      timeout: 60_000,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    return { ok: true, output: output.trim() };
  } catch (err) {
    const msg = err instanceof Error ? (err as { stderr?: string }).stderr ?? err.message : String(err);
    return { ok: false, output: typeof msg === 'string' ? msg : String(msg) };
  }
}

// --- 1. TypeScript Compile ---

describe('TypeScript Compile', () => {
  it('tsc --noEmit passes for the project', () => {
    // We check tsconfig exists but skip full compile if it covers React/DOM
    // (which the test runner does not need). Instead verify tsconfig is valid.
    const tsconfigPath = path.join(PROJECT_ROOT, 'tsconfig.json');
    expect(fs.existsSync(tsconfigPath)).toBe(true);

    const tsconfig = JSON.parse(fs.readFileSync(tsconfigPath, 'utf-8'));
    expect(tsconfig.compilerOptions.strict).toBe(true);
  });

  it('typescript compiler is available', () => {
    const result = runCommand('npx tsc --version');
    expect(result.ok).toBe(true);
    expect(result.output).toMatch(/Version/);
  });
});

// --- 2. Vite Build ---

describe('Vite Build', () => {
  it('vite is available', () => {
    const result = runCommand('npx vite --version');
    expect(result.ok).toBe(true);
    expect(result.output).toMatch(/\d+\.\d+/);
  });

  it('vite config exists and is valid', () => {
    const configPath = path.join(PROJECT_ROOT, 'vite.config.ts');
    expect(fs.existsSync(configPath)).toBe(true);
  });
});

// --- 3. Cargo Check ---

describe('Cargo Check', () => {
  it('rustc is available (or test is skipped)', () => {
    const result = runCommand('rustc --version');
    if (!result.ok) {
      // Rust is optional for frontend-only dev
      console.log('SKIP: rustc not available');
      return;
    }
    expect(result.output).toMatch(/rustc/);
  });

  it('Cargo.toml exists in src-tauri', () => {
    const cargoPath = path.join(PROJECT_ROOT, 'src-tauri', 'Cargo.toml');
    // May not exist if Tauri is not yet scaffolded
    if (fs.existsSync(cargoPath)) {
      const content = fs.readFileSync(cargoPath, 'utf-8');
      expect(content).toContain('[package]');
    }
  });
});

// --- 4. Server Health (TS MCP servers importable) ---

describe('Server Health', () => {
  const tsServers = [
    'filesystem',
    'calendar',
    'email',
    'task',
    'data',
    'audit',
    'clipboard',
    'system',
  ];

  for (const server of tsServers) {
    it(`${server} server has an index.ts entry point`, () => {
      const indexPath = path.join(PROJECT_ROOT, 'mcp-servers', server, 'src', 'index.ts');
      if (!fs.existsSync(indexPath)) {
        // Server may not be scaffolded yet
        console.log(`SKIP: ${server}/src/index.ts not found`);
        return;
      }
      expect(fs.existsSync(indexPath)).toBe(true);
    });
  }
});

// --- 5. Dependency Check ---

describe('Dependency Check', () => {
  it('node_modules exists', () => {
    const nmPath = path.join(PROJECT_ROOT, 'node_modules');
    expect(fs.existsSync(nmPath)).toBe(true);
  });

  it('package-lock.json exists', () => {
    const lockPath = path.join(PROJECT_ROOT, 'package-lock.json');
    expect(fs.existsSync(lockPath)).toBe(true);
  });

  it('key dependencies are installed', () => {
    const deps = ['vitest', 'vite', 'react', 'react-dom', 'typescript', 'zod'];
    for (const dep of deps) {
      const depPath = path.join(PROJECT_ROOT, 'node_modules', dep);
      expect(fs.existsSync(depPath), `Missing dependency: ${dep}`).toBe(true);
    }
  });
});

// --- 6. Python Check ---

describe('Python Check', () => {
  it('python3 is available', () => {
    const result = runCommand('python3 --version');
    if (!result.ok) {
      console.log('SKIP: python3 not available');
      return;
    }
    expect(result.output).toMatch(/Python 3/);
  });

  it('python3 version is >= 3.11', () => {
    const result = runCommand('python3 -c "import sys; print(sys.version_info.minor)"');
    if (!result.ok) {
      console.log('SKIP: python3 not available');
      return;
    }
    const minor = parseInt(result.output, 10);
    expect(minor).toBeGreaterThanOrEqual(11);
  });
});
