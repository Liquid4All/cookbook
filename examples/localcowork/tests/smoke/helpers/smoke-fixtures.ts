/**
 * ─────────────────────────────────────────────────────────────────────────────
 * Smoke Test Helpers — Shared Fixtures and Utilities
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Lightweight helpers for smoke tests. These create minimal temp directories
 * and sample files for tools that need filesystem input.
 *
 * Design principles:
 *   - Every helper must clean up after itself
 *   - No helper should take more than 50ms
 *   - Prefer in-memory data over disk I/O where possible
 * ─────────────────────────────────────────────────────────────────────────────
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

/**
 * Create a temporary directory with sample files.
 * Returns the path and a cleanup function.
 */
export async function createTempDir(
  files: Record<string, string> = {}
): Promise<{ dir: string; cleanup: () => void }> {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'localcowork-smoke-'));

  for (const [name, content] of Object.entries(files)) {
    const filePath = path.join(dir, name);
    const fileDir = path.dirname(filePath);
    if (!fs.existsSync(fileDir)) {
      fs.mkdirSync(fileDir, { recursive: true });
    }
    fs.writeFileSync(filePath, content, 'utf-8');
  }

  const cleanup = () => {
    fs.rmSync(dir, { recursive: true, force: true });
  };

  return { dir, cleanup };
}

/**
 * Sample file contents for common test scenarios.
 */
export const SAMPLE_FILES = {
  text: 'Hello, this is a sample text file for smoke testing.',

  csv: [
    'name,amount,date',
    'Coffee Shop,4.50,2025-01-15',
    'Office Depot,23.99,2025-01-16',
    'Gas Station,45.00,2025-01-17',
  ].join('\n'),

  json: JSON.stringify({
    name: 'Test Document',
    version: 1,
    items: [
      { id: 1, label: 'Item One' },
      { id: 2, label: 'Item Two' },
    ],
  }, null, 2),

  markdown: [
    '# Test Document',
    '',
    '## Section 1',
    'This is a test document for smoke testing.',
    '',
    '## Section 2',
    '- Item A',
    '- Item B',
    '- Item C',
  ].join('\n'),

  /** Contains fake PII for security server smoke tests */
  withPii: [
    'Name: John Doe',
    'SSN: 123-45-6789',
    'Email: john@example.com',
    'Phone: (555) 123-4567',
  ].join('\n'),

  /** Contains fake secrets for security server smoke tests */
  withSecrets: [
    'AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE',
    'DATABASE_URL=postgres://user:password@localhost/db',
    'STRIPE_SECRET_KEY=sk_test_FAKE_DO_NOT_USE_000000',
  ].join('\n'),
};

/**
 * Minimal valid params for common tool types.
 * Use these as starting points in smoke tests.
 */
export const MINIMAL_PARAMS = {
  filesystem: {
    list_directory: { path: '/tmp' },
    read_file: { path: '/dev/null' },
    get_file_info: { path: '/tmp' },
    search_files: { path: '/tmp', pattern: '*.txt' },
  },
  data: {
    parse_csv: { content: SAMPLE_FILES.csv },
  },
};
