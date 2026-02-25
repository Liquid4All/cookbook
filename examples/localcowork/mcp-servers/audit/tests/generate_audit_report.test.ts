import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { generateAuditReport } from '../src/tools/generate_audit_report';
import { setupTestDb, teardownTestDb } from './helpers';

describe('audit.generate_audit_report', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({ sessionId: 'sess-report-001', entryCount: 5 });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should generate a text report', async () => {
    const result = await generateAuditReport.execute({
      session_id: 'sess-report-001',
    });
    expect(result.success).toBe(true);
    expect(typeof result.data.report).toBe('string');
    expect(result.data.report).toContain('AUDIT REPORT');
    expect(result.data.report).toContain('sess-report-001');
  });

  it('should include tool usage counts', async () => {
    const result = await generateAuditReport.execute({
      session_id: 'sess-report-001',
    });
    expect(result.data.report).toContain('Tool Usage');
  });

  it('should include timeline', async () => {
    const result = await generateAuditReport.execute({
      session_id: 'sess-report-001',
    });
    expect(result.data.report).toContain('Timeline');
  });

  it('should handle empty session gracefully', async () => {
    const result = await generateAuditReport.execute({
      session_id: 'nonexistent-session',
    });
    expect(result.success).toBe(true);
    expect(result.data.report).toContain('No audit entries found');
  });

  it('has correct metadata', () => {
    expect(generateAuditReport.name).toBe('audit.generate_audit_report');
    expect(generateAuditReport.confirmationRequired).toBe(false);
    expect(generateAuditReport.undoSupported).toBe(false);
  });
});
