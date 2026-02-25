import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { summarizeThread } from '../src/tools/summarize_thread';
import { setupTestDb, teardownTestDb } from './helpers';

describe('email.summarize_thread', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({
      emails: { count: 0, withThreads: true },
    });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should return a summary with all fields', async () => {
    const result = await summarizeThread.execute({ thread_id: 'thread-abc-123' });
    expect(result.success).toBe(true);
    expect(result.data.summary).toBeDefined();
    expect(typeof result.data.summary).toBe('string');
    expect(result.data.participants).toBeInstanceOf(Array);
    expect(result.data.key_points).toBeInstanceOf(Array);
  });

  it('should extract unique participants', async () => {
    const result = await summarizeThread.execute({ thread_id: 'thread-abc-123' });
    expect(result.success).toBe(true);
    const participants = result.data.participants as string[];
    expect(participants).toContain('alice@example.com');
    expect(participants).toContain('bob@example.com');
    expect(participants.length).toBe(2);
  });

  it('should extract key points from subjects', async () => {
    const result = await summarizeThread.execute({ thread_id: 'thread-abc-123' });
    expect(result.success).toBe(true);
    const keyPoints = result.data.key_points as string[];
    // "Re: Project kickoff meeting" should be cleaned to "Project kickoff meeting"
    expect(keyPoints).toContain('Project kickoff meeting');
  });

  it('should deduplicate key points from Re: subjects', async () => {
    const result = await summarizeThread.execute({ thread_id: 'thread-abc-123' });
    expect(result.success).toBe(true);
    const keyPoints = result.data.key_points as string[];
    // All subjects reduce to "Project kickoff meeting" after removing Re: prefix
    expect(keyPoints.length).toBe(1);
  });

  it('should include message count in summary', async () => {
    const result = await summarizeThread.execute({ thread_id: 'thread-abc-123' });
    expect(result.success).toBe(true);
    expect(result.data.summary).toContain('3 message(s)');
  });

  it('should include date range in summary', async () => {
    const result = await summarizeThread.execute({ thread_id: 'thread-abc-123' });
    expect(result.success).toBe(true);
    expect(result.data.summary).toContain('2026-02-01T09:00:00Z');
    expect(result.data.summary).toContain('2026-02-01T11:00:00Z');
  });

  it('should throw MCPError for non-existent thread', async () => {
    await expect(
      summarizeThread.execute({ thread_id: 'nonexistent-thread' }),
    ).rejects.toThrow('No emails found for thread_id');
  });

  it('has correct metadata', () => {
    expect(summarizeThread.name).toBe('email.summarize_thread');
    expect(summarizeThread.confirmationRequired).toBe(false);
    expect(summarizeThread.undoSupported).toBe(false);
  });
});
