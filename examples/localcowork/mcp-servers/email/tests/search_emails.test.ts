import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { searchEmails } from '../src/tools/search_emails';
import { setupTestDb, teardownTestDb } from './helpers';

describe('email.search_emails', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({
      emails: {
        count: 10,
        folders: ['inbox', 'sent', 'archive'],
        withThreads: true,
      },
    });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should find emails matching subject', async () => {
    const result = await searchEmails.execute({ query: 'Email subject 1' });
    expect(result.success).toBe(true);
    expect(result.data.length).toBeGreaterThanOrEqual(1);
    const matchingSubjects = result.data.some(
      (email: { subject: string }) => email.subject.includes('Email subject 1'),
    );
    expect(matchingSubjects).toBe(true);
  });

  it('should find emails matching body content', async () => {
    const result = await searchEmails.execute({ query: 'searching' });
    expect(result.success).toBe(true);
    // All seeded non-thread emails contain "searching" in body
    expect(result.data.length).toBeGreaterThanOrEqual(1);
  });

  it('should filter by folder', async () => {
    const result = await searchEmails.execute({ query: 'Email subject', folder: 'inbox' });
    expect(result.success).toBe(true);
    for (const email of result.data) {
      expect(email.folder).toBe('inbox');
    }
  });

  it('should respect the limit parameter', async () => {
    const result = await searchEmails.execute({ query: 'Email', limit: 3 });
    expect(result.success).toBe(true);
    expect(result.data.length).toBeLessThanOrEqual(3);
  });

  it('should return results ordered by received_at DESC', async () => {
    const result = await searchEmails.execute({ query: 'Email' });
    expect(result.success).toBe(true);
    if (result.data.length > 1) {
      for (let i = 1; i < result.data.length; i++) {
        const prev = result.data[i - 1].received_at;
        const curr = result.data[i].received_at;
        expect(prev >= curr).toBe(true);
      }
    }
  });

  it('should return empty array when no matches', async () => {
    const result = await searchEmails.execute({ query: 'xyznonexistent12345' });
    expect(result.success).toBe(true);
    expect(result.data).toEqual([]);
  });

  it('should find thread emails', async () => {
    const result = await searchEmails.execute({ query: 'kickoff meeting' });
    expect(result.success).toBe(true);
    expect(result.data.length).toBeGreaterThanOrEqual(1);
  });

  it('should parse to_addresses as array', async () => {
    const result = await searchEmails.execute({ query: 'Email' });
    expect(result.success).toBe(true);
    for (const email of result.data) {
      expect(Array.isArray(email.to)).toBe(true);
    }
  });

  it('should include is_read as boolean', async () => {
    const result = await searchEmails.execute({ query: 'Email' });
    expect(result.success).toBe(true);
    for (const email of result.data) {
      expect(typeof email.is_read).toBe('boolean');
    }
  });

  it('should default limit to 20', async () => {
    const result = await searchEmails.execute({ query: 'Email' });
    expect(result.success).toBe(true);
    expect(result.data.length).toBeLessThanOrEqual(20);
  });

  it('has correct metadata', () => {
    expect(searchEmails.name).toBe('email.search_emails');
    expect(searchEmails.confirmationRequired).toBe(false);
    expect(searchEmails.undoSupported).toBe(false);
  });
});
