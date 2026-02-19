import { describe, it, expect, beforeAll } from 'vitest';
import { deduplicateRecords } from '../src/tools/deduplicate_records';
import { setupTestDir } from './helpers';

describe('data.deduplicate_records', () => {
  beforeAll(() => {
    // Initialize sandbox (needed by shared helpers even if not file-based)
    setupTestDir();
  });

  it('should find exact duplicates', async () => {
    const result = await deduplicateRecords.execute({
      data: [
        { name: 'Alice', email: 'alice@test.com' },
        { name: 'Bob', email: 'bob@test.com' },
        { name: 'Alice', email: 'alice@test.com' },
      ],
      match_fields: ['name', 'email'],
      threshold: 1.0,
    });

    expect(result.success).toBe(true);
    expect(result.data.unique).toHaveLength(1); // Bob
    expect(result.data.duplicates).toHaveLength(1); // Alice group
    expect(result.data.duplicates[0]).toHaveLength(2); // two Alices
  });

  it('should find fuzzy string matches', async () => {
    const result = await deduplicateRecords.execute({
      data: [
        { name: 'Robert Smith', city: 'NYC' },
        { name: 'Robert Smyth', city: 'NYC' },
        { name: 'Jane Doe', city: 'LA' },
      ],
      match_fields: ['name'],
      threshold: 0.8,
    });

    expect(result.success).toBe(true);
    // Robert Smith and Robert Smyth should be duplicates
    expect(result.data.duplicates).toHaveLength(1);
    expect(result.data.unique).toHaveLength(1); // Jane
  });

  it('should return all unique when no duplicates', async () => {
    const result = await deduplicateRecords.execute({
      data: [
        { name: 'Alice', id: 1 },
        { name: 'Bob', id: 2 },
        { name: 'Charlie', id: 3 },
      ],
      match_fields: ['name'],
      threshold: 0.85,
    });

    expect(result.success).toBe(true);
    expect(result.data.unique).toHaveLength(3);
    expect(result.data.duplicates).toHaveLength(0);
  });

  it('should reject unknown fields', async () => {
    await expect(
      deduplicateRecords.execute({
        data: [{ name: 'Alice' }],
        match_fields: ['nonexistent'],
        threshold: 0.85,
      }),
    ).rejects.toThrow('Field "nonexistent" not found');
  });

  it('has correct metadata', () => {
    expect(deduplicateRecords.name).toBe('data.deduplicate_records');
    expect(deduplicateRecords.confirmationRequired).toBe(false);
    expect(deduplicateRecords.undoSupported).toBe(false);
  });
});
