import { describe, it, expect, beforeAll } from 'vitest';
import { summarizeAnomalies } from '../src/tools/summarize_anomalies';
import { setupTestDir } from './helpers';

describe('data.summarize_anomalies', () => {
  beforeAll(() => {
    setupTestDir();
  });

  it('should detect range anomalies', async () => {
    const result = await summarizeAnomalies.execute({
      data: [
        { name: 'Alice', score: 85 },
        { name: 'Bob', score: 200 },
        { name: 'Charlie', score: -10 },
      ],
      rules: [{ field: 'score', type: 'range', min: 0, max: 100 }],
    });

    expect(result.success).toBe(true);
    expect(result.data.anomalies).toHaveLength(2);
    expect(result.data.anomalies[0].message).toContain('above maximum');
    expect(result.data.anomalies[1].message).toContain('below minimum');
  });

  it('should detect z-score anomalies', async () => {
    const result = await summarizeAnomalies.execute({
      data: [
        { value: 10 },
        { value: 11 },
        { value: 9 },
        { value: 10 },
        { value: 10 },
        { value: 100 }, // outlier
      ],
      rules: [{ field: 'value', type: 'z_score', z_threshold: 2 }],
    });

    expect(result.success).toBe(true);
    expect(result.data.anomalies.length).toBeGreaterThanOrEqual(1);
    // The outlier (100) should be flagged
    const outlier = result.data.anomalies.find((a: { row_index: number }) => a.row_index === 5);
    expect(outlier).toBeDefined();
    expect(outlier?.rule).toBe('z_score');
  });

  it('should detect missing values', async () => {
    const result = await summarizeAnomalies.execute({
      data: [
        { name: 'Alice', email: 'alice@test.com' },
        { name: 'Bob', email: '' },
        { name: '', email: 'charlie@test.com' },
      ],
      rules: [
        { field: 'name', type: 'missing' },
        { field: 'email', type: 'missing' },
      ],
    });

    expect(result.success).toBe(true);
    expect(result.data.anomalies).toHaveLength(2);
  });

  it('should detect pattern violations', async () => {
    const result = await summarizeAnomalies.execute({
      data: [
        { email: 'alice@test.com' },
        { email: 'not-an-email' },
        { email: 'bob@example.org' },
      ],
      rules: [{ field: 'email', type: 'pattern', pattern: '^.+@.+\\..+$' }],
    });

    expect(result.success).toBe(true);
    expect(result.data.anomalies).toHaveLength(1);
    expect(result.data.anomalies[0].value).toBe('not-an-email');
  });

  it('should auto-detect rules when none provided', async () => {
    const result = await summarizeAnomalies.execute({
      data: [
        { name: 'Alice', score: 10 },
        { name: 'Bob', score: 11 },
        { name: 'Charlie', score: 9 },
        { name: 'Diana', score: 10 },
        { name: '', score: 10 },
        { name: 'Eve', score: 100 },
      ],
    });

    expect(result.success).toBe(true);
    // Should auto-detect z-score for 'score' and missing for all
    expect(result.data.anomalies.length).toBeGreaterThan(0);
  });

  it('has correct metadata', () => {
    expect(summarizeAnomalies.name).toBe('data.summarize_anomalies');
    expect(summarizeAnomalies.confirmationRequired).toBe(false);
    expect(summarizeAnomalies.undoSupported).toBe(false);
  });
});
