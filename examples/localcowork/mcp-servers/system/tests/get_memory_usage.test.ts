import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { getMemoryUsage } from '../src/tools/get_memory_usage';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system.get_memory_usage', () => {
  beforeAll(() => {
    setupTestBridge();
  });

  afterAll(() => {
    teardownTestBridge();
  });

  it('should return memory usage object', async () => {
    const result = await getMemoryUsage.execute({});
    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
  });

  it('should have required fields', async () => {
    const result = await getMemoryUsage.execute({});
    const mem = result.data;

    expect(mem).toHaveProperty('total_gb');
    expect(mem).toHaveProperty('used_gb');
    expect(mem).toHaveProperty('free_gb');
    expect(mem).toHaveProperty('swap_total_gb');
    expect(mem).toHaveProperty('swap_used_gb');
    expect(mem).toHaveProperty('usage_percent');
  });

  it('should return correct mock values', async () => {
    const result = await getMemoryUsage.execute({});
    const mem = result.data;

    expect(mem.total_gb).toBe(16.0);
    expect(mem.used_gb).toBe(10.5);
    expect(mem.free_gb).toBe(5.5);
    expect(mem.usage_percent).toBe(65.6);
  });

  it('should have numeric values', async () => {
    const result = await getMemoryUsage.execute({});
    const mem = result.data;

    expect(typeof mem.total_gb).toBe('number');
    expect(typeof mem.used_gb).toBe('number');
    expect(typeof mem.free_gb).toBe('number');
    expect(typeof mem.usage_percent).toBe('number');
    expect(mem.total_gb).toBeGreaterThan(0);
    expect(mem.usage_percent).toBeGreaterThanOrEqual(0);
    expect(mem.usage_percent).toBeLessThanOrEqual(100);
  });

  it('has correct metadata', () => {
    expect(getMemoryUsage.name).toBe('system.get_memory_usage');
    expect(getMemoryUsage.confirmationRequired).toBe(false);
    expect(getMemoryUsage.undoSupported).toBe(false);
  });
});
