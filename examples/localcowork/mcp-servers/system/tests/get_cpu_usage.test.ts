import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { getCpuUsage } from '../src/tools/get_cpu_usage';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system.get_cpu_usage', () => {
  beforeAll(() => {
    setupTestBridge();
  });

  afterAll(() => {
    teardownTestBridge();
  });

  it('should return CPU usage object', async () => {
    const result = await getCpuUsage.execute({});
    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
  });

  it('should have required fields', async () => {
    const result = await getCpuUsage.execute({});
    const cpu = result.data;

    expect(cpu).toHaveProperty('cores');
    expect(cpu).toHaveProperty('model');
    expect(cpu).toHaveProperty('load_average');
    expect(cpu).toHaveProperty('per_core_percent');
  });

  it('should return correct mock values', async () => {
    const result = await getCpuUsage.execute({});
    const cpu = result.data;

    expect(cpu.cores).toBe(10);
    expect(cpu.model).toBe('Apple M2 Pro');
    expect(cpu.load_average).toHaveLength(3);
    expect(cpu.per_core_percent).toHaveLength(10);
  });

  it('per-core percentages should be valid', async () => {
    const result = await getCpuUsage.execute({});
    for (const pct of result.data.per_core_percent) {
      expect(typeof pct).toBe('number');
      expect(pct).toBeGreaterThanOrEqual(0);
      expect(pct).toBeLessThanOrEqual(100);
    }
  });

  it('has correct metadata', () => {
    expect(getCpuUsage.name).toBe('system.get_cpu_usage');
    expect(getCpuUsage.confirmationRequired).toBe(false);
    expect(getCpuUsage.undoSupported).toBe(false);
  });
});
