import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { getDiskUsage } from '../src/tools/get_disk_usage';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system.get_disk_usage', () => {
  beforeAll(() => {
    setupTestBridge();
  });

  afterAll(() => {
    teardownTestBridge();
  });

  it('should return disk usage result', async () => {
    const result = await getDiskUsage.execute({});
    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
    expect(result.data.volumes).toBeDefined();
  });

  it('should return array of volumes', async () => {
    const result = await getDiskUsage.execute({});
    expect(Array.isArray(result.data.volumes)).toBe(true);
    expect(result.data.volumes.length).toBeGreaterThan(0);
  });

  it('each volume should have required fields', async () => {
    const result = await getDiskUsage.execute({});
    for (const vol of result.data.volumes) {
      expect(vol).toHaveProperty('mount_point');
      expect(vol).toHaveProperty('filesystem');
      expect(vol).toHaveProperty('total_gb');
      expect(vol).toHaveProperty('used_gb');
      expect(vol).toHaveProperty('free_gb');
      expect(vol).toHaveProperty('usage_percent');
      expect(typeof vol.mount_point).toBe('string');
      expect(typeof vol.total_gb).toBe('number');
      expect(vol.total_gb).toBeGreaterThan(0);
    }
  });

  it('should return correct mock volumes', async () => {
    const result = await getDiskUsage.execute({});
    const root = result.data.volumes.find((v) => v.mount_point === '/');
    expect(root).toBeDefined();
    expect(root!.total_gb).toBe(500.0);
    expect(root!.usage_percent).toBe(70.0);
  });

  it('has correct metadata', () => {
    expect(getDiskUsage.name).toBe('system.get_disk_usage');
    expect(getDiskUsage.confirmationRequired).toBe(false);
    expect(getDiskUsage.undoSupported).toBe(false);
  });
});
