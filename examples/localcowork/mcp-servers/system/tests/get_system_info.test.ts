import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { getSystemInfo } from '../src/tools/get_system_info';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system.get_system_info', () => {
  beforeAll(() => {
    setupTestBridge();
  });

  afterAll(() => {
    teardownTestBridge();
  });

  it('should return system info object', async () => {
    const result = await getSystemInfo.execute({});
    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
  });

  it('should have required fields: os, arch, cpu, ram_gb', async () => {
    const result = await getSystemInfo.execute({});
    const info = result.data;

    expect(info).toHaveProperty('os');
    expect(info).toHaveProperty('arch');
    expect(info).toHaveProperty('cpu');
    expect(info).toHaveProperty('ram_gb');

    expect(typeof info.os).toBe('string');
    expect(typeof info.arch).toBe('string');
    expect(typeof info.cpu).toBe('string');
    expect(typeof info.ram_gb).toBe('number');
  });

  it('should include optional gpu field from mock', async () => {
    const result = await getSystemInfo.execute({});
    const info = result.data;

    // MockSystemBridge returns gpu and npu
    expect(info.gpu).toBe('Apple M2 Pro GPU');
    expect(info.npu).toBe(true);
  });

  it('should return correct mock values', async () => {
    const result = await getSystemInfo.execute({});
    const info = result.data;

    expect(info.os).toBe('darwin');
    expect(info.arch).toBe('arm64');
    expect(info.cpu).toBe('Apple M2 Pro');
    expect(info.ram_gb).toBe(16.0);
  });

  it('has correct metadata', () => {
    expect(getSystemInfo.name).toBe('system.get_system_info');
    expect(getSystemInfo.confirmationRequired).toBe(false);
    expect(getSystemInfo.undoSupported).toBe(false);
  });
});
