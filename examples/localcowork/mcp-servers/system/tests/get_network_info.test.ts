import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { getNetworkInfo } from '../src/tools/get_network_info';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system.get_network_info', () => {
  beforeAll(() => {
    setupTestBridge();
  });

  afterAll(() => {
    teardownTestBridge();
  });

  it('should return network info result', async () => {
    const result = await getNetworkInfo.execute({});
    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
    expect(result.data.interfaces).toBeDefined();
  });

  it('should return array of interfaces', async () => {
    const result = await getNetworkInfo.execute({});
    expect(Array.isArray(result.data.interfaces)).toBe(true);
    expect(result.data.interfaces.length).toBeGreaterThan(0);
  });

  it('each interface should have required fields', async () => {
    const result = await getNetworkInfo.execute({});
    for (const iface of result.data.interfaces) {
      expect(iface).toHaveProperty('name');
      expect(iface).toHaveProperty('internal');
      expect(typeof iface.name).toBe('string');
      expect(typeof iface.internal).toBe('boolean');
    }
  });

  it('should include loopback and external interface', async () => {
    const result = await getNetworkInfo.execute({});
    const lo = result.data.interfaces.find((i) => i.name === 'lo0');
    const en = result.data.interfaces.find((i) => i.name === 'en0');

    expect(lo).toBeDefined();
    expect(lo!.internal).toBe(true);
    expect(lo!.ip4).toBe('127.0.0.1');

    expect(en).toBeDefined();
    expect(en!.internal).toBe(false);
    expect(en!.ip4).toBe('192.168.1.42');
  });

  it('has correct metadata', () => {
    expect(getNetworkInfo.name).toBe('system.get_network_info');
    expect(getNetworkInfo.confirmationRequired).toBe(false);
    expect(getNetworkInfo.undoSupported).toBe(false);
  });
});
