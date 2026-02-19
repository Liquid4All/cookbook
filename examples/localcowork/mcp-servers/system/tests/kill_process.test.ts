import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { killProcess } from '../src/tools/kill_process';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system.kill_process', () => {
  beforeAll(() => {
    setupTestBridge();
  });

  afterAll(() => {
    teardownTestBridge();
  });

  it('should kill a process by PID', async () => {
    const result = await killProcess.execute({ pid: 1234 });
    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
    expect(result.data.pid).toBe(1234);
    expect(result.data.signal).toBe('SIGTERM');
  });

  it('should accept a custom signal', async () => {
    const result = await killProcess.execute({ pid: 5678, signal: 'SIGKILL' });
    expect(result.success).toBe(true);
    expect(result.data.signal).toBe('SIGKILL');
  });

  it('should reject invalid signals', async () => {
    await expect(killProcess.execute({ pid: 1, signal: 'INVALID' })).rejects.toThrow(
      'Invalid signal',
    );
  });

  it('should accept SIGINT signal', async () => {
    const result = await killProcess.execute({ pid: 999, signal: 'SIGINT' });
    expect(result.success).toBe(true);
    expect(result.data.signal).toBe('SIGINT');
  });

  it('has correct metadata', () => {
    expect(killProcess.name).toBe('system.kill_process');
    expect(killProcess.confirmationRequired).toBe(true);
    expect(killProcess.undoSupported).toBe(false);
  });

  it('is marked as requiring confirmation (destructive)', () => {
    expect(killProcess.confirmationRequired).toBe(true);
  });
});
