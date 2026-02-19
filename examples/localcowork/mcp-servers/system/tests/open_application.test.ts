import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { openApplication } from '../src/tools/open_application';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system.open_application', () => {
  beforeAll(() => {
    setupTestBridge();
  });

  afterAll(() => {
    teardownTestBridge();
  });

  it('should open an application and return success', async () => {
    const result = await openApplication.execute({ app_name: 'Safari' });
    expect(result.success).toBe(true);
    expect(result.data.success).toBe(true);
  });

  it('should return a pid on success', async () => {
    const result = await openApplication.execute({ app_name: 'Terminal' });
    expect(result.data.pid).toBeDefined();
    expect(typeof result.data.pid).toBe('number');
    // MockSystemBridge returns pid 12345
    expect(result.data.pid).toBe(12345);
  });

  it('should reject empty app_name via schema validation', () => {
    const parseResult = openApplication.paramsSchema.safeParse({ app_name: '' });
    expect(parseResult.success).toBe(false);
  });

  it('should reject missing app_name via schema validation', () => {
    const parseResult = openApplication.paramsSchema.safeParse({});
    expect(parseResult.success).toBe(false);
  });

  it('has confirmation metadata set to true', () => {
    expect(openApplication.confirmationRequired).toBe(true);
    expect(openApplication.undoSupported).toBe(false);
  });

  it('has correct tool name', () => {
    expect(openApplication.name).toBe('system.open_application');
  });
});
