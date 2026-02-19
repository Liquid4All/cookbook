import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { getDefaultApps } from '../src/tools/get_default_apps';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system-settings.get_default_apps', () => {
  beforeAll(() => { setupTestBridge(); });
  afterAll(() => { teardownTestBridge(); });

  it('should return default apps', async () => {
    const result = await getDefaultApps.execute({});
    expect(result.success).toBe(true);
    expect(result.data).toHaveProperty('browser');
  });

  it('should return mock values', async () => {
    const result = await getDefaultApps.execute({});
    expect(result.data.browser).toBe('Safari');
    expect(result.data.email).toBe('Mail');
    expect(result.data.pdf_viewer).toBe('Preview');
  });

  it('has correct metadata', () => {
    expect(getDefaultApps.name).toBe('system-settings.get_default_apps');
    expect(getDefaultApps.confirmationRequired).toBe(false);
  });
});
