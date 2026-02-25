import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { getDisplaySettings } from '../src/tools/get_display_settings';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system-settings.get_display_settings', () => {
  beforeAll(() => { setupTestBridge(); });
  afterAll(() => { teardownTestBridge(); });

  it('should return display settings', async () => {
    const result = await getDisplaySettings.execute({});
    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
    expect(result.data).toHaveProperty('sleep_minutes');
  });

  it('should return mock values', async () => {
    const result = await getDisplaySettings.execute({});
    expect(result.data.sleep_minutes).toBe(10);
    expect(result.data.brightness).toBe(75);
  });

  it('has correct metadata', () => {
    expect(getDisplaySettings.name).toBe('system-settings.get_display_settings');
    expect(getDisplaySettings.confirmationRequired).toBe(false);
    expect(getDisplaySettings.undoSupported).toBe(false);
  });
});
