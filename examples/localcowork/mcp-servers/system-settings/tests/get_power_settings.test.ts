import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { getPowerSettings } from '../src/tools/get_power_settings';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system-settings.get_power_settings', () => {
  beforeAll(() => { setupTestBridge(); });
  afterAll(() => { teardownTestBridge(); });

  it('should return power settings', async () => {
    const result = await getPowerSettings.execute({});
    expect(result.success).toBe(true);
    expect(result.data).toHaveProperty('display_sleep_minutes');
    expect(result.data).toHaveProperty('system_sleep_minutes');
  });

  it('should return mock values', async () => {
    const result = await getPowerSettings.execute({});
    expect(result.data.system_sleep_minutes).toBe(30);
    expect(result.data.wake_on_network).toBe(false);
  });

  it('has correct metadata', () => {
    expect(getPowerSettings.name).toBe('system-settings.get_power_settings');
    expect(getPowerSettings.confirmationRequired).toBe(false);
  });
});
