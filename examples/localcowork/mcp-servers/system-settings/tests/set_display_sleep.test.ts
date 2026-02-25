import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { setDisplaySleep } from '../src/tools/set_display_sleep';
import { getDisplaySettings } from '../src/tools/get_display_settings';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system-settings.set_display_sleep', () => {
  beforeAll(() => { setupTestBridge(); });
  afterAll(() => { teardownTestBridge(); });

  it('should set display sleep timer', async () => {
    const result = await setDisplaySleep.execute({ minutes: 30 });
    expect(result.success).toBe(true);
    expect(result.data.previous_value).toBe(10);
    expect(result.data.new_value).toBe(30);
  });

  it('should update the stored value', async () => {
    await setDisplaySleep.execute({ minutes: 15 });
    const get = await getDisplaySettings.execute({});
    expect(get.data.sleep_minutes).toBe(15);
  });

  it('should accept 0 for never sleep', async () => {
    const result = await setDisplaySleep.execute({ minutes: 0 });
    expect(result.success).toBe(true);
    expect(result.data.new_value).toBe(0);
  });

  it('has correct metadata', () => {
    expect(setDisplaySleep.name).toBe('system-settings.set_display_sleep');
    expect(setDisplaySleep.confirmationRequired).toBe(true);
    expect(setDisplaySleep.undoSupported).toBe(true);
  });
});
