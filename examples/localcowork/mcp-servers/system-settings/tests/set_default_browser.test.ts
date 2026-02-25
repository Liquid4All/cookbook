import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { setDefaultBrowser } from '../src/tools/set_default_browser';
import { getDefaultApps } from '../src/tools/get_default_apps';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system-settings.set_default_browser', () => {
  beforeAll(() => { setupTestBridge(); });
  afterAll(() => { teardownTestBridge(); });

  it('should set default browser', async () => {
    const result = await setDefaultBrowser.execute({ browser: 'Firefox' });
    expect(result.success).toBe(true);
    expect(result.data.previous_value).toBe('Safari');
    expect(result.data.new_value).toBe('Firefox');
  });

  it('should update the stored value', async () => {
    await setDefaultBrowser.execute({ browser: 'Google Chrome' });
    const get = await getDefaultApps.execute({});
    expect(get.data.browser).toBe('Google Chrome');
  });

  it('should reject unknown browsers', async () => {
    await expect(
      setDefaultBrowser.execute({ browser: 'NotABrowser' }),
    ).rejects.toThrow('Unknown browser');
  });

  it('should be case-insensitive for known browsers', async () => {
    const result = await setDefaultBrowser.execute({ browser: 'firefox' });
    expect(result.success).toBe(true);
  });

  it('has correct metadata', () => {
    expect(setDefaultBrowser.name).toBe('system-settings.set_default_browser');
    expect(setDefaultBrowser.confirmationRequired).toBe(true);
    expect(setDefaultBrowser.undoSupported).toBe(true);
  });
});
