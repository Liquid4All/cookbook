import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { toggleDoNotDisturb } from '../src/tools/toggle_do_not_disturb';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system-settings.toggle_do_not_disturb', () => {
  beforeAll(() => { setupTestBridge(); });
  afterAll(() => { teardownTestBridge(); });

  it('should enable DND', async () => {
    const result = await toggleDoNotDisturb.execute({ enable: true });
    expect(result.success).toBe(true);
    expect(result.data.previous_value).toBe(false);
    expect(result.data.new_value).toBe(true);
  });

  it('should disable DND', async () => {
    await toggleDoNotDisturb.execute({ enable: true });
    const result = await toggleDoNotDisturb.execute({ enable: false });
    expect(result.success).toBe(true);
    expect(result.data.previous_value).toBe(true);
    expect(result.data.new_value).toBe(false);
  });

  it('has correct metadata', () => {
    expect(toggleDoNotDisturb.name).toBe('system-settings.toggle_do_not_disturb');
    expect(toggleDoNotDisturb.confirmationRequired).toBe(true);
    expect(toggleDoNotDisturb.undoSupported).toBe(true);
  });
});
