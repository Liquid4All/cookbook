import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { setAudioVolume } from '../src/tools/set_audio_volume';
import { getAudioSettings } from '../src/tools/get_audio_settings';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system-settings.set_audio_volume', () => {
  beforeAll(() => { setupTestBridge(); });
  afterAll(() => { teardownTestBridge(); });

  it('should set volume', async () => {
    const result = await setAudioVolume.execute({ volume: 75 });
    expect(result.success).toBe(true);
    expect(result.data.previous_value).toBe(50);
    expect(result.data.new_value).toBe(75);
  });

  it('should update the stored value', async () => {
    await setAudioVolume.execute({ volume: 30 });
    const get = await getAudioSettings.execute({});
    expect(get.data.output_volume).toBe(30);
  });

  it('should accept 0 for mute', async () => {
    const result = await setAudioVolume.execute({ volume: 0 });
    expect(result.success).toBe(true);
    expect(result.data.new_value).toBe(0);
  });

  it('has correct metadata', () => {
    expect(setAudioVolume.name).toBe('system-settings.set_audio_volume');
    expect(setAudioVolume.confirmationRequired).toBe(true);
    expect(setAudioVolume.undoSupported).toBe(true);
  });
});
