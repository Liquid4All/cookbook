import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { getAudioSettings } from '../src/tools/get_audio_settings';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system-settings.get_audio_settings', () => {
  beforeAll(() => { setupTestBridge(); });
  afterAll(() => { teardownTestBridge(); });

  it('should return audio settings', async () => {
    const result = await getAudioSettings.execute({});
    expect(result.success).toBe(true);
    expect(result.data).toHaveProperty('output_volume');
    expect(result.data).toHaveProperty('muted');
  });

  it('should return mock values', async () => {
    const result = await getAudioSettings.execute({});
    expect(result.data.output_volume).toBe(50);
    expect(result.data.muted).toBe(false);
    expect(result.data.output_device).toBe('Speakers');
  });

  it('has correct metadata', () => {
    expect(getAudioSettings.name).toBe('system-settings.get_audio_settings');
    expect(getAudioSettings.confirmationRequired).toBe(false);
  });
});
