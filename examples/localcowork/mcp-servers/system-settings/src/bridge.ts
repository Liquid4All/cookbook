/**
 * Settings Bridge — Abstraction Layer for OS Settings
 *
 * Platform-agnostic interface for reading and modifying system settings.
 * Implementations: DarwinSettingsBridge (macOS), WindowsSettingsBridge (Windows),
 * MockSettingsBridge (tests).
 */

import * as os from 'os';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface DisplaySettings {
  readonly sleep_minutes: number;
  readonly brightness?: number;
  readonly resolution?: string;
}

export interface AudioSettings {
  readonly output_volume: number;
  readonly input_volume: number;
  readonly muted: boolean;
  readonly output_device?: string;
}

export interface DefaultApps {
  readonly browser?: string;
  readonly email?: string;
  readonly pdf_viewer?: string;
}

export interface PowerSettings {
  readonly display_sleep_minutes: number;
  readonly system_sleep_minutes: number;
  readonly disk_sleep_minutes: number;
  readonly wake_on_network: boolean;
}

export interface DndStatus {
  readonly enabled: boolean;
}

export interface SetResult {
  readonly success: boolean;
  readonly previous_value?: unknown;
  readonly new_value?: unknown;
}

// ─── Bridge Interface ────────────────────────────────────────────────────────

export interface SettingsBridge {
  getDisplaySettings(): Promise<DisplaySettings>;
  setDisplaySleep(minutes: number): Promise<SetResult>;
  getAudioSettings(): Promise<AudioSettings>;
  setAudioVolume(volume: number): Promise<SetResult>;
  getDefaultApps(): Promise<DefaultApps>;
  setDefaultBrowser(browser: string): Promise<SetResult>;
  getPowerSettings(): Promise<PowerSettings>;
  toggleDoNotDisturb(enable: boolean): Promise<SetResult>;
}

// ─── Platform Factory ────────────────────────────────────────────────────────

async function createPlatformBridge(): Promise<SettingsBridge> {
  const platform = os.platform();
  if (platform === 'darwin') {
    const { DarwinSettingsBridge } = await import('./platforms/darwin.js');
    return new DarwinSettingsBridge();
  }
  if (platform === 'win32') {
    const { WindowsSettingsBridge } = await import('./platforms/win32.js');
    return new WindowsSettingsBridge();
  }
  return new UnsupportedPlatformBridge(platform);
}

/** Initialize the platform bridge. Call once at server startup. */
export async function initBridge(): Promise<void> {
  activeBridge = await createPlatformBridge();
}

// ─── Unsupported Platform ────────────────────────────────────────────────────

class UnsupportedPlatformBridge implements SettingsBridge {
  private readonly platform: string;

  constructor(platform: string) {
    this.platform = platform;
  }

  private fail(): never {
    throw new Error(`System settings not available on ${this.platform}. Supported: macOS, Windows.`);
  }

  async getDisplaySettings(): Promise<DisplaySettings> { this.fail(); }
  async setDisplaySleep(_m: number): Promise<SetResult> { this.fail(); }
  async getAudioSettings(): Promise<AudioSettings> { this.fail(); }
  async setAudioVolume(_v: number): Promise<SetResult> { this.fail(); }
  async getDefaultApps(): Promise<DefaultApps> { this.fail(); }
  async setDefaultBrowser(_b: string): Promise<SetResult> { this.fail(); }
  async getPowerSettings(): Promise<PowerSettings> { this.fail(); }
  async toggleDoNotDisturb(_e: boolean): Promise<SetResult> { this.fail(); }
}

// ─── Mock Bridge ─────────────────────────────────────────────────────────────

export class MockSettingsBridge implements SettingsBridge {
  private displaySleep = 10;
  private volume = 50;
  private browser = 'Safari';
  private dnd = false;

  async getDisplaySettings(): Promise<DisplaySettings> {
    return { sleep_minutes: this.displaySleep, brightness: 75, resolution: '2560x1600' };
  }

  async setDisplaySleep(minutes: number): Promise<SetResult> {
    const prev = this.displaySleep;
    this.displaySleep = minutes;
    return { success: true, previous_value: prev, new_value: minutes };
  }

  async getAudioSettings(): Promise<AudioSettings> {
    return { output_volume: this.volume, input_volume: 80, muted: false, output_device: 'Speakers' };
  }

  async setAudioVolume(volume: number): Promise<SetResult> {
    const prev = this.volume;
    this.volume = volume;
    return { success: true, previous_value: prev, new_value: volume };
  }

  async getDefaultApps(): Promise<DefaultApps> {
    return { browser: this.browser, email: 'Mail', pdf_viewer: 'Preview' };
  }

  async setDefaultBrowser(browser: string): Promise<SetResult> {
    const prev = this.browser;
    this.browser = browser;
    return { success: true, previous_value: prev, new_value: browser };
  }

  async getPowerSettings(): Promise<PowerSettings> {
    return {
      display_sleep_minutes: this.displaySleep,
      system_sleep_minutes: 30,
      disk_sleep_minutes: 10,
      wake_on_network: false,
    };
  }

  async toggleDoNotDisturb(enable: boolean): Promise<SetResult> {
    const prev = this.dnd;
    this.dnd = enable;
    return { success: true, previous_value: prev, new_value: enable };
  }
}

// ─── Bridge Singleton ────────────────────────────────────────────────────────

let activeBridge: SettingsBridge = new UnsupportedPlatformBridge(os.platform());

export function getBridge(): SettingsBridge {
  return activeBridge;
}

export function setBridge(bridge: SettingsBridge): void {
  activeBridge = bridge;
}
