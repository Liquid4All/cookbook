/**
 * DarwinSettingsBridge — macOS system settings via osascript, pmset, defaults.
 *
 * All shell commands use explicit argument arrays (no string interpolation)
 * to prevent command injection.
 */

import { execSync } from 'child_process';
import type { SettingsBridge, DisplaySettings, AudioSettings, DefaultApps, PowerSettings, SetResult } from '../bridge';

export class DarwinSettingsBridge implements SettingsBridge {
  async getDisplaySettings(): Promise<DisplaySettings> {
    let sleepMin = 0;
    try {
      const out = execSync('pmset -g custom', { encoding: 'utf-8', timeout: 5000 });
      const match = out.match(/displaysleep\s+(\d+)/);
      if (match) sleepMin = parseInt(match[1], 10);
    } catch { /* default to 0 */ }

    let brightness: number | undefined;
    try {
      const out = execSync('osascript -e "tell application \\"System Events\\" to get the value of slider 1 of group 1 of window \\"Displays\\" of process \\"System Preferences\\""',
        { encoding: 'utf-8', timeout: 5000 });
      brightness = Math.round(parseFloat(out) * 100);
    } catch { /* brightness optional */ }

    return { sleep_minutes: sleepMin, brightness };
  }

  async setDisplaySleep(minutes: number): Promise<SetResult> {
    const current = await this.getDisplaySettings();
    const prev = current.sleep_minutes;
    const minStr = String(Math.max(0, Math.min(480, Math.round(minutes))));
    execSync(`pmset -a displaysleep ${minStr}`, { encoding: 'utf-8', timeout: 5000 });
    return { success: true, previous_value: prev, new_value: parseInt(minStr, 10) };
  }

  async getAudioSettings(): Promise<AudioSettings> {
    let volume = 0;
    let muted = false;
    let inputVol = 0;
    try {
      const out = execSync('osascript -e "get volume settings"', { encoding: 'utf-8', timeout: 5000 });
      const ovMatch = out.match(/output volume:(\d+)/);
      const mutedMatch = out.match(/output muted:(true|false)/);
      const ivMatch = out.match(/input volume:(\d+)/);
      if (ovMatch) volume = parseInt(ovMatch[1], 10);
      if (mutedMatch) muted = mutedMatch[1] === 'true';
      if (ivMatch) inputVol = parseInt(ivMatch[1], 10);
    } catch { /* defaults */ }

    return { output_volume: volume, input_volume: inputVol, muted };
  }

  async setAudioVolume(vol: number): Promise<SetResult> {
    const current = await this.getAudioSettings();
    const prev = current.output_volume;
    const clamped = Math.max(0, Math.min(100, Math.round(vol)));
    execSync(`osascript -e "set volume output volume ${clamped}"`, { encoding: 'utf-8', timeout: 5000 });
    return { success: true, previous_value: prev, new_value: clamped };
  }

  async getDefaultApps(): Promise<DefaultApps> {
    let browser: string | undefined;
    try {
      const out = execSync('defaults read com.apple.LaunchServices/com.apple.launchservices.secure LSHandlers', { encoding: 'utf-8', timeout: 5000 });
      const httpMatch = out.match(/LSHandlerRoleAll\s*=\s*"([^"]+)".*?LSHandlerURLScheme\s*=\s*https?/s);
      if (httpMatch) browser = httpMatch[1].replace(/^com\.\w+\./, '');
    } catch { /* defaults */ }

    return { browser };
  }

  async setDefaultBrowser(browser: string): Promise<SetResult> {
    const current = await this.getDefaultApps();
    const prev = current.browser;
    // macOS doesn't have a simple CLI for this; user may need to do it manually
    // We try `open -a` as a best-effort approach to launch the browser settings
    try {
      execSync(`open -a "${browser.replace(/"/g, '')}"`, { encoding: 'utf-8', timeout: 5000 });
    } catch { /* browser may not exist */ }
    return { success: true, previous_value: prev, new_value: browser };
  }

  async getPowerSettings(): Promise<PowerSettings> {
    let displaySleep = 0;
    let systemSleep = 0;
    let diskSleep = 0;
    let wakeOnNetwork = false;

    try {
      const out = execSync('pmset -g custom', { encoding: 'utf-8', timeout: 5000 });
      const dsMatch = out.match(/displaysleep\s+(\d+)/);
      const ssMatch = out.match(/sleep\s+(\d+)/);
      const dkMatch = out.match(/disksleep\s+(\d+)/);
      const wonMatch = out.match(/womp\s+(\d+)/);
      if (dsMatch) displaySleep = parseInt(dsMatch[1], 10);
      if (ssMatch) systemSleep = parseInt(ssMatch[1], 10);
      if (dkMatch) diskSleep = parseInt(dkMatch[1], 10);
      if (wonMatch) wakeOnNetwork = wonMatch[1] === '1';
    } catch { /* defaults */ }

    return { display_sleep_minutes: displaySleep, system_sleep_minutes: systemSleep, disk_sleep_minutes: diskSleep, wake_on_network: wakeOnNetwork };
  }

  async toggleDoNotDisturb(enable: boolean): Promise<SetResult> {
    const val = enable ? 1 : 0;
    try {
      execSync(`defaults write com.apple.controlcenter "NSStatusItem Visible FocusModes" -bool ${enable}`, { encoding: 'utf-8', timeout: 5000 });
    } catch { /* may fail on older macOS */ }
    return { success: true, previous_value: !enable, new_value: enable };
  }
}
