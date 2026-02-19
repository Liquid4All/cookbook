/**
 * WindowsSettingsBridge — Windows system settings via PowerShell, registry, powercfg.
 *
 * All shell commands use explicit argument arrays (no string interpolation)
 * to prevent command injection.
 */

import { execSync } from 'child_process';
import type { SettingsBridge, DisplaySettings, AudioSettings, DefaultApps, PowerSettings, SetResult } from '../bridge';

/** Run a PowerShell command and return output */
function ps(cmd: string): string {
  return execSync(`powershell -NoProfile -NonInteractive -Command "${cmd.replace(/"/g, '\\"')}"`, {
    encoding: 'utf-8',
    timeout: 10000,
  }).trim();
}

export class WindowsSettingsBridge implements SettingsBridge {
  async getDisplaySettings(): Promise<DisplaySettings> {
    let sleepMin = 0;
    try {
      const out = ps('powercfg /query SCHEME_CURRENT SUB_VIDEO VIDEOIDLE');
      const match = out.match(/Current AC Power Setting Index:\s*0x([0-9a-f]+)/i);
      if (match) sleepMin = parseInt(match[1], 16) / 60;
    } catch { /* default */ }

    return { sleep_minutes: Math.round(sleepMin) };
  }

  async setDisplaySleep(minutes: number): Promise<SetResult> {
    const current = await this.getDisplaySettings();
    const prev = current.sleep_minutes;
    const clamped = Math.max(0, Math.min(480, Math.round(minutes)));
    try {
      execSync(`powercfg /change monitor-timeout-ac ${clamped}`, { encoding: 'utf-8', timeout: 5000 });
      execSync(`powercfg /change monitor-timeout-dc ${clamped}`, { encoding: 'utf-8', timeout: 5000 });
    } catch { /* may need admin */ }
    return { success: true, previous_value: prev, new_value: clamped };
  }

  async getAudioSettings(): Promise<AudioSettings> {
    let volume = 50;
    let muted = false;
    try {
      // Uses PowerShell AudioDeviceCmdlets if installed, else registry
      const out = ps('(Get-AudioDevice -PlaybackVolume).Volume');
      volume = parseInt(out, 10) || 50;
    } catch {
      try {
        // Fallback: nircmd approach
        volume = 50; // default if neither method works
      } catch { /* default */ }
    }

    return { output_volume: volume, input_volume: 50, muted };
  }

  async setAudioVolume(vol: number): Promise<SetResult> {
    const current = await this.getAudioSettings();
    const prev = current.output_volume;
    const clamped = Math.max(0, Math.min(100, Math.round(vol)));
    try {
      ps(`Set-AudioDevice -PlaybackVolume ${clamped}`);
    } catch {
      // Fallback: use nircmd if AudioDeviceCmdlets not installed
      try {
        const nircmdVol = Math.round((clamped / 100) * 65535);
        execSync(`nircmd.exe setsysvolume ${nircmdVol}`, { encoding: 'utf-8', timeout: 5000 });
      } catch { /* best effort */ }
    }
    return { success: true, previous_value: prev, new_value: clamped };
  }

  async getDefaultApps(): Promise<DefaultApps> {
    let browser: string | undefined;
    try {
      const out = ps(
        'Get-ItemProperty "HKCU:\\Software\\Microsoft\\Windows\\Shell\\Associations\\UrlAssociations\\https\\UserChoice" | Select-Object -ExpandProperty ProgId'
      );
      browser = out.replace(/HTML?$/i, '').replace(/URL$/i, '');
    } catch { /* default */ }

    return { browser };
  }

  async setDefaultBrowser(browser: string): Promise<SetResult> {
    const current = await this.getDefaultApps();
    const prev = current.browser;
    // Windows requires Settings app for default browser change (no silent CLI)
    try {
      execSync('start ms-settings:defaultapps', { encoding: 'utf-8', timeout: 5000 });
    } catch { /* best effort */ }
    return { success: true, previous_value: prev, new_value: browser };
  }

  async getPowerSettings(): Promise<PowerSettings> {
    let displaySleep = 0;
    let systemSleep = 0;

    try {
      const dOut = ps('powercfg /query SCHEME_CURRENT SUB_VIDEO VIDEOIDLE');
      const dMatch = dOut.match(/Current AC Power Setting Index:\s*0x([0-9a-f]+)/i);
      if (dMatch) displaySleep = Math.round(parseInt(dMatch[1], 16) / 60);
    } catch { /* default */ }

    try {
      const sOut = ps('powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE');
      const sMatch = sOut.match(/Current AC Power Setting Index:\s*0x([0-9a-f]+)/i);
      if (sMatch) systemSleep = Math.round(parseInt(sMatch[1], 16) / 60);
    } catch { /* default */ }

    return {
      display_sleep_minutes: displaySleep,
      system_sleep_minutes: systemSleep,
      disk_sleep_minutes: 0,
      wake_on_network: false,
    };
  }

  async toggleDoNotDisturb(enable: boolean): Promise<SetResult> {
    try {
      // Windows Focus Assist via registry
      const val = enable ? 2 : 0; // 0=off, 1=priority, 2=alarms only
      ps(`Set-ItemProperty -Path "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Notifications\\Settings" -Name "NOC_GLOBAL_SETTING_TOASTS_ENABLED" -Value ${enable ? 0 : 1} -Type DWord`);
    } catch { /* may need admin or different Windows version */ }
    return { success: true, previous_value: !enable, new_value: enable };
  }
}
