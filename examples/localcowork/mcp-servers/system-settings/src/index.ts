/**
 * System Settings MCP Server — Entry Point
 *
 * Provides conversational OS settings control for macOS and Windows.
 * All write tools require confirmation. Read tools execute immediately.
 *
 * Tools (8):
 *   system-settings.get_display_settings  — display sleep, brightness
 *   system-settings.set_display_sleep     — set display sleep timer (confirm)
 *   system-settings.get_audio_settings    — volume, mute, device
 *   system-settings.set_audio_volume      — set output volume (confirm)
 *   system-settings.get_default_apps      — browser, email, PDF viewer
 *   system-settings.set_default_browser   — set default browser (confirm)
 *   system-settings.get_power_settings    — sleep timers, wake-on-network
 *   system-settings.toggle_do_not_disturb — enable/disable DND (confirm)
 */

import { MCPServer } from '../../_shared/ts/mcp-base';
import { initBridge } from './bridge';
import { getDisplaySettings } from './tools/get_display_settings';
import { setDisplaySleep } from './tools/set_display_sleep';
import { getAudioSettings } from './tools/get_audio_settings';
import { setAudioVolume } from './tools/set_audio_volume';
import { getDefaultApps } from './tools/get_default_apps';
import { setDefaultBrowser } from './tools/set_default_browser';
import { getPowerSettings } from './tools/get_power_settings';
import { toggleDoNotDisturb } from './tools/toggle_do_not_disturb';

// ─── Server Setup ───────────────────────────────────────────────────────────

const server = new MCPServer({
  name: 'system-settings',
  version: '0.1.0',
  tools: [
    getDisplaySettings, setDisplaySleep,
    getAudioSettings, setAudioVolume,
    getDefaultApps, setDefaultBrowser,
    getPowerSettings, toggleDoNotDisturb,
  ],
});

// ─── Graceful Shutdown ──────────────────────────────────────────────────────

process.on('SIGINT', () => {
  process.exit(0);
});

process.on('SIGTERM', () => {
  process.exit(0);
});

// ─── Start (load platform bridge, then serve) ──────────────────────────────

await initBridge();
server.start();
