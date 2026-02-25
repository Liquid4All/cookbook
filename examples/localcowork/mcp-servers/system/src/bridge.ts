/**
 * System Bridge — Abstraction Layer for OS Operations
 *
 * Provides a bridge interface for system-level operations.
 * NodeSystemBridge uses real Node.js APIs where possible and stubs
 * for Tauri-dependent features (screenshot, open application).
 * MockSystemBridge provides predictable values for tests.
 *
 * The active bridge can be swapped via setBridge() for DI/testing.
 */

import * as os from 'os';
import { execSync } from 'child_process';

// ─── Interfaces ──────────────────────────────────────────────────────────────

/** Hardware and OS information */
export interface SystemInfo {
  readonly os: string;
  readonly arch: string;
  readonly cpu: string;
  readonly ram_gb: number;
  readonly gpu?: string;
  readonly npu?: boolean;
}

/** Screen region for screenshot capture */
export interface ScreenRegion {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/** Running process information */
export interface ProcessInfo {
  readonly pid: number;
  readonly name: string;
  readonly cpu_percent: number;
  readonly memory_mb: number;
}

/** Screenshot result */
export interface ScreenshotResult {
  readonly path: string;
  readonly width: number;
  readonly height: number;
}

/** Application open result */
export interface OpenAppResult {
  readonly success: boolean;
  readonly pid?: number;
}

/** File open result */
export interface OpenFileResult {
  readonly success: boolean;
}

/** Memory usage information */
export interface MemoryUsage {
  readonly total_gb: number;
  readonly used_gb: number;
  readonly free_gb: number;
  readonly swap_total_gb: number;
  readonly swap_used_gb: number;
  readonly usage_percent: number;
}

/** Disk volume information */
export interface DiskVolume {
  readonly mount_point: string;
  readonly filesystem: string;
  readonly total_gb: number;
  readonly used_gb: number;
  readonly free_gb: number;
  readonly usage_percent: number;
}

/** CPU usage information */
export interface CpuUsage {
  readonly cores: number;
  readonly model: string;
  readonly load_average: readonly number[];
  readonly per_core_percent: readonly number[];
}

/** Network interface information */
export interface NetworkInterface {
  readonly name: string;
  readonly ip4?: string;
  readonly ip6?: string;
  readonly mac?: string;
  readonly internal: boolean;
}

/** Kill process result */
export interface KillProcessResult {
  readonly success: boolean;
  readonly pid: number;
  readonly signal: string;
}

/** System bridge contract — all OS operations go through this */
export interface SystemBridge {
  getSystemInfo(): Promise<SystemInfo>;
  openApplication(appName: string): Promise<OpenAppResult>;
  takeScreenshot(region?: ScreenRegion): Promise<ScreenshotResult>;
  listProcesses(filter?: string): Promise<ProcessInfo[]>;
  openFileWith(path: string, app?: string): Promise<OpenFileResult>;
  getMemoryUsage(): Promise<MemoryUsage>;
  getDiskUsage(): Promise<DiskVolume[]>;
  getCpuUsage(): Promise<CpuUsage>;
  getNetworkInfo(): Promise<NetworkInterface[]>;
  killProcess(pid: number, signal?: string): Promise<KillProcessResult>;
}

// ─── NodeSystemBridge ────────────────────────────────────────────────────────

/**
 * Production bridge using Node.js os module and child_process.
 * Real data for getSystemInfo and listProcesses; stubs for Tauri-dependent ops.
 */
export class NodeSystemBridge implements SystemBridge {
  async getSystemInfo(): Promise<SystemInfo> {
    const cpus = os.cpus();
    const cpuModel = cpus.length > 0 ? cpus[0].model : 'Unknown CPU';
    const ramGb = Math.round((os.totalmem() / (1024 * 1024 * 1024)) * 10) / 10;

    return {
      os: os.platform(),
      arch: os.arch(),
      cpu: cpuModel,
      ram_gb: ramGb,
    };
  }

  async openApplication(appName: string): Promise<OpenAppResult> {
    // Stub — actual implementation will use Tauri IPC
    return {
      success: true,
      pid: Math.floor(Math.random() * 65535),
    };
  }

  async takeScreenshot(region?: ScreenRegion): Promise<ScreenshotResult> {
    // Stub — actual implementation will use Tauri IPC
    const timestamp = Date.now();
    const width = region?.width ?? 1920;
    const height = region?.height ?? 1080;

    return {
      path: `/tmp/screenshot-${timestamp}.png`,
      width,
      height,
    };
  }

  async listProcesses(filter?: string): Promise<ProcessInfo[]> {
    const platform = os.platform();

    if (platform !== 'darwin' && platform !== 'linux') {
      // Return mock data on unsupported platforms
      return this.getMockProcesses(filter);
    }

    try {
      const output = execSync('ps -eo pid,pcpu,rss,comm', {
        encoding: 'utf-8',
        timeout: 5000,
      });

      const lines = output.trim().split('\n');
      // Skip the header line
      const processes: ProcessInfo[] = [];

      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.length === 0) continue;

        const parts = line.split(/\s+/);
        if (parts.length < 4) continue;

        const pid = parseInt(parts[0], 10);
        const cpuPercent = parseFloat(parts[1]);
        const rssKb = parseInt(parts[2], 10);
        const name = parts.slice(3).join(' ');

        if (isNaN(pid) || isNaN(cpuPercent) || isNaN(rssKb)) continue;

        const proc: ProcessInfo = {
          pid,
          name,
          cpu_percent: cpuPercent,
          memory_mb: Math.round((rssKb / 1024) * 10) / 10,
        };

        if (filter) {
          if (name.toLowerCase().includes(filter.toLowerCase())) {
            processes.push(proc);
          }
        } else {
          processes.push(proc);
        }
      }

      return processes;
    } catch {
      return this.getMockProcesses(filter);
    }
  }

  async openFileWith(filePath: string, app?: string): Promise<OpenFileResult> {
    // Stub — actual implementation will use Tauri IPC
    return { success: true };
  }

  async getMemoryUsage(): Promise<MemoryUsage> {
    const totalBytes = os.totalmem();
    const freeBytes = os.freemem();
    const usedBytes = totalBytes - freeBytes;
    const toGb = (b: number): number => Math.round((b / (1024 * 1024 * 1024)) * 100) / 100;

    // Swap detection: platform-specific
    let swapTotal = 0;
    let swapUsed = 0;
    try {
      if (os.platform() === 'darwin') {
        const out = execSync('sysctl -n vm.swapusage', { encoding: 'utf-8', timeout: 5000 });
        const totalMatch = out.match(/total\s*=\s*([\d.]+)M/);
        const usedMatch = out.match(/used\s*=\s*([\d.]+)M/);
        if (totalMatch) swapTotal = parseFloat(totalMatch[1]) / 1024;
        if (usedMatch) swapUsed = parseFloat(usedMatch[1]) / 1024;
      } else if (os.platform() === 'win32') {
        const out = execSync('wmic os get TotalVirtualMemorySize,FreeVirtualMemory /value', {
          encoding: 'utf-8', timeout: 5000,
        });
        const tvmMatch = out.match(/TotalVirtualMemorySize=(\d+)/);
        const fvmMatch = out.match(/FreeVirtualMemory=(\d+)/);
        if (tvmMatch && fvmMatch) {
          swapTotal = parseInt(tvmMatch[1], 10) / (1024 * 1024);
          swapUsed = swapTotal - parseInt(fvmMatch[1], 10) / (1024 * 1024);
        }
      }
    } catch { /* swap info optional */ }

    return {
      total_gb: toGb(totalBytes),
      used_gb: toGb(usedBytes),
      free_gb: toGb(freeBytes),
      swap_total_gb: Math.round(swapTotal * 100) / 100,
      swap_used_gb: Math.round(swapUsed * 100) / 100,
      usage_percent: Math.round((usedBytes / totalBytes) * 1000) / 10,
    };
  }

  async getDiskUsage(): Promise<DiskVolume[]> {
    const platform = os.platform();
    const volumes: DiskVolume[] = [];

    try {
      if (platform === 'win32') {
        const out = execSync(
          'wmic logicaldisk get DeviceID,FileSystem,FreeSpace,Size /value',
          { encoding: 'utf-8', timeout: 10000 },
        );
        const blocks = out.split(/\n\n+/).filter((b) => b.includes('DeviceID'));
        for (const block of blocks) {
          const devMatch = block.match(/DeviceID=(.+)/);
          const fsMatch = block.match(/FileSystem=(.+)/);
          const freeMatch = block.match(/FreeSpace=(\d+)/);
          const sizeMatch = block.match(/Size=(\d+)/);
          if (devMatch && sizeMatch && freeMatch) {
            const totalBytes = parseInt(sizeMatch[1], 10);
            const freeBytes = parseInt(freeMatch[1], 10);
            const toGb = (b: number): number => Math.round((b / (1024 ** 3)) * 100) / 100;
            volumes.push({
              mount_point: devMatch[1].trim(),
              filesystem: fsMatch?.[1]?.trim() ?? 'unknown',
              total_gb: toGb(totalBytes),
              used_gb: toGb(totalBytes - freeBytes),
              free_gb: toGb(freeBytes),
              usage_percent: Math.round(((totalBytes - freeBytes) / totalBytes) * 1000) / 10,
            });
          }
        }
      } else {
        const out = execSync('df -k', { encoding: 'utf-8', timeout: 5000 });
        const lines = out.trim().split('\n').slice(1);
        for (const line of lines) {
          const parts = line.trim().split(/\s+/);
          if (parts.length < 6) continue;
          const mountPoint = parts[parts.length - 1];
          if (mountPoint.startsWith('/dev') || mountPoint.startsWith('/System')) continue;
          const totalKb = parseInt(parts[1], 10);
          const usedKb = parseInt(parts[2], 10);
          const freeKb = parseInt(parts[3], 10);
          if (isNaN(totalKb) || totalKb === 0) continue;
          const toGb = (kb: number): number => Math.round((kb / (1024 * 1024)) * 100) / 100;
          volumes.push({
            mount_point: mountPoint,
            filesystem: parts[0],
            total_gb: toGb(totalKb),
            used_gb: toGb(usedKb),
            free_gb: toGb(freeKb),
            usage_percent: Math.round((usedKb / totalKb) * 1000) / 10,
          });
        }
      }
    } catch { /* return empty if df fails */ }

    return volumes;
  }

  async getCpuUsage(): Promise<CpuUsage> {
    const cpus = os.cpus();
    const model = cpus.length > 0 ? cpus[0].model : 'Unknown';
    const loadAvg = os.loadavg();
    const perCore = cpus.map((cpu) => {
      const total = Object.values(cpu.times).reduce((a, b) => a + b, 0);
      const idle = cpu.times.idle;
      return Math.round(((total - idle) / total) * 1000) / 10;
    });

    return {
      cores: cpus.length,
      model,
      load_average: loadAvg,
      per_core_percent: perCore,
    };
  }

  async getNetworkInfo(): Promise<NetworkInterface[]> {
    const ifaces = os.networkInterfaces();
    const result: NetworkInterface[] = [];

    for (const [name, addrs] of Object.entries(ifaces)) {
      if (!addrs) continue;
      const iface: NetworkInterface = {
        name,
        internal: addrs.some((a) => a.internal),
        ip4: addrs.find((a) => a.family === 'IPv4')?.address,
        ip6: addrs.find((a) => a.family === 'IPv6')?.address,
        mac: addrs.find((a) => a.mac !== '00:00:00:00:00:00')?.mac,
      };
      result.push(iface);
    }

    return result;
  }

  async killProcess(pid: number, signal?: string): Promise<KillProcessResult> {
    const sig = signal ?? 'SIGTERM';
    try {
      process.kill(pid, sig as NodeJS.Signals);
      return { success: true, pid, signal: sig };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`Failed to kill process ${pid}: ${msg}`);
    }
  }

  /** Fallback mock processes when ps is unavailable */
  private getMockProcesses(filter?: string): ProcessInfo[] {
    const mocks: ProcessInfo[] = [
      { pid: 1, name: 'init', cpu_percent: 0.0, memory_mb: 10.0 },
      { pid: 100, name: 'node', cpu_percent: 2.5, memory_mb: 150.0 },
      { pid: 200, name: 'python3', cpu_percent: 1.2, memory_mb: 80.0 },
    ];

    if (filter) {
      return mocks.filter((p) =>
        p.name.toLowerCase().includes(filter.toLowerCase()),
      );
    }

    return mocks;
  }
}

// ─── MockSystemBridge ────────────────────────────────────────────────────────

/** Fully mocked bridge for unit tests — returns predictable values */
export class MockSystemBridge implements SystemBridge {
  async getSystemInfo(): Promise<SystemInfo> {
    return {
      os: 'darwin',
      arch: 'arm64',
      cpu: 'Apple M2 Pro',
      ram_gb: 16.0,
      gpu: 'Apple M2 Pro GPU',
      npu: true,
    };
  }

  async openApplication(appName: string): Promise<OpenAppResult> {
    return {
      success: true,
      pid: 12345,
    };
  }

  async takeScreenshot(region?: ScreenRegion): Promise<ScreenshotResult> {
    const width = region?.width ?? 1920;
    const height = region?.height ?? 1080;

    return {
      path: '/tmp/screenshot-mock.png',
      width,
      height,
    };
  }

  async listProcesses(filter?: string): Promise<ProcessInfo[]> {
    const processes: ProcessInfo[] = [
      { pid: 1, name: 'launchd', cpu_percent: 0.1, memory_mb: 12.5 },
      { pid: 256, name: 'Finder', cpu_percent: 1.0, memory_mb: 85.3 },
      { pid: 512, name: 'node', cpu_percent: 5.2, memory_mb: 210.0 },
      { pid: 768, name: 'python3', cpu_percent: 3.1, memory_mb: 145.7 },
      { pid: 1024, name: 'Safari', cpu_percent: 8.4, memory_mb: 520.0 },
    ];

    if (filter) {
      return processes.filter((p) =>
        p.name.toLowerCase().includes(filter.toLowerCase()),
      );
    }

    return processes;
  }

  async openFileWith(filePath: string, app?: string): Promise<OpenFileResult> {
    return { success: true };
  }

  async getMemoryUsage(): Promise<MemoryUsage> {
    return {
      total_gb: 16.0,
      used_gb: 10.5,
      free_gb: 5.5,
      swap_total_gb: 4.0,
      swap_used_gb: 1.2,
      usage_percent: 65.6,
    };
  }

  async getDiskUsage(): Promise<DiskVolume[]> {
    return [
      {
        mount_point: '/',
        filesystem: '/dev/disk1s1',
        total_gb: 500.0,
        used_gb: 350.0,
        free_gb: 150.0,
        usage_percent: 70.0,
      },
      {
        mount_point: '/Volumes/Data',
        filesystem: '/dev/disk2s1',
        total_gb: 1000.0,
        used_gb: 600.0,
        free_gb: 400.0,
        usage_percent: 60.0,
      },
    ];
  }

  async getCpuUsage(): Promise<CpuUsage> {
    return {
      cores: 10,
      model: 'Apple M2 Pro',
      load_average: [2.5, 3.1, 2.8],
      per_core_percent: [15.2, 22.1, 8.5, 45.3, 12.0, 5.8, 30.2, 18.7, 10.1, 25.4],
    };
  }

  async getNetworkInfo(): Promise<NetworkInterface[]> {
    return [
      { name: 'lo0', internal: true, ip4: '127.0.0.1', ip6: '::1' },
      { name: 'en0', internal: false, ip4: '192.168.1.42', mac: 'aa:bb:cc:dd:ee:ff' },
    ];
  }

  async killProcess(pid: number, signal?: string): Promise<KillProcessResult> {
    return { success: true, pid, signal: signal ?? 'SIGTERM' };
  }
}

// ─── Bridge Singleton ────────────────────────────────────────────────────────

let activeBridge: SystemBridge = new NodeSystemBridge();

/** Get the active system bridge */
export function getBridge(): SystemBridge {
  return activeBridge;
}

/** Set the active system bridge (for DI/testing) */
export function setBridge(bridge: SystemBridge): void {
  activeBridge = bridge;
}
