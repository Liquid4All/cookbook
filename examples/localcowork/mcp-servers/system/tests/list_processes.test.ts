import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { listProcesses } from '../src/tools/list_processes';
import { setupTestBridge, teardownTestBridge } from './helpers';

describe('system.list_processes', () => {
  beforeAll(() => {
    setupTestBridge();
  });

  afterAll(() => {
    teardownTestBridge();
  });

  it('should list all processes when no filter is provided', async () => {
    const result = await listProcesses.execute({});
    expect(result.success).toBe(true);
    expect(result.data.processes).toBeDefined();
    expect(Array.isArray(result.data.processes)).toBe(true);
    // MockSystemBridge returns 5 processes
    expect(result.data.processes.length).toBe(5);
  });

  it('should filter processes by name', async () => {
    const result = await listProcesses.execute({ filter: 'node' });
    expect(result.success).toBe(true);
    expect(result.data.processes.length).toBe(1);
    expect(result.data.processes[0].name).toBe('node');
  });

  it('should return empty array for non-matching filter', async () => {
    const result = await listProcesses.execute({ filter: 'nonexistentprocess' });
    expect(result.success).toBe(true);
    expect(result.data.processes.length).toBe(0);
  });

  it('should return all processes when filter is undefined', async () => {
    const result = await listProcesses.execute({});
    expect(result.success).toBe(true);
    expect(result.data.processes.length).toBeGreaterThan(0);
  });

  it('should have correct fields on each process: pid, name, cpu_percent, memory_mb', async () => {
    const result = await listProcesses.execute({});
    const proc = result.data.processes[0];

    expect(proc).toHaveProperty('pid');
    expect(proc).toHaveProperty('name');
    expect(proc).toHaveProperty('cpu_percent');
    expect(proc).toHaveProperty('memory_mb');

    expect(typeof proc.pid).toBe('number');
    expect(typeof proc.name).toBe('string');
    expect(typeof proc.cpu_percent).toBe('number');
    expect(typeof proc.memory_mb).toBe('number');
  });

  it('should do case-insensitive filtering', async () => {
    const result = await listProcesses.execute({ filter: 'SAFARI' });
    expect(result.success).toBe(true);
    expect(result.data.processes.length).toBe(1);
    expect(result.data.processes[0].name).toBe('Safari');
  });

  it('has correct metadata', () => {
    expect(listProcesses.name).toBe('system.list_processes');
    expect(listProcesses.confirmationRequired).toBe(false);
    expect(listProcesses.undoSupported).toBe(false);
  });
});
