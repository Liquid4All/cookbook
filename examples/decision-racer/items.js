import { CONFIG } from "./config.js";

export const OBSTACLE_TYPES = ["traffic-car", "traffic-truck", "traffic-van", "traffic-bus"];

export function mulberry32(seed) {
  return function random() {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function createSpawnPattern(seed, count = 280) {
  const random = mulberry32(seed);
  const pattern = [];

  for (let index = 0; index < count; index += 1) {
    const wave = [];
    const isCluster = random() < 0.14;

    if (isCluster) {
      const openLane = Math.floor(random() * CONFIG.LANE_COUNT);
      for (let lane = 0; lane < CONFIG.LANE_COUNT; lane += 1) {
        if (lane !== openLane && random() < 0.7) wave.push({ lane, type: createItemType(random) });
      }
    } else {
      const lane = Math.floor(random() * CONFIG.LANE_COUNT);
      wave.push({ lane, type: createItemType(random) });
    }

    if (pattern.length > 0) {
      ensureEscapePath(wave, pattern[pattern.length - 1]);
    }

    pattern.push(wave);
  }

  return pattern;
}

function ensureEscapePath(wave, prevWave) {
  if (wave.length === 0) return;
  const prevBlocked = new Set(prevWave.map((e) => e.lane));
  const currBlocked = new Set(wave.map((e) => e.lane));

  if (currBlocked.size >= CONFIG.LANE_COUNT) {
    wave.splice(wave.length - 1, 1);
    return;
  }

  const prevClear = [0, 1, 2].filter((l) => !prevBlocked.has(l));
  const currClear = [0, 1, 2].filter((l) => !currBlocked.has(l));

  const reachable = currClear.some((cl) =>
    prevClear.some((pl) => Math.abs(cl - pl) <= 1),
  );

  if (!reachable) {
    for (const pl of prevClear) {
      for (const target of [pl, Math.max(0, pl - 1), Math.min(2, pl + 1)]) {
        const idx = wave.findIndex((e) => e.lane === target);
        if (idx !== -1) {
          wave.splice(idx, 1);
          return;
        }
      }
    }
  }
}

function createItemType(random) {
  return OBSTACLE_TYPES[Math.floor(random() * OBSTACLE_TYPES.length)];
}

export function isObstacle(type) {
  return OBSTACLE_TYPES.includes(type);
}
