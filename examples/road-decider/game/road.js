import { CONFIG, LANES } from "../config.js";
import { createSpawnPattern } from "./items.js";

export class Road {
  constructor({ x, y, seed }) {
    this.x = x;
    this.y = y;
    this.width = CONFIG.ROAD_WIDTH;
    this.height = CONFIG.ROAD_HEIGHT;
    this.scrollSpeed = CONFIG.INITIAL_SCROLL_SPEED;
    this.items = [];
    this.pattern = createSpawnPattern(seed);
    this.patternIndex = 0;
    this.spawnTimer = 0;
    this.lineOffset = 0;
  }

  update(deltaMs, speedFactor, frozen = false) {
    this.scrollSpeed = Math.min(CONFIG.MAX_SCROLL_SPEED, CONFIG.INITIAL_SCROLL_SPEED * speedFactor);
    if (frozen) return;

    const deltaSeconds = deltaMs / 1000;
    this.lineOffset = (this.lineOffset + this.scrollSpeed * deltaSeconds) % 64;

    for (const item of this.items) {
      item.y += this.scrollSpeed * deltaSeconds;
    }

    this.items = this.items.filter((item) => !item.collected && item.y < this.height + 70);
    this.spawnTimer -= deltaMs;

    if (this.spawnTimer <= 0) {
      this.spawn();
      const interval = Math.max(
        CONFIG.MIN_SPAWN_INTERVAL_MS,
        CONFIG.INITIAL_SPAWN_INTERVAL_MS / Math.max(1, speedFactor * 0.92),
      );
      this.spawnTimer = interval;
    }
  }

  spawn() {
    const wave = this.pattern[this.patternIndex % this.pattern.length];
    this.patternIndex += 1;
    for (const entry of wave) {
      this.items.push({
        lane: entry.lane,
        type: entry.type,
        y: -CONFIG.ITEM_SIZE - 8,
        collected: false,
      });
    }
  }

  laneCenter(laneIndex) {
    return this.x + laneIndex * CONFIG.LANE_WIDTH + CONFIG.LANE_WIDTH / 2;
  }

  getLookAhead(carY, rows = CONFIG.LOOK_AHEAD_ROWS) {
    const rowHeight = CONFIG.LOOK_AHEAD_ROW_HEIGHT;
    const lookAhead = Object.fromEntries(LANES.map((lane) => [lane, Array(rows).fill(null)]));

    for (const item of this.items) {
      if (item.collected || item.y >= carY) continue;
      const distance = carY - item.y;
      const row = Math.floor(distance / rowHeight);
      if (row >= 0 && row < rows) {
        lookAhead[LANES[item.lane]][row] ||= item;
      }
    }

    return lookAhead;
  }
}
