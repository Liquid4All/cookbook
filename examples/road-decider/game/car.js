import { CONFIG, LANES } from "../config.js";

export class Car {
  constructor({ name, racer = null, controller, color, mascot }) {
    this.name = name;
    this.racer = racer;
    this.controller = controller;
    this.color = color;
    this.mascot = mascot;
    this.laneIndex = 1;
    this.targetLaneIndex = 1;
    this.y = CONFIG.ROAD_HEIGHT - 78;
    this.maxLives = CONFIG.STARTING_LIVES;
    this.lives = CONFIG.STARTING_LIVES;
    this.crashed = false;
    this.crashTimer = 0;
    this.invulnerableTimer = 0;
    this.slowTimer = 0;
    this.spin = 0;
    this.pendingDecision = false;
    this.lastChoice = "center";
    this.lastConfidence = 0;
    this.confidenceTotal = 0;
    this.decisionCount = 0;
    this.lastError = null;
  }

  update(deltaMs, road) {
    if (this.crashed) {
      this.crashTimer += deltaMs;
      this.spin += deltaMs * 0.015;
      if (this.lives > 0 && this.crashTimer >= CONFIG.RESPAWN_MS) {
        this.crashed = false;
        this.crashTimer = 0;
        this.spin = 0;
        this.laneIndex = 1;
        this.targetLaneIndex = 1;
        this.invulnerableTimer = CONFIG.INVULNERABLE_MS;
        this.slowTimer = CONFIG.RESPAWN_SLOW_MS;
      }
      return;
    }

    const laneDelta = this.targetLaneIndex - this.laneIndex;
    if (laneDelta !== 0) {
      this.laneIndex += Math.sign(laneDelta) * Math.min(Math.abs(laneDelta), deltaMs / 65);
      if (Math.abs(this.targetLaneIndex - this.laneIndex) < 0.06) {
        this.laneIndex = this.targetLaneIndex;
      }
    }

    if (this.slowTimer > 0) this.slowTimer -= deltaMs;
    if (this.invulnerableTimer > 0) this.invulnerableTimer -= deltaMs;
  }

  setLane(choice) {
    const index = LANES.indexOf(choice);
    if (index === -1) return;
    this.targetLaneIndex = index;
    this.lastChoice = choice;
  }

  moveBy(direction) {
    const lane = Math.round(this.targetLaneIndex) + direction;
    this.targetLaneIndex = Math.max(0, Math.min(CONFIG.LANE_COUNT - 1, lane));
  }

  applyDecision(decision) {
    this.setLane(decision.lane);
    this.lastConfidence = Number.isFinite(decision.confidence) ? decision.confidence : 0;
    this.confidenceTotal += this.lastConfidence;
    this.decisionCount += 1;
  }

  checkCollisions(road) {
    if (this.crashed || this.invulnerableTimer > 0) return;
    const carLane = Math.round(this.laneIndex);
    for (const item of road.items) {
      if (item.collected || item.lane !== carLane) continue;
      if (Math.abs(item.y - this.y) > 28) continue;
      item.collected = true;
      this.lives = Math.max(0, this.lives - 1);
      this.crashed = true;
      this.crashTimer = 0;
      return;
    }
  }

  get averageConfidence() {
    return this.decisionCount ? this.confidenceTotal / this.decisionCount : 0;
  }

  get eliminated() {
    return this.crashed && this.lives <= 0;
  }
}
