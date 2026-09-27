import { CONFIG } from "./config.js";
import { isObstacle } from "./items.js";

export class Car {
  constructor({ name, driver, side, controller, color, mascot }) {
    this.name = name;
    this.driver = driver;
    this.side = side;
    this.controller = controller;
    this.color = color;
    this.mascot = mascot;
    this.laneIndex = 1;
    this.targetLaneIndex = 1;
    this.y = CONFIG.ROAD_HEIGHT - 78;
    this.distanceTravelled = 0;
    this.maxLives = CONFIG.STARTING_LIVES;
    this.lives = CONFIG.STARTING_LIVES;
    this.crashed = false;
    this.crashTimer = 0;
    this.invulnerableTimer = 0;
    this.spin = 0;
    this.pendingDecision = false;
    this.lastChoice = "center";
    this.lastConfidence = 0;
    this.lastProbabilities = { left: 0, center: 0, right: 0 };
    this.confidenceTotal = 0;
    this.decisionCount = 0;
    this.errorCount = 0;
    this.errorMessage = "";
    this.lastDecisionSource = "api";
    this.boostTimer = 0;
    this.slowTimer = 0;
  }

  update(deltaMs, road, raceFrozen = false) {
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
        this.slowTimer = 1200;
      }
      return;
    }

    if (raceFrozen) return;

    const laneDelta = this.targetLaneIndex - this.laneIndex;
    if (laneDelta !== 0) {
      this.laneIndex += Math.sign(laneDelta) * Math.min(Math.abs(laneDelta), deltaMs / 65);
      if (Math.abs(this.targetLaneIndex - this.laneIndex) < 0.06) {
        this.laneIndex = this.targetLaneIndex;
      }
    }

    if (this.boostTimer > 0) this.boostTimer -= deltaMs;
    if (this.slowTimer > 0) this.slowTimer -= deltaMs;
    if (this.invulnerableTimer > 0) this.invulnerableTimer -= deltaMs;
    this.distanceTravelled = road.distanceTravelled;
  }

  setLane(choice) {
    const laneMap = { left: 0, center: 1, right: 2 };
    if (choice in laneMap) {
      this.targetLaneIndex = laneMap[choice];
      this.lastChoice = choice;
    }
  }

  moveBy(direction) {
    this.targetLaneIndex = Math.max(0, Math.min(2, Math.round(this.targetLaneIndex) + direction));
  }

  applyDecision(decision) {
    this.setLane(decision.lane);
    this.lastConfidence = Number.isFinite(decision.confidence) ? decision.confidence : 0;
    this.lastProbabilities = decision.probabilities || this.lastProbabilities;
    this.confidenceTotal += this.lastConfidence;
    this.decisionCount += 1;
    this.lastDecisionSource = decision.source || "api";
    if (this.lastDecisionSource === "api") {
      this.errorCount = 0;
      this.errorMessage = "";
    }
  }

  recordDecisionError(message) {
    this.errorCount += 1;
    this.errorMessage = message || "Decision error";
  }

  checkCollisions(road) {
    if (this.crashed) return null;
    const carLane = Math.round(this.laneIndex);
    for (const item of road.items) {
      if (item.collected || item.lane !== carLane) continue;
      if (Math.abs(item.y - this.y) > 28) continue;

      item.collected = true;
      if (isObstacle(item.type)) {
        if (this.invulnerableTimer > 0) return null;
        this.lives = Math.max(0, this.lives - 1);
        this.crashed = true;
        this.crashTimer = 0;
        return { type: "crash", item };
      }
    }
    return null;
  }

  get averageConfidence() {
    return this.decisionCount ? this.confidenceTotal / this.decisionCount : 0;
  }

  get eliminated() {
    return this.crashed && this.lives <= 0;
  }
}
