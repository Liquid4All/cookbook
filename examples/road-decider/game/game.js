import { requestDecision, buildState } from "../ai/ai.js";
import { Car } from "./car.js";
import { COLORS, CONFIG, MODES } from "../config.js";
import { Road } from "./road.js";
import { drawCar, drawItem } from "../ui/sprites.js";
import { drawHud } from "../ui/ui.js";

export class Game {
  constructor({ canvas, input, onComplete }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.input = input;
    this.onComplete = onComplete;
    this.animationId = null;
    this.lastTime = 0;
  }

  start(mode) {
    this.stop();
    const seed = Date.now() & 0xffffffff;
    this.mode = mode;
    this.elapsedMs = 0;
    this.lastSpeedStep = 0;
    this.speedFactor = 1;
    this.ended = false;

    this.roads = CONFIG.ROAD_X.map((x) => new Road({ x, y: CONFIG.ROAD_Y, seed }));

    this.cars = mode === MODES.HUMAN
      ? [
          new Car({ name: "d1", racer: "d1", controller: "ai", color: COLORS.PURPLE, mascot: null }),
          new Car({ name: "You", controller: "human", color: COLORS.SKY, mascot: null }),
        ]
      : [
          new Car({ name: "d1", racer: "d1", controller: "ai", color: COLORS.PURPLE, mascot: null }),
          new Car({ name: "Jev", racer: "jev", controller: "ai", color: COLORS.NEUTRAL_700, mascot: "flame" }),
        ];

    this.lastTime = performance.now();
    this.animationId = requestAnimationFrame((time) => this.loop(time));
  }

  stop() {
    if (this.animationId) cancelAnimationFrame(this.animationId);
    this.animationId = null;
  }

  loop(time) {
    const deltaMs = Math.min(50, time - this.lastTime);
    this.lastTime = time;
    this.update(deltaMs);
    this.render();

    if (!this.ended) {
      this.animationId = requestAnimationFrame((nextTime) => this.loop(nextTime));
    }
  }

  update(deltaMs) {
    this.elapsedMs += deltaMs;
    const speedStep = Math.floor(this.elapsedMs / CONFIG.SPEED_INTERVAL_MS);
    if (speedStep !== this.lastSpeedStep) {
      this.speedFactor = Math.min(CONFIG.MAX_SPEED_FACTOR, 1 + speedStep * CONFIG.SPEED_INCREMENT);
      this.lastSpeedStep = speedStep;
    }

    const humanMove = this.input.consumeLaneMove();
    if (humanMove) {
      const human = this.cars.find((car) => car.controller === "human");
      if (human && !human.crashed) human.moveBy(humanMove);
    }

    for (let index = 0; index < this.roads.length; index += 1) {
      const car = this.cars[index];
      const road = this.roads[index];
      const frozen = car.crashed;
      // Ease back up to full speed after a respawn.
      const slowFactor = car.slowTimer > 0 ? 0.55 + 0.45 * (1 - car.slowTimer / CONFIG.RESPAWN_SLOW_MS) : 1;
      road.update(deltaMs, this.speedFactor * slowFactor, frozen);
      car.update(deltaMs, road);
      this.maybeRequestDecision(car, road);
      car.checkCollisions(road);
    }

    this.checkRaceEnd();
  }

  maybeRequestDecision(car, road) {
    if (car.controller !== "ai" || car.crashed || car.pendingDecision) return;
    const tickMs = Math.max(
      CONFIG.MIN_TICK_MS,
      CONFIG.INITIAL_TICK_MS - Math.floor(this.elapsedMs / CONFIG.TICK_SPEEDUP_INTERVAL_MS) * 80,
    );

    car.nextDecisionAt ??= 0;
    if (this.elapsedMs < car.nextDecisionAt) return;
    car.nextDecisionAt = this.elapsedMs + tickMs;
    car.pendingDecision = true;

    requestDecision({ racer: car.racer, state: buildState(car, road) })
      .then((decision) => {
        car.applyDecision(decision);
        car.lastError = null;
      })
      .catch((error) => {
        car.lastError = error.message;
        console.warn(`${car.name} decision failed`, error);
      })
      .finally(() => { car.pendingDecision = false; });
  }

  checkRaceEnd() {
    if (this.cars.some((car) => car.eliminated)) {
      this.ended = true;
      setTimeout(() => this.onComplete(this.getOutcome()), 650);
    }
  }

  getOutcome() {
    const racers = this.cars.map((car, index) => ({ car, road: this.roads[index] }));
    const [left, right] = racers;
    const tied = left.car.eliminated === right.car.eliminated;
    const winner = left.car.eliminated ? right : left;

    return {
      title: tied ? "Photo finish!" : winner.car.name === "You" ? "You win!" : `${winner.car.name} wins!`,
      racers,
    };
  }

  // Draw the empty roads shown behind the start screen.
  renderIdle() {
    this.ctx.fillStyle = COLORS.PAPER;
    this.ctx.fillRect(0, 0, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT);
    for (const x of CONFIG.ROAD_X) {
      this.drawRoad(new Road({ x, y: CONFIG.ROAD_Y, seed: 0 }));
    }
  }

  render() {
    const ctx = this.ctx;
    ctx.fillStyle = COLORS.PAPER;
    ctx.fillRect(0, 0, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT);

    for (let index = 0; index < this.roads.length; index += 1) {
      const road = this.roads[index];
      const car = this.cars[index];
      this.drawRoad(road);
      ctx.save();
      ctx.beginPath();
      ctx.rect(road.x - CONFIG.SHOULDER_WIDTH, road.y, road.width + 2 * CONFIG.SHOULDER_WIDTH, road.height);
      ctx.clip();
      this.drawItems(road);
      drawCar(ctx, road.laneCenter(car.laneIndex), road.y + car.y, car);
      ctx.restore();
      if (car.crashed) this.drawCrashLabel(road, car);
    }

    ctx.strokeStyle = COLORS.NEUTRAL_200;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(CONFIG.CANVAS_WIDTH / 2, 26);
    ctx.lineTo(CONFIG.CANVAS_WIDTH / 2, 680);
    ctx.stroke();

    drawHud(ctx, [
      { car: this.cars[0], labelX: 80 },
      { car: this.cars[1], labelX: 600 },
    ]);
  }

  drawRoad(road) {
    const ctx = this.ctx;
    ctx.fillStyle = COLORS.NEUTRAL_100;
    ctx.fillRect(road.x - CONFIG.SHOULDER_WIDTH, road.y, road.width + 2 * CONFIG.SHOULDER_WIDTH, road.height);
    ctx.fillStyle = COLORS.NEUTRAL_50;
    ctx.fillRect(road.x, road.y, road.width, road.height);
    ctx.strokeStyle = COLORS.INK;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(road.x, road.y, road.width, road.height);

    for (let lane = 1; lane < CONFIG.LANE_COUNT; lane += 1) {
      const x = road.x + lane * CONFIG.LANE_WIDTH;
      ctx.strokeStyle = COLORS.NEUTRAL_200;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([22, 20]);
      ctx.lineDashOffset = road.lineOffset;
      ctx.beginPath();
      ctx.moveTo(x, road.y);
      ctx.lineTo(x, road.y + road.height);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  drawItems(road) {
    for (const item of road.items) {
      if (item.collected) continue;
      drawItem(this.ctx, road.laneCenter(item.lane), road.y + item.y, item.type);
    }
  }

  drawCrashLabel(road, car) {
    if (car.crashTimer < 160) return;
    this.ctx.fillStyle = "rgba(7, 7, 13, 0.92)";
    this.ctx.fillRect(road.x + 30, road.y + 232, road.width - 60, 58);
    this.ctx.strokeStyle = COLORS.INK;
    this.ctx.strokeRect(road.x + 30, road.y + 232, road.width - 60, 58);
    this.ctx.fillStyle = car.eliminated ? COLORS.INK : COLORS.POSIE;
    this.ctx.font = "22px 'Press Start 2P', 'Courier New', monospace";
    this.ctx.textAlign = "center";
    this.ctx.fillText(car.eliminated ? "Out of lives" : "Life lost", road.x + road.width / 2, road.y + 268);
  }
}
