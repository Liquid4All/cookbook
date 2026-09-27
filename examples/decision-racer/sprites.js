import { COLORS, CONFIG } from "./config.js";

export function drawCar(ctx, x, y, car) {
  const size = CONFIG.CAR_SIZE;
  const flash = car.crashed && Math.floor(Math.max(0, car.crashTimer) / 90) % 2 === 0;

  ctx.save();
  ctx.translate(x, y);
  if (!car.crashed && car.invulnerableTimer > 0 && Math.floor(car.invulnerableTimer / 120) % 2 === 0) {
    ctx.globalAlpha = 0.42;
  }
  if (car.crashed) {
    ctx.globalAlpha = 0.5 + 0.5 * Math.abs(Math.sin(car.spin * 3));
  }
  ctx.fillStyle = flash ? COLORS.POSIE : car.color;
  pixelRect(ctx, -15, -16, 30, 34, 3);
  ctx.fillStyle = COLORS.INK;
  pixelRect(ctx, -10, -7, 20, 12, 2);
  ctx.fillStyle = COLORS.PAPER;
  pixelRect(ctx, -7, -12, 14, 6, 2);
  ctx.fillStyle = COLORS.INK;
  pixelRect(ctx, -20, -8, 6, 12, 1);
  pixelRect(ctx, 14, -8, 6, 12, 1);
  pixelRect(ctx, -20, 9, 6, 12, 1);
  pixelRect(ctx, 14, 9, 6, 12, 1);

  if (car.mascot === "droplet") drawDroplet(ctx, 0, -25);
  if (car.mascot === "flame") drawFlame(ctx, 0, -25);
  if (car.boostTimer > 0) drawBoost(ctx);
  ctx.restore();
}

export function drawItem(ctx, x, y, type, frame) {
  ctx.save();
  ctx.translate(x, y);
  if (type === "traffic-car") drawTrafficCar(ctx, COLORS.POSIE);
  if (type === "traffic-truck") drawTrafficCar(ctx, COLORS.SKY);
  if (type === "traffic-van") drawTrafficCar(ctx, COLORS.FOREST);
  if (type === "traffic-bus") drawTrafficBus(ctx);
  ctx.restore();
}

function drawDroplet(ctx, x, y) {
  ctx.fillStyle = COLORS.LILAC;
  pixelRect(ctx, x - 5, y - 2, 10, 12, 2);
  pixelRect(ctx, x - 3, y - 8, 6, 8, 2);
  ctx.fillStyle = COLORS.PAPER;
  pixelRect(ctx, x + 1, y - 2, 2, 4, 1);
}

function drawFlame(ctx, x, y) {
  ctx.fillStyle = COLORS.NEUTRAL_700;
  pixelRect(ctx, x - 6, y - 3, 12, 12, 2);
  pixelRect(ctx, x - 3, y - 10, 6, 10, 1);
  ctx.fillStyle = COLORS.NEUTRAL_200;
  pixelRect(ctx, x - 2, y - 2, 4, 8, 1);
}

function drawTrafficCar(ctx, color) {
  ctx.fillStyle = color;
  pixelRect(ctx, -14, -16, 28, 34);
  pixelRect(ctx, -10, -22, 20, 10);
  ctx.fillStyle = COLORS.PAPER;
  pixelRect(ctx, -7, -15, 14, 7);
  ctx.fillStyle = COLORS.INK;
  pixelRect(ctx, -18, -10, 5, 10);
  pixelRect(ctx, 13, -10, 5, 10);
  pixelRect(ctx, -18, 9, 5, 10);
  pixelRect(ctx, 13, 9, 5, 10);
}

function drawTrafficBus(ctx) {
  ctx.fillStyle = COLORS.DANDELION;
  pixelRect(ctx, -14, -26, 28, 52);
  pixelRect(ctx, -10, -30, 20, 8);
  ctx.fillStyle = COLORS.PAPER;
  pixelRect(ctx, -7, -24, 14, 7);
  pixelRect(ctx, -9, -6, 8, 6);
  pixelRect(ctx, 1, -6, 8, 6);
  ctx.fillStyle = COLORS.INK;
  pixelRect(ctx, -18, -18, 5, 10);
  pixelRect(ctx, 13, -18, 5, 10);
  pixelRect(ctx, -18, 12, 5, 10);
  pixelRect(ctx, 13, 12, 5, 10);
}

function drawBoost(ctx) {
  ctx.fillStyle = COLORS.SKY;
  pixelRect(ctx, -8, 20, 5, 18, 1);
  pixelRect(ctx, 3, 20, 5, 18, 1);
}


function pixelRect(ctx, x, y, width, height, radius = 0) {
  ctx.fillRect(Math.round(x), Math.round(y), Math.round(width), Math.round(height));
}
