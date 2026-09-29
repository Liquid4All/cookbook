import { COLORS, LANES } from "../config.js";

export function drawHud(ctx, racers) {
  for (const racer of racers) {
    drawRacerHud(ctx, racer);
  }
  updateConfidenceHtml(racers);
}

function drawRacerHud(ctx, { car, labelX }) {
  ctx.textAlign = "left";
  ctx.fillStyle = COLORS.INK;
  ctx.font = "27px 'Press Start 2P', 'Courier New', monospace";
  ctx.fillText(car.name, labelX, 48);
  ctx.fillStyle = COLORS.NEUTRAL_500;
  ctx.font = "21px 'Press Start 2P', 'Courier New', monospace";
  ctx.fillText("LIVES", labelX, 82);
  const heartsX = labelX + ctx.measureText("LIVES").width + 24;
  for (let i = 0; i < car.maxLives; i += 1) {
    ctx.fillStyle = i < car.lives ? COLORS.POSIE : COLORS.NEUTRAL_200;
    drawHeart(ctx, heartsX + i * 30, 74, 20);
  }
}

const confidenceSlots = [
  document.querySelector("#confidence-left"),
  document.querySelector("#confidence-right"),
];

// Rebuild a slot only when its content changes, not on every frame.
function updateConfidenceHtml(racers) {
  racers.forEach(({ car }, index) => {
    const slot = confidenceSlots[index];
    const html = confidenceHtml(car);
    if (slot.dataset.html === html) return;
    slot.dataset.html = html;
    slot.innerHTML = html;
  });
}

function confidenceHtml(car) {
  if (car.controller !== "ai") return "";
  if (car.lastError) {
    return `<span class="conf-error">${escapeHtml(car.name)} error: ${escapeHtml(car.lastError)}</span>`;
  }

  const confidence = car.lastConfidence;
  const width = Math.max(0, Math.min(100, confidence * 100));
  const lanes = LANES.map(
    (lane) => `<span class="${lane === car.lastChoice ? "active" : ""}">${lane.toUpperCase()}</span>`,
  ).join("");

  return `
    <div class="conf-bar-bg">
      <div class="conf-bar-fill" style="width:${width}%;background:${car.color}"></div>
    </div>
    <div class="conf-info">
      <span class="conf-label">Confidence: ${confidence.toFixed(2)}</span>
      <div class="conf-lanes">${lanes}</div>
    </div>
  `;
}

function drawHeart(ctx, cx, cy, size) {
  const w = size;
  const h = size;
  const y = cy - h * 0.45;
  const top = h * 0.3;

  ctx.beginPath();
  ctx.moveTo(cx, y + top);
  ctx.bezierCurveTo(cx, y, cx - w / 2, y, cx - w / 2, y + top);
  ctx.bezierCurveTo(cx - w / 2, y + h * 0.7, cx, y + h * 0.85, cx, y + h);
  ctx.bezierCurveTo(cx, y + h * 0.85, cx + w / 2, y + h * 0.7, cx + w / 2, y + top);
  ctx.bezierCurveTo(cx + w / 2, y, cx, y, cx, y + top);
  ctx.fill();
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (char) => `&#${char.charCodeAt(0)};`);
}

export function renderResults(resultsGrid, winnerTitle, outcome) {
  winnerTitle.textContent = outcome.title;
  resultsGrid.innerHTML = outcome.racers
    .map(
      ({ car }) => `
        <div class="result-card">
          <strong>${car.name}</strong><br />
          Lives left: ${car.lives}<br />
          Avg confidence: ${car.controller === "ai" ? car.averageConfidence.toFixed(2) : "n/a"}
        </div>
      `,
    )
    .join("");
}
