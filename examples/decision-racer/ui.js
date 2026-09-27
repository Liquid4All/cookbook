import { COLORS, CONFIG, LANES } from "./config.js";

export function drawHud(ctx, state) {
  for (const racer of state.racers) {
    drawRacerHud(ctx, racer);
  }
  updateConfidenceHtml(state.racers);
}

function drawRacerHud(ctx, { car, road, labelX }) {
  ctx.textAlign = "left";
  ctx.fillStyle = COLORS.INK;
  ctx.font = "27px 'Press Start 2P', 'Courier New', monospace";
  ctx.fillText(car.name, labelX, 48);
  drawLives(ctx, labelX, 82, car);

  if (car.errorCount > 0) {
    ctx.fillStyle = COLORS.POSIE;
    ctx.font = "13px 'Press Start 2P', 'Courier New', monospace";
    ctx.fillText(`API fallback`, labelX, 622);
  }
}

function drawLives(ctx, x, y, car) {
  ctx.fillStyle = COLORS.NEUTRAL_500;
  ctx.font = "21px 'Press Start 2P', 'Courier New', monospace";
  ctx.fillText("LIVES", x, y);
  for (let i = 0; i < car.maxLives; i += 1) {
    ctx.beginPath();
    ctx.arc(x + 105 + i * 30, y - 6, 10, 0, Math.PI * 2);
    ctx.fillStyle = i < car.lives ? (car.name === "LFM2.5-S1" ? COLORS.PURPLE : COLORS.INK) : COLORS.NEUTRAL_200;
    ctx.fill();
  }
}

const confidenceSlots = {
  left: document.querySelector("#confidence-left"),
  right: document.querySelector("#confidence-right"),
};

function updateConfidenceHtml(racers) {
  const slots = [confidenceSlots.left, confidenceSlots.right];
  for (let i = 0; i < racers.length; i += 1) {
    const { car } = racers[i];
    const slot = slots[i];
    if (!slot) continue;

    if (car.controller !== "ai") {
      slot.innerHTML = "";
      continue;
    }

    const conf = car.lastConfidence;
    const barColor = car.name === "Liquid" ? COLORS.PURPLE : COLORS.NEUTRAL_700;
    const source = car.lastDecisionSource === "fallback" ? "Fallback" : "Confidence";

    slot.innerHTML = `
      <span class="conf-label">${source}: ${conf.toFixed(2)}</span>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:${Math.max(0, Math.min(100, conf * 100))}%;background:${barColor}"></div>
      </div>
      <div class="conf-lanes">${LANES.map(
        (lane) => `<span class="${lane === car.lastChoice ? "active" : ""}">${lane.toUpperCase()}</span>`,
      ).join("")}</div>
    `;
  }
}

function truncate(value, maxLength) {
  if (!value) return "decision unavailable";
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}…` : value;
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
