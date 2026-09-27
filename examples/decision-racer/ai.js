import { CONFIG, LANES } from "./config.js";

const QUESTION = {
  lane: {
    type: "choice",
    instructions:
      "Pick the safest lane. Row 1 is most urgent. Prefer the current lane when options are equally safe.",
    criteria: {
      left: "Left lane",
      center: "Center lane",
      right: "Right lane",
    },
  },
};

export function buildState(car, road) {
  const currentLane = LANES[Math.round(car.laneIndex)];
  const rows = 5;
  const lookAhead = road.getLookAhead(car.y, rows);

  const laneSummaries = LANES.map((lane) => {
    const firstObstacle = lookAhead[lane].findIndex((item) => item !== null);
    if (firstObstacle === -1) return `${lane}: clear`;
    return `${lane}: obstacle at row ${firstObstacle + 1}`;
  });

  return [
    `Current lane: ${currentLane}. Pick the lane with the most room ahead.`,
    "",
    ...laneSummaries,
  ].join("\n");
}

export function requestDecision({ provider, model, state }) {
  const headers = { "Content-Type": "application/json" };

  return fetch(`/api/decision/${provider}`, {
    method: "POST",
    headers,
    body: JSON.stringify({ model, state, questions: QUESTION }),
  })
    .then(async (response) => {
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(formatApiError(data, response.status));
      }
      return normalizeDecision(data);
    });
}

export function getHeuristicDecision(car, road) {
  const lookAhead = road.getLookAhead(car.y, CONFIG.LOOK_AHEAD_ROWS);
  const weights = [10, 6, 3, 1.5, 0.5];
  const scores = LANES.map((lane, laneIndex) => {
    let score = -Math.abs(laneIndex - Math.round(car.laneIndex)) * 0.7;
    lookAhead[lane].forEach((item, row) => {
      if (!item) return;
      const weight = weights[row] || 1;
      score -= 10 * weight;
    });
    return score;
  });
  const max = Math.max(...scores);
  const exp = scores.map((score) => Math.exp(score - max));
  const sum = exp.reduce((total, value) => total + value, 0);
  const probabilities = Object.fromEntries(LANES.map((lane, index) => [lane, exp[index] / sum]));
  const lane = LANES[scores.indexOf(max)];

  return {
    lane,
    probabilities,
    confidence: probabilities[lane],
    source: "fallback",
  };
}

function normalizeDecision(data) {
  const answer = data?.answers?.lane || data?.answer?.lane || data?.lane;
  const choice = answer?.choice || answer;
  const lane = LANES.includes(choice) ? choice : "center";
  const probabilities = answer?.probabilities || {
    left: lane === "left" ? 1 : 0,
    center: lane === "center" ? 1 : 0,
    right: lane === "right" ? 1 : 0,
  };
  const confidence = Number.isFinite(answer?.confidence)
    ? answer.confidence
    : Number.isFinite(probabilities[lane])
      ? probabilities[lane]
      : 0;

  return { lane, probabilities, confidence, source: "api" };
}

function formatApiError(data, status) {
  const error = data?.error || data?.detail || data?.message;
  if (typeof error === "string") return error;
  if (error?.message) return error.message;
  if (Array.isArray(error)) return error.map((entry) => entry?.message || String(entry)).join("; ");
  return `Decision API returned ${status}`;
}

