import { CONFIG, LANES } from "../config.js";

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
  const lookAhead = road.getLookAhead(car.y, CONFIG.LOOK_AHEAD_ROWS);

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

export function requestDecision({ racer, state }) {
  return fetch(`/api/decision/${racer}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state, questions: QUESTION }),
  })
    .then(async (response) => {
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const msg = data?.error?.message || data?.error || data?.detail || `API returned ${response.status}`;
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      }
      return normalizeDecision(data);
    });
}

function normalizeDecision(data) {
  const answer = data?.answers?.lane;
  if (!LANES.includes(answer?.choice)) {
    throw new Error("Unexpected response from the decision API.");
  }

  return {
    lane: answer.choice,
    probabilities: answer.probabilities,
    confidence: answer.confidence,
  };
}
