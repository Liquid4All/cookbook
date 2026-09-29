export const CONFIG = {
  STARTING_LIVES: 3,
  RESPAWN_MS: 900,
  INVULNERABLE_MS: 1200,
  RESPAWN_SLOW_MS: 1200,
  LANE_COUNT: 3,
  LOOK_AHEAD_ROWS: 5,
  LOOK_AHEAD_ROW_HEIGHT: 108,
  INITIAL_SCROLL_SPEED: 180,
  SPEED_INCREMENT: 0.15,
  SPEED_INTERVAL_MS: 3000,
  MAX_SPEED_FACTOR: 2.9,
  MAX_SCROLL_SPEED: 680,
  INITIAL_SPAWN_INTERVAL_MS: 560,
  MIN_SPAWN_INTERVAL_MS: 240,
  INITIAL_TICK_MS: 500,
  MIN_TICK_MS: 200,
  TICK_SPEEDUP_INTERVAL_MS: 10000,
  CANVAS_WIDTH: 1040,
  CANVAS_HEIGHT: 700,
  // Left edge of each road on the canvas. Each road has a 48px shoulder on both sides.
  ROAD_X: [116, 636],
  ROAD_Y: 92,
  SHOULDER_WIDTH: 48,
  ROAD_WIDTH: 288,
  ROAD_HEIGHT: 560,
  LANE_WIDTH: 96,
  ITEM_SIZE: 30,
};

export const COLORS = {
  PAPER: "#07070d",
  INK: "#f7f7ff",
  PURPLE: "#6C4FE0",
  SKY: "#A4BDFF",
  POSIE: "#FF6E6E",
  DANDELION: "#ffd84d",
  FOREST: "#35946A",
  NEUTRAL_50: "#11121d",
  NEUTRAL_100: "#1a1b2b",
  NEUTRAL_200: "#31334b",
  NEUTRAL_500: "#8588a8",
  NEUTRAL_700: "#c9cbed",
};

// Each racer's defaults. Override them in .env with <ENV_PREFIX>_MODEL_NAME,
// <ENV_PREFIX>_API_KEY, and <ENV_PREFIX>_BASE_URL.
export const RACERS = {
  d1: { envPrefix: "LIQUID", model: "d1:free", baseUrl: "https://api.liquid.ai" },
  jev: { envPrefix: "JEV", model: "typesafe/jev-1.13", baseUrl: "https://openrouter.ai" },
};

// The decision endpoint path differs per API, so it is picked from the base URL's host.
export const DECISION_PATHS = {
  "openrouter.ai": "/api/alpha/decisions",
  default: "/v1/systemone",
};

export const LANES = ["left", "center", "right"];

export const MODES = {
  AI: "ai",
  HUMAN: "human",
};
