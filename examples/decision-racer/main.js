import { COLORS, CONFIG, MODES } from "./config.js";
import { Game } from "./game.js";
import { KeyboardInput } from "./input.js";
import { renderResults } from "./ui.js";

const canvas = document.querySelector("#game");
const startScreen = document.querySelector("#start-screen");
const resultsScreen = document.querySelector("#results-screen");
const modeAiButton = document.querySelector("#mode-ai");
const modeHumanButton = document.querySelector("#mode-human");
const raceAgainButton = document.querySelector("#race-again");
const winnerTitle = document.querySelector("#winner-title");
const resultsGrid = document.querySelector("#results-grid");

const input = new KeyboardInput();
const game = new Game({
  canvas,
  input,
  liquidModel: undefined,
  jevModel: undefined,
  onComplete: showResults,
});

let selectedMode = MODES.AI;
let serverConfig = { hasLiquidKey: false, hasOpenRouterKey: false };

drawAttractScreen();
loadServerConfig();

modeAiButton.addEventListener("click", () => beginRace(MODES.AI));
modeHumanButton.addEventListener("click", () => beginRace(MODES.HUMAN));
raceAgainButton.addEventListener("click", () => beginRace(selectedMode));
input.onEnter = () => {
  if (!startScreen.classList.contains("hidden")) beginRace(selectedMode);
  if (!resultsScreen.classList.contains("hidden")) beginRace(selectedMode);
};

async function loadServerConfig() {
  try {
    const response = await fetch("/api/config");
    serverConfig = await response.json();
  } catch {
    serverConfig = { hasLiquidKey: false, hasOpenRouterKey: false };
  }
}

function beginRace(mode) {
  selectedMode = mode;
  hideAllOverlays();
  game.liquidModel = serverConfig.liquidModel;
  game.jevModel = serverConfig.jevModel;
  game.start(mode);
}

function showResults(outcome) {
  renderResults(resultsGrid, winnerTitle, outcome);
  hideAllOverlays();
  resultsScreen.classList.remove("hidden");
}


function hideAllOverlays() {
  startScreen.classList.add("hidden");
  resultsScreen.classList.add("hidden");
}

function drawAttractScreen() {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT);
  ctx.fillStyle = COLORS.PAPER;
  ctx.fillRect(0, 0, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT);
  ctx.fillStyle = COLORS.NEUTRAL_100;
  ctx.fillRect(86, 92, 348, 560);
  ctx.fillRect(606, 92, 348, 560);
  ctx.fillStyle = COLORS.NEUTRAL_50;
  ctx.fillRect(134, 92, 252, 560);
  ctx.fillRect(654, 92, 252, 560);
  ctx.strokeStyle = COLORS.INK;
  ctx.lineWidth = 2;
  ctx.strokeRect(134, 92, 252, 560);
  ctx.strokeRect(654, 92, 252, 560);
}
