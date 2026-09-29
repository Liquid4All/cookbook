import { MODES } from "./config.js";
import { Game } from "./game/game.js";
import { KeyboardInput } from "./ui/input.js";
import { renderResults } from "./ui/ui.js";

const canvas = document.querySelector("#game");
const startScreen = document.querySelector("#start-screen");
const resultsScreen = document.querySelector("#results-screen");
const modeAiButton = document.querySelector("#mode-ai");
const modeHumanButton = document.querySelector("#mode-human");
const raceAgainButton = document.querySelector("#race-again");
const winnerTitle = document.querySelector("#winner-title");
const resultsGrid = document.querySelector("#results-grid");
const setupNotice = document.querySelector("#setup-notice");

const input = new KeyboardInput();
const game = new Game({ canvas, input, onComplete: showResults });

// Enter starts You vs d1, which only needs the Liquid key.
let selectedMode = MODES.HUMAN;

game.renderIdle();
loadServerConfig();

modeAiButton.addEventListener("click", () => beginRace(MODES.AI));
modeHumanButton.addEventListener("click", () => beginRace(MODES.HUMAN));
raceAgainButton.addEventListener("click", () => beginRace(selectedMode));
input.onEnter = () => {
  const onMenu = !startScreen.classList.contains("hidden") || !resultsScreen.classList.contains("hidden");
  if (onMenu) beginRace(selectedMode);
};

async function loadServerConfig() {
  try {
    const racers = await fetch("/api/config").then((r) => r.json());
    const problems = Object.entries(racers)
      .filter(([, racer]) => !racer.ready)
      .map(([name, racer]) => `${name}: ${racer.error}`);
    setupNotice.textContent = problems.join(" · ");

    if (!racers.d1?.ready) {
      modeHumanButton.disabled = true;
      modeAiButton.disabled = true;
    } else if (!racers.jev?.ready) {
      modeAiButton.disabled = true;
      modeAiButton.title = "Add OPENROUTER_API_KEY to .env to enable this mode";
    }
  } catch {
    setupNotice.textContent = "Could not reach the dev server API. Start the demo with npm run dev.";
  }
}

function beginRace(mode) {
  const button = mode === MODES.AI ? modeAiButton : modeHumanButton;
  if (button.disabled) return;
  selectedMode = mode;
  startScreen.classList.add("hidden");
  resultsScreen.classList.add("hidden");
  game.start(mode);
}

function showResults(outcome) {
  renderResults(resultsGrid, winnerTitle, outcome);
  startScreen.classList.add("hidden");
  resultsScreen.classList.remove("hidden");
}

