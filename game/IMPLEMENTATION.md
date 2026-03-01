# Steering Game - Implementation Plan

Ordered steps to go from an empty directory to a working webcam-controlled driving game. Each step produces something independently runnable or verifiable before the next step begins.

## Step 1: Inference spike (plain HTML, no build tooling)

**Goal:** Confirm that `LFM2.5-VL-1.6B-ONNX` loads in the browser and classifies two images correctly. Validate latency and output format before writing any game code.

**Files to create:**
- `spike/index.html` - standalone, no bundler

**What to implement:**
- Import `@huggingface/transformers` from a CDN (esm.sh or jsdelivr)
- Load the pipeline with `dtype: "q4"`, `device: "webgpu"`
- Hardcode two static images (e.g. a person with arms left vs. center)
- Call the pipeline with `max_new_tokens: 1`, `do_sample: false`
- Log the raw output token to the console

**Done when:** Console prints `LEFT`, `RIGHT`, or `STRAIGHT` for hand-picked test images. Note the wall-clock latency for one inference call.

**Watch for:** WebGPU unavailable error (wrong browser), COOP/COEP header errors (serve with `npx serve --cors`), unexpected multi-token output.

---

## Step 2: Project scaffold

**Goal:** Set up the Vite + TypeScript project with the correct structure and config.

**Commands:**
```bash
npm create vite@latest frontend -- --template vanilla-ts
cd frontend
npm install @huggingface/transformers
```

**Files to create/edit:**

`vite.config.ts`:
```ts
import { defineConfig } from "vite"

export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
})
```

`tsconfig.json` - enable `"strict": true`, `"target": "ES2022"`.

`src/` directory layout matching ARCHITECTURE.md:
```
src/
  main.ts
  webcam/
    capture.ts
    sampler.ts
  inference/
    model.ts
    prompt.ts
    steering.ts
  game/
    engine.ts
    car.ts
    road.ts
    renderer.ts
```

Create each file as an empty module export so TypeScript resolves all imports from the start.

**Done when:** `npm run dev` starts without errors and `index.html` renders a blank page.

---

## Step 3: Inference module

**Goal:** A clean, typed inference API that the rest of the app can call without knowing about `@huggingface/transformers` internals.

### `src/inference/model.ts`

Singleton pipeline loader. Exposes:
```ts
export type ModelStatus =
  | { state: "idle" }
  | { state: "loading"; progress: number }
  | { state: "ready" }
  | { state: "error"; message: string }

export function loadModel(onStatus: (s: ModelStatus) => void): Promise<void>
export function getModel(): ImageTextToTextPipeline  // throws if not ready
```

- Guard against calling `pipeline()` twice (singleton pattern from reference impl)
- Progress callback filters for `.onnx_data` files only (same as reference impl)
- On error, reset the singleton so `loadModel` can be retried

### `src/inference/prompt.ts`

Builds the multimodal message array:
```ts
export function buildMessages(framePrev: string, frameCurr: string): Message[]
```

- `framePrev` and `frameCurr` are base64 data URLs (`data:image/jpeg;base64,...`)
- Text prompt: `"Two consecutive images show a person holding an imaginary steering wheel. Classify the hand motion as exactly one word: LEFT, RIGHT, or STRAIGHT."`
- Return format: `[{ role: "user", content: [{ type: "image" }, { type: "image" }, { type: "text" }] }]`

### `src/inference/steering.ts`

Thin wrapper that calls the model and maps output to a typed direction:
```ts
export type Direction = "left" | "straight" | "right"

export async function classify(framePrev: string, frameCurr: string): Promise<Direction>
```

- Calls `getModel()`, builds messages via `buildMessages()`
- Runs with `{ max_new_tokens: 1, do_sample: false }`
- Parses the single output token with `.includes()` matching
- Returns `"straight"` as the safe default for any unrecognized token

**Done when:** Calling `classify(img1, img2)` from `main.ts` with two hardcoded base64 images returns the correct direction.

---

## Step 4: Webcam module

**Goal:** Continuously capture webcam frames and expose a rolling `[prev, curr]` pair as base64 strings, ready to pass directly to `classify()`.

### `src/webcam/capture.ts`

```ts
export async function startCapture(videoEl: HTMLVideoElement): Promise<void>
export function captureFrame(videoEl: HTMLVideoElement, canvas: OffscreenCanvas): string
// returns a base64 JPEG data URL at reduced resolution (e.g. 320x240)
```

- Calls `navigator.mediaDevices.getUserMedia({ video: true })`
- Assigns the stream to `videoEl.srcObject`
- `captureFrame` draws the video frame to an `OffscreenCanvas` and returns `canvas.convertToBlob()` encoded as base64
- Reduce resolution on capture (320x240) to keep inference payload small

### `src/webcam/sampler.ts`

```ts
export type FramePair = { prev: string; curr: string }

export function startSampler(
  videoEl: HTMLVideoElement,
  onPair: (pair: FramePair) => void,
  intervalMs?: number   // default 300
): () => void          // returns a stop function
```

- Keeps the last captured frame as `prev`
- Every `intervalMs`, captures a new frame as `curr`, calls `onPair`, then promotes `curr` to `prev`
- Does not call `onPair` until at least two frames have been captured
- Returns a cleanup function (clears the interval)

**Done when:** `startSampler` is wired in `main.ts` and the console logs two different base64 strings every 300ms.

---

## Step 5: Wire inference to webcam

**Goal:** Live steering direction printed to the console from real webcam input.

In `main.ts`:
- Call `loadModel()` on page load, show a text status on the page
- Once model is ready, start the sampler
- In `onPair`, call `classify(pair.prev, pair.curr)` and log the result

**Done when:** Waving hands left/right in front of the webcam produces the correct direction in the console. Tune the prompt if accuracy is poor at this step.

---

## Step 6: Game engine (keyboard-driven)

**Goal:** A playable game driven by keyboard arrows before connecting the VLM. Lets you tune gameplay feel independently of inference.

### `src/game/car.ts`

```ts
export interface CarState {
  x: number          // lateral position, 0 = left edge, 1 = right edge
  speed: number      // forward speed (pixels/sec, constant for now)
  steer: number      // current steering value: -1 left, 0 straight, +1 right
}

export function updateCar(car: CarState, dt: number): CarState
```

- `steer` is applied to `x` each frame: `x += steer * STEER_RATE * dt`
- Clamp `x` to `[0.1, 0.9]`
- `speed` is constant (e.g. 300px/s) for the MVP

### `src/game/road.ts`

```ts
export interface RoadSegment {
  curveOffset: number   // lateral offset of the road center at this segment
}

export function generateRoad(segmentCount: number): RoadSegment[]
export function advanceRoad(segments: RoadSegment[], speed: number, dt: number): RoadSegment[]
```

- Road is a fixed-width lane scrolling toward the player from the top
- `curveOffset` gives gentle S-curves by incrementing with a sine function
- `advanceRoad` shifts segments downward each frame and appends new ones at the top

### `src/game/renderer.ts`

```ts
export function render(
  ctx: CanvasRenderingContext2D,
  road: RoadSegment[],
  car: CarState,
  score: number
): void
```

- Top-down view: road as a gray band, car as a colored rectangle
- Road edges drawn as two lines following `curveOffset` per segment
- Score displayed in the top-left corner
- Keep draw calls minimal (no images, just fills and strokes)

### `src/game/engine.ts`

```ts
export function startGame(
  canvas: HTMLCanvasElement,
  getDirection: () => Direction
): () => void    // returns stop function
```

- Owns the `requestAnimationFrame` loop
- Reads `getDirection()` each frame to update `car.steer`
- Calls `updateCar`, `advanceRoad`, `render` each frame
- Tracks score as elapsed seconds (or distance traveled)
- Detects off-road: if `car.x` goes outside road bounds, stop the loop

**Done when:** The game runs end-to-end with arrow keys (`getDirection` reads `ArrowLeft`/`ArrowRight`).

---

## Step 7: Connect VLM steering to the game

**Goal:** Replace keyboard input with live inference output.

Changes in `main.ts`:
- Add a `currentDirection` variable, initialized to `"straight"`
- In the sampler's `onPair` callback, call `classify()` and update `currentDirection`
- Pass `() => currentDirection` as `getDirection` to `startGame`

**Done when:** The car responds to hand gestures instead of keyboard. The game loop never blocks waiting for inference.

---

## Step 8: Loading screen and UX polish

**Goal:** A user-facing app that can be demoed without explaining the console.

**Add to `index.html` / `main.ts`:**

- **Browser check on startup:** Detect `navigator.gpu` availability. If absent, show a message: "This app requires Chrome 113+ or Edge 113+ with WebGPU enabled."
- **Loading screen:** Full-page overlay with model download progress bar (fed by `onStatus` from `loadModel`)
  - Show percentage from the `.onnx_data` progress callback
  - Show "Initializing WebGPU..." while the pipeline compiles shaders
- **Webcam permission prompt:** Show a visible "Allow camera access" instruction before calling `getUserMedia`
- **Score display:** Overlay on the canvas showing current score and personal best (stored in `localStorage`)
- **Game over screen:** "You went off-road. Score: X. Press Enter to restart."

**Done when:** A first-time user can open the app and play without touching the browser console.

---

## Acceptance Checklist

- [ ] Model loads with progress feedback and is cached on second visit (browser Cache API handled by transformers.js automatically)
- [ ] Webcam frame pair is captured and classified every ~300ms without blocking the game loop
- [ ] Car responds to LEFT/RIGHT gestures within one inference cycle (~300ms)
- [ ] Steering smoothing prevents jitter from occasional misclassifications
- [ ] Off-road detection ends the game
- [ ] Score is displayed and persisted across sessions
- [ ] Browser incompatibility shows a clear error, not a crash
- [ ] App runs correctly when served with COOP/COEP headers

## Notes

- Prompt tuning (Step 5) is the most unpredictable part. Budget time to experiment with different phrasings and test with varied lighting and distances from the camera.
- Model first-load is ~400-800MB. Users on slow connections need the progress bar. The browser caches the model files after the first download via the Cache API (managed internally by `@huggingface/transformers`).
- The spike in Step 1 is worth doing before any TypeScript scaffolding. If WebGPU inference does not work in the target environment, everything else changes.
