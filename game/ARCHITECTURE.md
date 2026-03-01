# Steering Game - Architecture

A minimal web-based driving game controlled by webcam gesture detection. The user steers an imaginary wheel; two consecutive webcam frames are classified as LEFT, RIGHT, or STRAIGHT by LFM2.5-VL-1.6B running fully in the browser via ONNX + WebGPU.

## System Diagram

```
Webcam (getUserMedia)
   |
   v
FrameSampler  -- every 300ms -->  [frame_prev, frame_curr]  (base64 blobs)
                                              |
                                              v
                                       inference/prompt.ts
                                    builds multimodal message:
                                    [{ role: "user", content: [
                                        { type: "image", image: frame_prev },
                                        { type: "image", image: frame_curr },
                                        { type: "text",  text:  "LEFT, RIGHT or STRAIGHT?" }
                                    ]}]
                                              |
                                              v
                                    @huggingface/transformers
                                    pipeline("image-text-to-text",
                                      "LiquidAI/LFM2.5-VL-1.6B-ONNX",
                                      { dtype: "q4", device: "webgpu" })
                                              |
                                              v
                                    SteeringState
                               <left | straight | right>
                                              |
                                              v
                                    GameEngine (60fps Canvas)
```

No backend. Pure static app. Deploy anywhere (GitHub Pages, Vercel, HuggingFace Spaces, etc.).

## File Structure

```
game/
├── vite.config.ts          # COOP/COEP headers (required for self-hosted; HF Spaces adds these at infra level)
├── index.html
└── src/
    ├── main.ts
    ├── webcam/
    │   ├── capture.ts      # getUserMedia, draws to offscreen <canvas>
    │   └── sampler.ts      # keeps [prev, curr] frame pair, exports as base64 on interval
    ├── inference/
    │   ├── model.ts        # loadModel(), singleton pipeline with progress callback
    │   ├── prompt.ts       # buildMessages(frame1, frame2) -> multimodal message array
    │   └── steering.ts     # reactive Direction state + runInference(frames) -> Direction
    └── game/
        ├── engine.ts       # requestAnimationFrame loop
        ├── car.ts          # position, angle, speed
        ├── road.ts         # procedural road generation
        └── renderer.ts     # canvas draw calls
```

## Key Design Decisions

### Game loop vs inference loop

The game loop (60fps) and inference loop (~3fps) run at different frequencies and are fully decoupled. The game always uses the last known steering direction. This prevents input lag from blocking the canvas.

### No processor.ts

`@huggingface/transformers` handles image resizing, normalization, and tokenization internally when base64 images are passed to the pipeline. No manual preprocessing needed.

### max_new_tokens: 1

The model only needs to produce a single token (LEFT, RIGHT, or STRAIGHT). Setting `max_new_tokens: 1` with greedy decoding (`do_sample: false`) means a single forward pass with no autoregressive loop. This is the key latency optimization.

### Steering smoothing

Raw model output is binary (one of three classes). To avoid jerky movement, maintain a rolling window of the last N predictions and interpolate toward the target steering angle each frame.

## Inference Layer

### Model

`LiquidAI/LFM2.5-VL-1.6B-ONNX` is published on HuggingFace. No manual ONNX export required.

### Library

`@huggingface/transformers` v4.x wraps `onnxruntime-web` internally and handles the full VL pipeline (image preprocessing, tokenization, generation).

### model.ts pattern

Derived from the reference implementation at `LiquidAI/LFM2.5-1.2B-Thinking-WebGPU` Space:

```ts
import { pipeline, type ImageTextToTextPipeline } from "@huggingface/transformers"

const MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B-ONNX"

let pipelinePromise: Promise<ImageTextToTextPipeline> | null = null

export function loadModel(onProgress: (pct: number) => void) {
  if (pipelinePromise) return pipelinePromise
  pipelinePromise = pipeline("image-text-to-text", MODEL_ID, {
    dtype: "q4",
    device: "webgpu",
    progress_callback: (p: any) => {
      if (p.status === "progress" && p.file?.endsWith(".onnx_data"))
        onProgress(p.progress)
    },
  })
  return pipelinePromise
}
```

### steering.ts pattern

```ts
const result = await pipe(messages, { max_new_tokens: 1, do_sample: false })
const token = result[0].generated_text.trim().toUpperCase()
return token.includes("LEFT") ? "left" : token.includes("RIGHT") ? "right" : "straight"
```

## Vite Config

ONNX Runtime Web with WebGPU requires `SharedArrayBuffer`, which needs security headers. HuggingFace Spaces adds these at infrastructure level. For self-hosted deployments, add them to `vite.config.ts`:

```ts
export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
})
```

Same headers must be set in production (Vercel `vercel.json`, Netlify `_headers`, etc.).

## Tech Stack

- **Vite + TypeScript** - build tooling and language
- **HTML5 Canvas** - game rendering
- **@huggingface/transformers v4.x** - inference (wraps onnxruntime-web + WebGPU backend)
- **getUserMedia API** - webcam capture

No UI framework. No game engine. No backend.

## Risk Summary

| Risk | Status |
|---|---|
| ONNX export of VL model | Resolved: `LFM2.5-VL-1.6B-ONNX` exists on HuggingFace |
| Model size | ~400-800MB with q4 quantization (acceptable for browser) |
| WebGPU support | Chrome 113+ and Edge 113+ only; add a browser-check with a clear error message |
| COOP/COEP headers | Needed for self-hosted; HF Spaces handles automatically |
| VLM generation latency | Mitigated by `max_new_tokens: 1` + greedy decoding (single forward pass) |

## Suggested Build Order

1. Bare inference spike: load `LFM2.5-VL-1.6B-ONNX` in a plain HTML file, run on two static images, confirm LEFT/RIGHT/STRAIGHT output
2. `inference/` module: model load with progress, `buildMessages`, `runInference`
3. `webcam/` module: capture working, sampler producing base64 frame pairs
4. Wire 2 + 3: live steering output in console
5. Minimal game loop driven by keyboard
6. Swap keyboard input for steering state
7. Polish: loading screen with download progress bar, score display, browser compatibility check

## Reference Implementation

`LiquidAI/LFM2.5-1.2B-Thinking-WebGPU` HuggingFace Space contains a working Vite + TypeScript + `@huggingface/transformers` + WebGPU implementation for an LFM text model. The inference pattern in `src/hooks/LLMProvider.tsx` is the direct reference for `model.ts` and `steering.ts`.
