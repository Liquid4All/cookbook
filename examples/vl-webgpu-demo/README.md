# Real-time video captioning with LFM2.5-VL-1.6B and WebGPU

[![Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.com/invite/liquid-ai)

This example showcases how to run a local vision language model on the browser using the LFM2.5-VL-1.6B model and the ONNX runtime.

This is a WebGPU-based vision-language model demo, so make sure you're using a browser that supports WebGPU (like Chrome or Edge).


## Table of Contents

- [The traditional approach: Cloud-based inference](#the-traditional-approach-cloud-based-inference)
- [The local alternative: In-browser inference with WebGPU](#the-local-alternative-in-browser-inference-with-webgpu)
  - [Key advantages](#key-advantages)
- [How to run the app locally](#how-to-run-the-app-locally)
- [How to deploy the app to production](#how-to-deploy-the-app-to-production)
- [Project Structure](#project-structure)
  - [How the code is organized](#how-the-code-is-organized)
- [Frequently Asked Questions for non-Node JS developers](#frequently-asked-questions-for-non-node-js-developers)
  - [What does `npm run` do?](#what-does-npm-run-do)
  - [What is Vite?](#what-is-vite)
- [Need help?](#need-help)


## The traditional approach: Cloud-based inference

Typically, vision-language model inference follows a server-client architecture. Your application sends images and prompts to a cloud-hosted frontier model (like Claude, GPT-4V, or Gemini), which processes the request on powerful servers and returns the results:

![Remote inference example](./media/remote-inference.gif)

While this approach works well for many use cases, it comes with several limitations:

- **Privacy concerns**: Images and data must be sent to external servers
- **Latency**: Network round-trips add delays, especially for real-time applications
- **Cost**: API calls accumulate charges based on usage
- **Connectivity dependency**: Requires stable internet connection
- **Rate limits**: Subject to API quotas and throttling

## The local alternative: In-browser inference with WebGPU

This demo showcases a different approach: running a vision-language model entirely in your browser using WebGPU for GPU acceleration. The LFM2.5-VL-1.6B model (1.6 billion parameters, quantized) runs directly on your device without sending data anywhere.

![Local inference example](./media/local-inference.gif)

### Key advantages

- **Complete privacy**: All data stays on your device
- **Low latency**: No network overhead, ideal for real-time video processing
- **Zero inference cost**: No API charges after initial model download
- **Offline capability**: Works without internet connection (after model caching)
- **No rate limits**: Process as many frames as your hardware can handle

## How to run the app locally

1. Clone the repository:
   ```sh
   git clone https://huggingface.co/spaces/LiquidAI/LFM2.5-VL-1.6B-WebGPU/
   cd LFM2.5-VL-1.6B-WebGPU
   ```

2. Make sure you have `npm` (Node Package Manager) installed in your system:
   ```sh
   npm --version
   ```

   If the previous command throws an error it means you don't have `npm` and you must install it to build this demo . If you come from the Python world, you can think of `npm` as the Node JS equivalent of `pip`.

   [Downloading and installing Node.js and npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)

2. Install dependencies specified in `package.json` with `npm`
   ```sh
   npm install
   ```
   
3. Start the development server:
   ```sh
   npm run dev
   ```

   The dev server will start and provide you with a local URL (typically `http://localhost:5173`) where you can access the app in your browser.

4. Optional. Run with Docker locally

   If you prefer to test the production build locally using Docker:

   ```sh
   # Build the Docker image
   docker build -t lfm-vl-webgpu -f LFM2.5-VL-1.6B-WebGPU/Dockerfile .

   # Run the container
   docker run -p 7860:7860 lfm-vl-webgpu
   ```

   Then access the app at `http://localhost:7860` in your browser.


## How to deploy the app to production

After running `npm run build`, you'll have a production-ready bundle in the `dist/` directory.

![Vite build](./media/vite.jpg)

This static website can be deployed to any static hosting service, such as:

- **HuggingFace Space** (for demo purposes only). It automatically uses the Dockerfile at the root of the directory. You can see it in action [here](https://huggingface.co/spaces/LiquidAI/LFM2.5-VL-1.6B-WebGPU).

- **Platform as a Service (PaaS)**
  - Vercel
  - Netlify

- **Cloud storage + CDN**
  - AWS S3 + CloudFront
  - GCS + Cloud CDN
  - Azure Blob + CDN
- **Traditional web servers**
  - nginx
  - Apache
  - Caddy

**Important**: This app requires specific CORS headers to enable WebGPU and SharedArrayBuffer. Ensure your hosting solution supports custom headers (see details below).


### How the code is organized

**Vite's role**: Vite is the build tool that bundles all JavaScript files and dependencies into optimized browser-ready code. During development (`npm run dev`), it serves files with hot-reload. For production (`npm run build`), it creates a minified bundle in the `dist/` directory.

**Code organization**: The application follows a modular architecture with separation of concerns:

- **Entry point** (`index.html` → `main.js`): Initializes the app, sets up event listeners, coordinates between modules
- **Configuration** (`config.js`): Model definitions, HuggingFace URLs, quantization options
- **Inference pipeline** (`infer.js` → `webgpu-inference.js` → `vl-model.js`):
  - Routes inference requests
  - Manages model lifecycle and state
  - Handles ONNX Runtime sessions and token generation
- **Image processing** (`vl-processor.js`): Preprocesses webcam frames into model-ready patches and tensors
- **UI layer** (`ui.js`): Updates DOM elements, displays progress, shows captions

Each JavaScript file is an ES module that exports/imports functions, keeping code organized and maintainable. Vite handles module resolution and bundling automatically.

## Frequently Asked Questions for non-Node JS developers

### What does `npm run` do?

`npm run` executes custom scripts defined in our `package.json` file.
In `package.json`, you define scripts in the "scripts" section:
```json
// package.json
{
   "name": "lfm25-vl-webgpu",
   ...
   "scripts": {
      "dev": "vite",
      "build": "vite build",
      "preview": "vite preview"
   },
}
```
Then you run them with:
- `npm run dev      # Runs "vite"`
- `npm run build    # Runs "vite build"`
- `npm run preview     # Runs "jest"`

So npm run is essentially npm's task runner, letting you define and execute project-specific commands.

### What is Vite?

`vite` is a modern build tool that serves two purposes:

1. **Development Server** (`npm run dev`): Serves your application locally with instant hot-reload when you edit code. Think of it like Flask's `debug=True` mode or Django's `runserver` - but optimized for JavaScript modules and incredibly fast.

2. **Production Bundler** (`npm run build`): Transforms and optimizes your source code (`.js`, `.css`, assets) into production-ready bundles that are minified, optimized, and efficient for browsers to load.

**Python analogy:** Vite combines `uvicorn --reload` (fast dev server) with `setuptools` (build/packaging tool) into one lightning-fast tool specifically designed for modern web development.

## Need help?

Join the [Liquid AI Discord Community](https://discord.com/invite/liquid-ai) and ask.

[![Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.com/invite/liquid-ai)