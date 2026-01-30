# LFM2.5-Audio WebGPU Demo

[![Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.com/invite/liquid-ai)

This example demonstrates [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) running entirely in your browser using WebGPU and ONNX Runtime Web.

You can find all the code [in this Hugging Face Space](https://huggingface.co/spaces/LiquidAI/LFM2.5-Audio-1.5B-transformers-js/tree/main), including a deployed version you can interact with [0 setup here](https://huggingface.co/spaces/LiquidAI/LFM2.5-Audio-1.5B-transformers-js).


## Quickstart

1. Clone the repository:
   ```sh
   git clone https://huggingface.co/spaces/LiquidAI/LFM2.5-Audio-1.5B-transformers-js/
   cd LFM2.5-Audio-1.5B-transformers-js
   ```

2. Make sure you have `npm` (Node Package Manager) installed in your system:
   ```sh
   npm --version
   ```

   If the previous command throws an error it means you don't have `npm` and you must install it to build this demo . If you come from the Python world, you can think of `npm` as the Node JS equivalent of `pip`.

   [Downloading and installing Node.js and npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)

3. Install dependencies specified in `package.json` with `npm`
   ```sh
   npm install
   ```
   
4. Start the development server:
   ```sh
   npm run dev
   ```

   The dev server will start and provide you with a local URL (typically `http://localhost:5173`) where you can access the app in your browser.

## Features

- **ASR (Speech Recognition)**: Transcribe audio to text
- **TTS (Text-to-Speech)**: Convert text to natural speech
- **Interleaved**: Mixed audio and text conversation

## Requirements

- A browser with WebGPU support (Chrome/Edge 113+)
- Enable WebGPU at `chrome://flags/#enable-unsafe-webgpu` if needed

## Model

Uses quantized ONNX models from [LiquidAI/LFM2.5-Audio-1.5B-ONNX](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX).

## License

Model weights are released under the [LFM 1.0 License](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/blob/main/LICENSE).