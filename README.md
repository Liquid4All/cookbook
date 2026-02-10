<div align="center">
  <img 
    src="https://github.com/user-attachments/assets/e0f42ac6-822f-4b7b-a0ae-07b2a619258d" 
    alt="Liquid AI" 
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> • 
    <a href="https://docs.liquid.ai/lfm"><strong>Documentation</strong></a> • 
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a>
  </div>
  <br/>
  <a href="https://discord.com/invite/liquid-ai"><img src="https://img.shields.io/discord/1385439864920739850?style=for-the-badge&logo=discord&logoColor=white&label=Discord&color=5865F2" alt="Join Discord"></a>
</div>
</br>

**Examples**, **tutorials**, and **applications** to help you build with our open-weight [LFMs](https://huggingface.co/LiquidAI) and the [LEAP SDK](https://leap.liquid.ai/) on laptops, mobile, and edge devices.

## 🤖 Local AI Apps

Ready-to-run applications showcasing agentic workflows and real-time inference on a local device.

| Name | Description | Link |
|------|-------------|------|
| Invoice Parser | Extract structured data from invoice images using LFM2-VL-3B | [Code](./examples/invoice-parser/README.md) |
| Audio Transcription CLI | Real-time audio-to-text transcription using LFM2-Audio-1.5B with llama.cpp | [Code](./examples/audio-transcription-cli/) |
| Flight Search Assistant | Find and book plane tickets using LFM2.5-1.2B-Thinking with tool calling | [Code](./examples/flight-search-assistant/README.md) |
| Audio Car Cockpit | Voice-controlled car cockpit demo combining LFM2.5-Audio-1.5B with LFM2-1.2B-Tool | [Code](./examples/audio-car-cockpit/README.md) |
| Audio WebGPU Demo | Run LFM2.5-Audio-1.5B entirely in your browser for speech recognition, TTS, and conversation | [Code](./examples/audio-webgpu-demo/README.md) |
| Vision WebGPU Demo | Real-time video captioning with LFM2.5-VL-1.6B running in-browser using WebGPU | [Code](./examples/vl-webgpu-demo/README.md) |

## 📱 Mobile App Deployment

Native examples for deploying LFM2 models on iOS and Android using the [LEAP Edge SDK](https://leap.liquid.ai/docs/edge-sdk/overview). Written for Android (Kotlin) and iOS (Swift), the goal of the Edge SDK is to make Small Language Model deployment as easy as calling a cloud LLM API endpoint.

### Android

| Name | Description | Link |
|------|-------------|------|
| LeapChat | Chat app with real-time streaming, persistent history, and modern UI | [Code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/Android/LeapChat) |
| LeapAudioDemo | Audio input and output with LFM2.5-Audio-1.5B for on-device AI inference | [Code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/Android/LeapAudioDemo) |
| LeapKoogAgent | Integration with Koog framework for AI agent functionality | [Code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/Android/LeapKoogAgent) |
| SloganApp | Single turn marketing slogan generation with Android Views | [Code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/Android/SloganApp) |
| ShareAI | Website summary generator | [Code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/Android/ShareAI) |
| Recipe Generator | Structured output generation with the LEAP SDK | [Code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/Android/RecipeGenerator) |
| VLM Example | Visual Language Model integration | [Code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/Android/VLMExample) |

### iOS

| Name | Description | Link |
|------|-------------|------|
| LeapChat | Chat app with real-time streaming, conversation management, and SwiftUI | [Code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/iOS/LeapChatExample) |
| LeapSloganExample | Basic LeapSDK integration for text generation in SwiftUI | [Code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/iOS/LeapSloganExample) |
| Recipe Generator | Structured output generation | [Code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/iOS/RecipeGenerator) |
| Audio Demo | Audio input/output with LeapSDK for on-device AI inference | [Code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/iOS/LeapAudioDemo) |

## 🎯 Fine-Tuning Notebooks

Colab notebooks and Python scripts for customizing LFM models with your own data.

| Name | Description | Link |
|------|-------------|------|
| **Supervised Fine-Tuning (SFT)** | | |
| SFT with Unsloth | Memory-efficient SFT using Unsloth with LoRA for 2x faster training | [Notebook](./finetuning/notebooks/sft_with_unsloth.ipynb) |
| SFT with TRL | Supervised fine-tuning using Hugging Face TRL library with parameter-efficient LoRA | [Notebook](./finetuning/notebooks/sft_with_trl.ipynb) |
| **Reinforcement Learning** | | |
| GRPO with Unsloth | Train reasoning models using Group Relative Policy Optimization for verifiable tasks | [Notebook](./finetuning/notebooks/grpo_with_unsloth.ipynb) |
| GRPO with TRL | Train reasoning models using Group Relative Policy Optimization with rule-based rewards | [Notebook](./finetuning/notebooks/grpo_for_verifiable_tasks.ipynb) |
| **Continued Pre-Training (CPT)** | | |
| CPT for Translation | Adapt models to specific languages or translation domains using domain data | [Notebook](./finetuning/notebooks/cpt_translation_with_unsloth.ipynb) |
| CPT for Text Completion | Teach models domain-specific knowledge and creative writing styles | [Notebook](./finetuning/notebooks/cpt_text_completion_with_unsloth.ipynb) |
| **Vision-Language Models** | | |
| VLM SFT with Unsloth | Supervised fine-tuning for LFM2-VL models on custom image-text datasets | [Notebook](./finetuning/notebooks/sft_for_vision_language_model.ipynb) |

## 🌟 Community Projects

Open-source projects built by the community showcasing LFMs with real use cases.

| Name | Description | Link |
|------|-------------|------|
| Image Classification on Edge | End-to-end tutorial covering fine-tuning and deployment for super fast and accurate image classification using local VLMs | [Code](https://github.com/Paulescu/image-classification-with-local-vlms) |
| Chess Game with Small LMs | End-to-end tutorial covering fine-tuning and deployment to build a Chess game using Small Language Models | [Code](https://github.com/Paulescu/chess-game) |
| TranslatorLens | Offline translation camera for real-time text translation | [Code](https://github.com/linmx0130/TranslatorLens) |
| Food Images Fine-tuning | Fine-tune LFM models on food image datasets | [Code](https://github.com/benitomartin/food-images-finetuning) |
| Meeting Intelligence CLI | CLI tool for meeting transcription and analysis | [Code](https://github.com/chintan-projects/meeting-prompter) |
| Private Doc Q&A | On-device document Q&A with RAG and voice input | [Code](https://github.com/chintan-projects/private-doc-qa) |
| Photo Triage Agent | Private photo library cleanup using LFM vision model | [Code](https://github.com/chintan-projects/photo-triage-agent) |
| LFM-Scholar | Automated literature review agent for finding and citing papers | [Code](https://github.com/gyunggyung/LFM-Scholar) |
| LFM2-KoEn-Tuning | Fine-tuned LFM2 1.2B for Korean-English translation | [Code](https://github.com/gyunggyung/LFM2-KoEn-Tuning) |
| Chat with LEAP SDK | LEAP SDK integration for React Native | [Code](https://github.com/glody007/expo-leap-sdk) |
| Private Summarizer | 100% local text summarization with multi-language support | [Code](https://github.com/Private-Intelligence/private_summarizer) |
| Tiny-MoA | Mixture of Agents on CPU with LFM2.5 Brain (1.2B) | [Code](https://github.com/gyunggyung/Tiny-MoA) |
| LFM-2.5 JP on Web | LFM2.5 1.2B parameter Japanese language model running locally in the browser with WebGPU, using Transformers.js and ONNX Runtime on Web | [Code](https://github.com/sitammeur/lfm2.5-jp-web) |
| LFM-2.5 Thinking on Web | LFM2.5 1.2B parameter reasoning language model running locally in the browser with WebGPU, using Transformers.js and ONNX Runtime Web | [Code](https://github.com/sitammeur/lfm2.5-thinking-web) |

## 🕐 Technical Deep Dives

Recorded sessions (~60 minutes) covering advanced topics and hands-on implementations.

| Date | Topic | Link |
|------|-------|------|
| 2025-11-06 | Fine-tuning LFM2-VL for image classification | [Video](https://www.youtube.com/watch?v=00IK9apncCg) |
| 2025-11-27 | Building a 100% local Audio-to-Speech CLI with LFM2-Audio | [Video](https://www.youtube.com/watch?v=yeu077gPmCA) |
| 2025-12-26 | Fine-tuning LFM2-350M for browser control with GRPO and OpenEnv | [Video](https://www.youtube.com/watch?v=gKQ08yee3Lw) |
| 2026-01-22 | Local video-captioning with LFM2.5-VL-1.6B and WebGPU | [Video](https://www.youtube.com/watch?v=xsWARHFoA3E) |

Join the next session! Head to the `#live-events` channel on [Discord](https://discord.com/invite/liquid-ai).

## Contributing

We welcome contributions! Open a PR with a link to your project GitHub repo in the Community Projects section.

## Support

- 📖 [Liquid AI Documentation](https://docs.liquid.ai/)
- 💬 [Join our community on Discord](https://discord.com/invite/liquid-ai)
