<h1 align="center">Liquid AI Cookbook</h1>
<p align="center"><em>Build with Liquid AI models and the Liquid Edge AI Platform</em></p>

<p align="center">
    🌊 <a href="https://docs.liquid.ai/"><b>Liquid Docs</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/LiquidAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🚀 <a href="https://leap.liquid.ai">Liquid Edge AI Platform</a>
</p>
<p align="center">
    <a href="https://discord.com/invite/liquid-ai"><img src="https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white" alt="Join Discord"></a>&nbsp&nbsp</a>
</p>


## Welcome dear developer! 👋

This repository contains **examples**, **tutorials**, and **applications** built with Liquid AI open-weight models (LFM) and the open-source LEAP SDK.

Whether you're looking to customize models, deploy to laptops, edge devices, or build complete applications, you'll find resources here to get started.

## What are you looking for? 🔍

- [Examples of local AI applications 🤖](#examples-of-local-ai-applications) that showcase the power of our LFM model family.
- [Fine tuning notebooks and examples 🎯](./finetuning/README.md) to increase the performance of LFM2 models for your specific use case.
- [Deployment to iOS and Android devices 📱](#deployment-to-ios-and-android-devices) with the LEAP SDK

- [End-to-end tutorials 📚](#end-2-end-tutorials) that cover the whole path from data collection, evaluation, fine-tuning and deployment.

- [Examples built by our growing community 🌟](#examples-built-by-our-community) that showcase what is possible with Liquid AI models.

- [60-minute technical deep dives on all-things-efficient-AI](#recorded-sessions-60-minute-technical-deep-dives-on-all-things-efficient-ai) every month in the [Liquid Discord community](https://discord.com/invite/liquid-ai).

## Examples of local AI applications

| Name | What's that? |  |
|-------|-----------|---------------|
| 🧾 **invoice-parser** | A Python CLI that extracts structured data from invoice pdfiles using LFM2-VL-3B | [▶️ Go to the code](./examples/invoice-parser/) |
| 🎙️ **audio-transcription-cli** | A Python CLI for real-time audio-to-text transcription using LFM2-Audio-1.5B with llama.cpp | [▶️ Go to the code](./examples/audio-transcription-cli/) |
| ✈️ **flight-search-assistant** | A Python CLI that helps you find and book plane tickets using LFM2.5-1.2B-Thinking with tool calling | [▶️ Go to the code](./examples/flight-search-assistant/) |


## Deployment to iOS and Android devices

The [LEAP Edge SDK](https://leap.liquid.ai/docs/edge-sdk/overview) is our native framework for running LFM2 models on mobile devices.

Written for Android (Kotlin) and iOS (Swift), the goal of the Edge SDK is to make Small Language Model deployment as easy as calling a cloud LLM API endpoint, for any app developer.

| Platform | Example |  |
|-------|-----------|---------------|
| **Android** | LeapChat: A comprehensive chat application with real-time token streaming, persistent message history, and modern chat UI featuring message bubbles and typing indicators | [🤖 Tutorial](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/Android/LeapChat) |
|  | SloganApp: Single turn generation for marketing. The UI is implemented with Android Views.| [🤖 Tutorial](./examples/leap-slogan-example-ios/README.md) |
|  | ShareAI: Website summary generator | [▶️ Go to the code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/Android/ShareAI) |
|  | Recipe Generator: Structured output generation with the LEAP SDK | [▶️ Go to the code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/Android/RecipeGenerator) |
|  | Visual Language Model example | [▶️ Go to the code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/Android/VLMExample) |
||||
| **iOS** | LeapChat: A comprehensive chat application demonstrating advanced LeapSDK features including real-time streaming, conversation management, and modern UI components. | [▶️ Go to the code ](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/iOS/LeapChatExample) |
|  | LeapSloganExample: A simple SwiftUI app demonstrating basic LeapSDK integration for text generation.| [▶️ Go to the code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/iOS/LeapChatExample) |
|  | Recipe Generator: Structured output generation | [▶️ Go to the code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/iOS/RecipeGenerator) |
|  | Audio demo: A SwiftUI app demonstrating audio input and output with the LeapSDK for on-device AI inference. | [▶️ Go to the code](https://github.com/Liquid4All/LeapSDK-Examples/tree/main/iOS/LeapAudioDemo) |



## End-2-end Tutorials

Complete end-to-end tutorials that take you from setup to deployment.

| Tutorial | Repository |
|----------|------------|
| Super fast and accurate image classification on edge devices | [▶️ Go to the repo](https://github.com/Paulescu/image-classification-with-local-vlms) ![GitHub Repo stars](https://img.shields.io/github/stars/Paulescu/image-classification-with-local-vlms) |
| Let's build a Chess game using small and local Large Language Models | [▶️ Go to the repo](https://github.com/Paulescu/chess-game) ![GitHub Repo stars](https://img.shields.io/github/stars/Paulescu/chess-game) |


## Examples built by our community

Working applications that demonstrate Liquid models in action.

| Project | Repository |
|---------|------------|
| TranslatorLens: Building An Offline Translation Camera | [▶️ Go to the repo](https://github.com/linmx0130/TranslatorLens) ![GitHub Repo stars](https://img.shields.io/github/stars/linmx0130/TranslatorLens) |
| Food Images Fine-tuning | [▶️ Go to the repo](https://github.com/benitomartin/food-images-finetuning) ![GitHub Repo stars](https://img.shields.io/github/stars/benitomartin/food-images-finetuning) |
| Meeting Intelligence CLI | [▶️ Go to the repo](https://github.com/chintan-projects/meeting-prompter) ![GitHub Repo stars](https://img.shields.io/github/stars/chintan-projects/meeting-prompter) |
| Private Doc Q&A: On-device document Q&A with RAG and voice input | [▶️ Go to the repo](https://github.com/chintan-projects/private-doc-qa) ![GitHub Repo stars](https://img.shields.io/github/stars/chintan-projects/private-doc-qa) |
| Photo Triage Agent: Private photo library cleanup using LFM vision model | [▶️ Go to the repo](https://github.com/chintan-projects/photo-triage-agent) ![GitHub Repo stars](https://img.shields.io/github/stars/chintan-projects/photo-triage-agent) | 
| LFM-Scholar: Automated literature review agent capable of finding and citing papers | [▶️ Go to the repo](https://github.com/gyunggyung/LFM-Scholar) ![GitHub Repo stars](https://img.shields.io/github/stars/gyunggyung/LFM-Scholar) |
| LFM2-KoEn-Tuning: Fine-tuned LFM2 1.2B model specialized for Korean-English translation | [▶️ Go to the repo](https://github.com/gyunggyung/LFM2-KoEn-Tuning) ![GitHub Repo stars](https://img.shields.io/github/stars/gyunggyung/LFM2-KoEn-Tuning) |

## Recorded sessions: 60-minute technical deep dives on all-things-efficient-AI

| Date | Topic |
|---------|------------|
| 2025-11-06 | [▶️ Fine-tuning LFM2-VL for image classification](https://www.youtube.com/watch?v=00IK9apncCg) |
| 2025-11-27 | [▶️ Building a 100% local Audio-to-Speech CLI with LFM2-Audio](https://www.youtube.com/watch?v=yeu077gPmCA) |
| 2025-12-26 | [▶️ Fine-tuning LFM2-350M for browser control with GRPO and OpenEnv](https://www.youtube.com/watch?v=gKQ08yee3Lw) |
| 2026-01-22 | [▶️ Local video-captioning with LFM2.5-VL-1.6B and WebGPU](https://www.youtube.com/watch?v=xsWARHFoA3E) |


Wanna participate in the next session? [Join the Liquid Discord community](https://discord.com/invite/liquid-ai) and head to the `#live-events` channel!

## Contributing

We welcome contributions!

- Open a PR with a link to your project github repo in the `Examples built by our community` section.


## Support 💬

- 📖 [Liquid AI Documentation](https://docs.liquid.ai/)
- 💬 [Join our community on Discord](https://discord.com/invite/liquid-ai)
