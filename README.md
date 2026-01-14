<h1 align="center">Liquid AI Cookbook</h1>
<p align="center"><em>Build with Liquid AI Models and the Liquid Edge AI Platform</em></p>

<p align="center">
    🌊 <a href="https://docs.liquid.ai/"><b>Liquid Docs</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/LiquidAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🚀 <a href="https://leap.liquid.ai">Liquid Edge AI Platform</a>
</p>
<p align="center">
    <a href="https://discord.com/invite/liquid-ai"><img src="https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white" alt="Join Discord"></a>&nbsp&nbsp</a>
</p>


## Welcome dear developer! 👋

This repository contains **examples**, **tutorials**, and **applications** built with Liquid AI open-weight models and the open-source LEAP SDK.

Whether you're looking to customize models, deploy to laptops, edge devices, or build complete applications, you'll find resources here to get started.

## What are you looking for? 🔍

- [Local agentic workflow examples 🤖](#local-agentic-workflow-examples)
- [Fine tune an LFM2 model 🎯](#fine-tune-an-lfm2-model)
- [Deploy an LFM2 model to an iOS or Android device 📱](#deploy-to-an-edge-device)
- [End-to-end tutorials 📚](#end-2-end-tutorials). Complete walkthroughs from setup to production.
- [Examples built by our community 🌟](#examples-built-by-our-community) that you can run and modify

## Local agentic workflow examples

| Name | What's that? |  |
|-------|-----------|---------------|
| **invoice-parser** | A Python CLI that extracts structured data from bill pictures using a 2-step agentic workflow | [▶️ Go to the code](./examples/invoice-parser/) |
| **audio-transcription-cli** | A Python CLI for real-time audio-to-text transcription using LFM2-Audio-1.5B with llama.cpp | [▶️ Go to the code](./examples/audio-transcription-cli/) |





## Fine-Tune an LFM2 model

### LFM2 (Text-to-text)

LFM2 is a generation of hybrid models, designed for on-device deployment, ranging from 350M up to 8B parameters.

These models are particularly suited for agentic tasks, data extraction, RAG, creative writing, and multi-turn conversations. We do not recommend using them for tasks that are knowledge-intensive or require programming skills.

<!-- #### Liquid Nanos for Text-to-Text problems -->

| Model | Technique |  |
|-------|-----------|---------------|
| [LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B) | Supervised Fine Tuning (TRL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OXLEuSmzF4AjJ7yqRCDTn-ltvFjoGR9j?usp=sharing) |
|  | Direct Preference Optimization (TRL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q8hIHIQ8oofshcNYHUcYp1akUcZ-ufSn?usp=sharing) |
||||
| [LFM2-2.6B](https://huggingface.co/LiquidAI/LFM2-2.6B) | Supervised Fine Tuning (TRL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j5Hk_SyBb2soUsuhU0eIEA9GwLNRnElF?usp=sharing) |
|  | Supervised Fine Tuning (Axolotl) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/155lr5-uYsOJmZfO6_QZPjbs8hA_v8S7t?usp=sharing) |
|  | Supervised Fine Tuning (Unsloth) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HROdGaPFt1tATniBcos11-doVaH7kOI3?usp=sharing) |
|  | Direct Preference Optimization (TRL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MQdsPxFHeZweGsNx4RH7Ia8lG8PiGE1t?usp=sharing) |
||||
| [LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B) | Supervised Fine Tuning (TRL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j5Hk_SyBb2soUsuhU0eIEA9GwLNRnElF?usp=sharing) |
|  | Supervised Fine Tuning (Axolotl) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/155lr5-uYsOJmZfO6_QZPjbs8hA_v8S7t?usp=sharing) |
|  | Supervised Fine Tuning (Unsloth) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HROdGaPFt1tATniBcos11-doVaH7kOI3?usp=sharing) |
|  | Direct Preference Optimization (TRL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MQdsPxFHeZweGsNx4RH7Ia8lG8PiGE1t?usp=sharing) |
||||
| [LFM2-700M](https://huggingface.co/LiquidAI/LFM2-700M) | Supervised Fine Tuning (TRL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j5Hk_SyBb2soUsuhU0eIEA9GwLNRnElF?usp=sharing) |
|  |Supervised Fine Tuning (Axolotl) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/155lr5-uYsOJmZfO6_QZPjbs8hA_v8S7t?usp=sharing) |
|  |Supervised Fine Tuning (Unsloth) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HROdGaPFt1tATniBcos11-doVaH7kOI3?usp=sharing) |
| | Direct Preference Optimization (TRL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MQdsPxFHeZweGsNx4RH7Ia8lG8PiGE1t?usp=sharing) |
||||
| [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) | Supervised Fine Tuning (TRL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j5Hk_SyBb2soUsuhU0eIEA9GwLNRnElF?usp=sharing) |
|  |Supervised Fine Tuning (Axolotl) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/155lr5-uYsOJmZfO6_QZPjbs8hA_v8S7t?usp=sharing) |
|  |Supervised Fine Tuning (Unsloth) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HROdGaPFt1tATniBcos11-doVaH7kOI3?usp=sharing) |
| | Direct Preference Optimization (TRL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MQdsPxFHeZweGsNx4RH7Ia8lG8PiGE1t?usp=sharing) |

Need a model for data extraction, RAG, tool use, or math reasoning? Start with our Nano checkpoints—they're already specialized for these tasks.

| Model | Use Cases |
|-------|-----------|
| •  [LFM2-1.2B-Extract](https://huggingface.co/LiquidAI/LFM2-1.2B-Extract)<br> • [LFM2-350M-Extract](https://huggingface.co/LiquidAI/LFM2-350M-Extract) | • Extracting invoice details from emails into structured JSON<br>• Converting regulatory filings into XML for compliance systems<br>• Transforming customer support tickets into YAML for analytics pipelines<br>• Populating knowledge graphs with entities and attributes from unstructured reports |
| [LFM2-1.2B-RAG](https://huggingface.co/LiquidAI/LFM2-1.2B-RAG) | • Chatbot to ask questions about the documentation of a particular product.<br> • Custom support with an internal knowledge base to provide grounded answers. <br> • Academic research assistant with multi-turn conversations about research papers and course materials.|
| [LFM2-1.2B-Tool](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool)| • Mobile and edge devices requiring instant API calls, database queries, or system integrations without cloud dependency.<br> • Real-time assistants in cars, IoT devices, or customer support, where response latency is critical. <br> • Resource-constrained environments like embedded systems or battery-powered devices needing efficient tool execution.|
| [LFM2‑350M‑Math](https://huggingface.co/LiquidAI/LFM2-350M-Math)| • Mathematical problem solving.<br> • Reasoning tasks.|

> [!NOTE]
>
> The supported languages for these models are: English, Arabic, Chinese, French, German, Japanese, Korean, Portuguese, and Spanish.
> 
> **Need support for another language?**
> 
> [Join the Liquid AI Discord Community](https://discord.com/invite/liquid-ai) and request it! Our community is working on expanding language support, and your input helps us prioritize which languages to tackle next. Connect with fellow developers, share your use cases, and collaborate on multilingual AI solutions.
> 
> [![Join Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.com/invite/liquid-ai)

### LFM2-VL (Text+Image to Text)

LFM2-VL is our first series of vision-language models, designed for on-device deployment.

| Model | Technique |  |
|-------|-----------|---------------|
| [LFM2-VL-1.6B](https://huggingface.co/LiquidAI/LFM2-VL-1.6B) | Supervised Fine Tuning (TRL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1csXCLwJx7wI7aruudBp6ZIcnqfv8EMYN?usp=sharing) |
| [LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) | Supervised Fine Tuning (TRL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1csXCLwJx7wI7aruudBp6ZIcnqfv8EMYN?usp=sharing) |


## Deploy to an edge device

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
| Food Images Fine-Tuning | [▶️ Go to the repo](https://github.com/benitomartin/food-images-finetuning) ![GitHub Repo stars](https://img.shields.io/github/stars/benitomartin/food-images-finetuning) |
| Meeting Intelligence CLI | [▶️ Go to the repo](https://github.com/csanghvi-stripe/meeting-prompter) ![GitHub Repo stars](https://img.shields.io/github/stars/csanghvi-stripe/meeting-prompter) |
| Private Doc Q&A: On-device document Q&A with RAG and voice input | [▶️ Go to the repo](https://github.com/csanghvi-stripe/private-doc-qa) ![GitHub Repo stars](https://img.shields.io/github/stars/csanghvi-stripe/private-doc-qa) |
| LFM-Scholar: Automated literature review agent capable of finding and citing papers | [▶️ Go to the repo](https://github.com/gyunggyung/LFM-Scholar) ![GitHub Repo stars](https://img.shields.io/github/stars/gyunggyung/LFM-Scholar) |
| LFM2-KoEn-Tuning: Fine-tuned LFM2 1.2B model specialized for Korean-English translation | [▶️ Go to the repo](https://github.com/gyunggyung/LFM2-KoEn-Tuning) ![GitHub Repo stars](https://img.shields.io/github/stars/gyunggyung/LFM2-KoEn-Tuning) |

## Contributing

We welcome contributions!

- Open a PR with a link to your project github repo in the `Examples built by our community` section.


## Support 💬

- 📖 [Liquid AI Documentation](https://leap.liquid.ai/docs)
- 💬 [Join our community on Discord](https://discord.com/invite/liquid-ai)
