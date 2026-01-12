# LFM2 English-Korean Translation: Efficient Bidirectional Translation with 1.2B Parameters

[![Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.com/invite/liquid-ai)

## What's inside?

An efficient bidirectional translation system powered by Liquid AI's LFM2 1.2B model fine-tuned for Korean-English translation. This project demonstrates how domain-specific fine-tuning can achieve superior performance compared to models 3x larger, outperforming Google's Gemma-3 4B and Alibaba's Qwen3 4B on the Flores-200 benchmark.

Key features:
- **Automatic language detection** - Intelligently detects input language and translates accordingly
- **High-quality translation** - CHrF++ 34.61 / BLEU 13.21 on Flores-200 benchmark
- **Efficient inference** - Runs on modest hardware with merged adapters for speed
- **Easy-to-use CLI** - Simple command-line interface powered by Fire

*This project was built and released by [Kiwoong Yeom](https://www.linkedin.com/in/kiwoong-yeom/) with the support of [Maxime Labone](https://www.linkedin.com/in/maxime-labonne/).*
*[Link to the original announcement on LinkedIn](https://www.linkedin.com/posts/activity-7406831565210583040-2B9p?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAqH-bMBMXBij-GI7SN8H4dk_E4j4k19f_w)*

## Quick Start

1. Clone the repository
```bash
git clone https://github.com/Liquid4All/cookbook.git
cd cookbook/examples/lfm2-english-to-korean
```

2. Install dependencies
```bash
uv sync
```

3. Run translation example
```bash
uv run python main.py --text "$(cat linkedin_post.txt)" --max-new-tokens 1024
```

## Understanding the Architecture

The system uses a two-stage training approach:

1. **Supervised Fine-tuning (SFT)**: 280K high-quality Korean-English parallel datasets establish the translation foundation
2. **Reinforcement Learning (RL)**: GRPO optimization with 10K additional samples refines translation quality

### Model Components

- **Base Model**: `gyung/lfm2-1.2b-koen-mt-v6.4-merged` - SFT fine-tuned LFM2 1.2B
- **Adapter**: `gyung/lfm2-1.2b-koen-mt-v8-rl-10k-adapter` - LoRA adapter trained with GRPO
- **Automatic Detection**: Regular expression pattern matching for Korean text (Hangul syllables, Jamo)

The system automatically detects the input language and applies the appropriate translation direction, supporting both English→Korean and Korean→English translation.

## CLI Usage

```bash
uv run python main.py [OPTIONS]

Options:
  --text TEXT                    Text to translate (required)
  --model-name TEXT             Base model name (default: gyung/lfm2-1.2b-koen-mt-v6.4-merged)
  --adapter-name TEXT           Adapter name (default: gyung/lfm2-1.2b-koen-mt-v8-rl-10k-adapter)
  --max-new-tokens INTEGER     Maximum tokens to generate (default: 256)
  --temperature FLOAT          Sampling temperature (default: 0.3)
  --min-p FLOAT               Minimum probability threshold (default: 0.15)
  --repetition-penalty FLOAT   Repetition penalty (default: 1.05)
```

### Example Usage

Translate English to Korean:
```bash
uv run python main.py --text "Hello, how are you today?"
```

Translate Korean to English:
```bash
uv run python main.py --text "안녕하세요, 오늘 어떻게 지내세요?"
```

Process a file:
```bash
uv run python main.py --text "$(cat your_file.txt)" --max-new-tokens 1024
```

## Performance Benchmarks

On Flores-200 benchmark (1,012 samples):

| Model | Parameters | CHrF++ | BLEU |
|-------|------------|--------|------|
| **LFM2-KoEn-v8-rl** | **1.2B** | **34.61** | **13.21** |
| Gemma-3-4B | 4B | 32.83 | 11.36 |
| Qwen3-4B | 4B | 25.62 | 7.46 |

The 1.2B parameter model outperforms models 3x larger, demonstrating that specialized training matters more than raw parameter count.

## Further Improvements

Next steps for enhanced performance and efficiency:

- **Speed optimization** with quantization techniques (GGUF, AWQ, GPTQ)
- **llama.cpp integration** for faster CPU inference
- **Full parameter RL training** with expanded compute resources
- **Length normalization removal** based on recent Qwen team findings
- **Extended dataset training** with 200K SFT + 25K RL samples

### Performance Optimization

The current implementation uses adapter merging for faster inference. Future improvements include:

- Quantized model variants for resource-constrained environments
- Streaming inference for real-time translation
- Batch processing for large document translation

## Need help?

Join the [Liquid AI Discord Community](https://discord.com/invite/liquid-ai) and ask.
[![Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.com/invite/liquid-ai)
