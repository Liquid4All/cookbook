# Fine-tuning recipes for LFM2 models

We recommend you fine-tune LFM2 models if you are looking to improve the performance of a model for your specific use case.

In this page you can find a collection of notebooks and examples to fine-tune LFM2 models using different techniques.

- [Text-to-text models](#text-to-text-models)
- [Vision Language Models](#vision-language-models)
- Audio models (COMING SOON)

## Text-to-text models

Models:
- `LFM2.5-1.2B-Base`
- `LFM2.5-1.2B-Instruct`
- `LFM2.5-1.2B-Thinking`
- `LFM2-2.6B-Exp`
- `LFM2-2.6B`
- `LFM2-8B-A1B`
- `LFM2-700M`
- `LFM2-350M`

| Fine-tuning technique | Link |
|---|---|
| Continued Pre-Training (CPT) (use only with `LFM2.5-1.2B-Base`)| [Text-completion pre-training](https://colab.research.google.com/drive/10fm7eNMezs-DSn36mF7vAsNYlOsx9YZO?usp=sharing)<br>[Cross-lingual pre-training](https://colab.research.google.com/drive/1gaP8yTle2_v35Um8Gpu9239fqbU7UgY8?usp=sharing) |
| Supervised fine-tuning (SFT) with LoRA | [With TRL](https://colab.research.google.com/drive/1j5Hk_SyBb2soUsuhU0eIEA9GwLNRnElF?usp=sharing)<br>[With Unsloth](https://colab.research.google.com/drive/1vGRg4ksRj__6OLvXkHhvji_Pamv801Ss?usp=sharing) |
| Direct Preference Optimization (DPO) with LoRA | [With TRL](https://colab.research.google.com/drive/1MQdsPxFHeZweGsNx4RH7Ia8lG8PiGE1t?usp=sharing) |
| Group Relative Policy Optimization (GRPO) with LoRA | - Turn a non-reasoning model into a reasoning model with [Unsloth](https://colab.research.google.com/drive/1mIikXFaGvcW4vXOZXLbVTxfBRw_XsXa5?usp=sharing) or [TRL]() <br> - [Boost browser control tasks with OpenEnv and GRPO](../examples/browser-control/README.md) |

## Vision Language Models

Models:

- `LFM2.5-VL-1.6B`
- `LFM2-VL-3B`
- `LFM2-VL-450M`

| Fine-tuning technique | Link |
|---|---|
| Supervised fine-tuning (SFT) with LoRA | - [OCR with Unsloth](https://colab.research.google.com/drive/1FaR2HSe91YDe88TG97-JVxMygl-rL6vB?usp=sharing#scrollTo=vITh0KVJ10qX)<br> - [Medical Vision Fine-tuning with TRL](https://colab.research.google.com/drive/10530_jt_Joa5zH2wgYlyXosypq1R7PIz?usp=sharing) <br> - [Car image classification example](../examples/car-maker-identification/README.md) |