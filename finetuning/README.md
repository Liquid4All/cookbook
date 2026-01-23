# Fine-tuning recipes

A collection of notebooks and examples to fine-tune Liquid AI models using different techniques.

- [Text-to-text models](#text-to-text-models)
- [Vision Language Models](#vision-language-models)

## Text-to-text models

| Fine-tuning technique | Models | Link |
|---|---|---|
| Continued Pre-Training (CPT)| LFM2.5-1.2B-Base| [General text-completion pre-training](https://colab.research.google.com/drive/10fm7eNMezs-DSn36mF7vAsNYlOsx9YZO?usp=sharing)<br>[Cross-lingual pre-training](https://colab.research.google.com/drive/1gaP8yTle2_v35Um8Gpu9239fqbU7UgY8?usp=sharing) |
| Supervised fine-tuning (SFT) with LoRA | LFM2.5-1.2B-Base<br>LFM2.5-1.2B-Instruct<br>LFM2.5-1.2B-Thinking<br>LFM2-2.6B-Exp<br>LFM2-2.6B<br>LFM2-8B-A1B<br>LFM2-700M<br>LFM2-350M | [With TRL](https://colab.research.google.com/drive/1j5Hk_SyBb2soUsuhU0eIEA9GwLNRnElF?usp=sharing)<br>[With Unsloth](https://colab.research.google.com/drive/1vGRg4ksRj__6OLvXkHhvji_Pamv801Ss?usp=sharing) |
| Direct Preference Optimization (DPO) with LoRA | LFM2.5-1.2B-Base<br>LFM2.5-1.2B-Instruct<br>LFM2.5-1.2B-Thinking<br>LFM2-2.6B-Exp<br>LFM2-2.6B<br>LFM2-8B-A1B<br>LFM2-700M<br>LFM2-350M | [With TRL](https://colab.research.google.com/drive/1MQdsPxFHeZweGsNx4RH7Ia8lG8PiGE1t?usp=sharing) |
| Group Relative Policy Optimization (GRPO) with LoRA | LFM2.5-1.2B-Base<br>LFM2.5-1.2B-Instruct<br>LFM2.5-1.2B-Thinking<br>LFM2-2.6B-Exp<br>LFM2-2.6B<br>LFM2-8B-A1B<br>LFM2-700M<br>LFM2-350M | Turn a non-reasoning model into a reasoning model with [Unsloth](https://colab.research.google.com/drive/1mIikXFaGvcW4vXOZXLbVTxfBRw_XsXa5?usp=sharing) or [TRL]() |

## Vision Language Models

| Fine-tuning technique | Models | Link |
|---|---|---|
| Supervised fine-tuning (SFT) with LoRA | LFM2.5-VL-1.6B<br>LFM2-VL-3B<br>LFM2-VL-450M | [Boost image classification accuracy](../examples/car-maker-identification/README.md)|
