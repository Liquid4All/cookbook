"""LFM3-1.7B QLoRA (NF4 4bit) smoke test for AMD RDNA1 / RX 5700 XT.

Verified on Windows 11, ROCm 7.14.1, gfx1010. Peak VRAM ~1.42 GiB.
Run from this directory with the environment described in README.md.
"""
import os

# Required on gfx1010 before torch/triton are imported: default Triton buffer
# ops report hipSuccess but write nothing on RDNA1.
os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "LiquidAI/LFM3-1.7B"
OUT_DIR = "./out/lfm3-qlora-smoke"


def main():
    print(f"[env] torch={torch.__version__}, device={torch.cuda.get_device_name(0)}")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="auto", quantization_config=quant,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Two synthetic examples only verify that the training path runs.
    dataset = Dataset.from_list([
        {"messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]},
        {"messages": [
            {"role": "user", "content": "What is two plus two?"},
            {"role": "assistant", "content": "Four."},
        ]},
    ])

    peft_config = LoraConfig(
        r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=OUT_DIR,
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_length=256,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=dataset, peft_config=peft_config, args=args,
    )
    trainer.train()
    trainer.save_model(f"{OUT_DIR}/adapter")
    print(f"[done] peak VRAM={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
