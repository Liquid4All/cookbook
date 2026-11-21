"""LFM3-1.7B LoRA smoke test in fp16 — reproduces the NaN gradient issue on RDNA1.

Keep this script as a negative control: on RX 5700 XT it reports finite loss but
grad_norm=NaN. Do not use this configuration on gfx1010.
"""
import os

os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "LiquidAI/LFM3-1.7B"
OUT_DIR = "./out/lfm3-lora-fp16"


def main():
    print(f"[env] torch={torch.__version__}, device={torch.cuda.get_device_name(0)}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="auto", dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

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
        fp16=True,
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
