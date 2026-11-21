"""LFM3-1.7B QLoRA training on a real dataset (smoltalk) for AMD RDNA1.

Verified on RX 5700 XT: 20 steps, seq=512, batch=2, grad_accum=2.
Loss drops from ~1.70 to ~1.16, grad_norm stays finite, peak VRAM 2.47 GiB,
and the saved adapter reloads successfully.
"""
import os

os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "LiquidAI/LFM3-1.7B"
OUT_DIR = "./out/lfm3-qlora-realdata"
N_SAMPLES = 64
MAX_LEN = 512


def main():
    print(f"[env] torch={torch.__version__}, device={torch.cuda.get_device_name(0)}")

    stream = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    rows = []
    for row in stream:
        rows.append(row)
        if len(rows) >= N_SAMPLES:
            break
    dataset = Dataset.from_list(rows)
    print(f"[data] {len(dataset)} examples loaded")

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

    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=OUT_DIR,
        max_steps=20,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_length=MAX_LEN,
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
    print(f"[train] peak VRAM={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

    adapter_dir = f"{OUT_DIR}/adapter"
    trainer.save_model(adapter_dir)

    # PeftModel loads adapters in eval mode by default; that is expected.
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True,
    )
    reloaded = PeftModel.from_pretrained(base, adapter_dir)
    print("[verify] adapter reloaded successfully")


if __name__ == "__main__":
    main()
