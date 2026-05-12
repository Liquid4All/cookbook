"""PyTorch-side smoke test of the fine-tuned LFM2.5-Audio checkpoint.

Bypasses llama.cpp entirely. Loads the fine-tuned `model.safetensors` from the
Modal training output volume into `LFM2AudioModel.from_pretrained`, drains a
few rows from the published test split, and runs inference end-to-end. Used
to disambiguate whether the published GGUF's runtime failure is a problem in
the weights themselves (in which case PyTorch will also misbehave) or in the
llama.cpp PR-18641 GGUF path (in which case PyTorch will produce sensible
function calls).

Usage:
    HF_TOKEN=hf_... uv run --group finetune python scripts/smoke_test_pytorch.py
"""

from __future__ import annotations

import io
import os

import modal
from dotenv import load_dotenv

load_dotenv()

app = modal.App("lfm2-audio-pytorch-smoke")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install_from_pyproject("pyproject.toml")
)

train_vol = modal.Volume.from_name("lfm2-training-output")
secrets = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}),
]


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/checkpoints": train_vol},
    secrets=secrets,
    timeout=1200,
    serialized=True,
)
def smoke_test() -> str:
    import torch
    import torchaudio
    from accelerate import load_checkpoint_in_model
    from datasets import load_dataset

    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
    from liquid_audio.processor import ChatState

    print("Loading processor + base model from upstream...", flush=True)
    processor = LFM2AudioProcessor.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B", device="cuda"
    ).eval()
    model = LFM2AudioModel.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B", device="cuda"
    ).eval()

    print("Overlaying fine-tuned weights from volume...", flush=True)
    load_checkpoint_in_model(
        model, "/checkpoints/ohf-voice-20260512-000928/final/model.safetensors"
    )

    print("Loading 5 samples from published test split...", flush=True)
    ds = load_dataset("Paulescu/OHF-Voice-audio-20260504", split="test").select(range(5))

    out_lines: list[str] = []
    for i, row in enumerate(ds):
        gt = row["text_chat"][1]["content"][0]["text"]
        audio_bytes = row["audio_chat"][0]["content"][0]["audio"]

        chat = ChatState(processor)
        chat.new_turn("user")
        wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
        chat.add_audio(wav, sr)
        chat.end_turn()
        chat.new_turn("assistant")

        with torch.no_grad():
            tokens = [t.item() for t in model.generate_sequential(**chat, max_new_tokens=64)]
        pred = processor.text.decode(tokens, skip_special_tokens=True).strip()

        out_lines.append(f"[{i + 1}/5]")
        out_lines.append(f"  gt:   {gt}")
        out_lines.append(f"  pred: {pred}")
        out_lines.append("")

    report = "\n".join(out_lines)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    with app.run():
        result = smoke_test.remote()
        print("\n=== final report ===")
        print(result)
