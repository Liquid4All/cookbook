"""Preprocess the OHF-Voice train split into the on-disk tensor format that
liquid_audio.trainer.Trainer expects.

For each row in `Paulescu/OHF-Voice-audio-20260504` (train split only; the test
split is reserved for evaluation per docs/adr/0001), we materialise the
audio + ground-truth chat as an LFM2AudioChatMapper output, write it through
`preprocess_dataset` to an Arrow dataset on disk, and store the result under
`--output-path`. Once preprocessing has run, `scripts/train.py` consumes the
output directory directly via `LFM2DataLoader`.

Run locally (slow, requires CUDA or MPS):
    uv run python scripts/preprocess_ohf_voice.py --output-path data/ohf_voice/train

Run on Modal with an A100 (much faster, persists to the modal volume named
ohf-voice-data):
    HF_TOKEN=hf_... uv run --group finetune python scripts/preprocess_ohf_voice.py --modal

Vendored from liquid-audio-staging on 2026-05-11.
  source : examples/preprocess_ohf_voice.py
  branch : examples/audio-to-function-calling
  commit : 376b06a10386b0887b320122b13d2d99378c19ea

Adaptations from upstream:
  - Dataset id updated from `LiquidAI/OHF-Voice-audio-20260504` to
    `Paulescu/OHF-Voice-audio-20260504` (the published 95/5 fork), and the
    iterator is hard-pinned to the `train` split so preprocessing cannot
    accidentally touch test data.
  - `.add_local_python_source("liquid_audio")` removed from the Modal image
    build: this cookbook installs `liquid-audio` from PyPI rather than from a
    local editable workspace, so there is no local source to add.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator

from datasets import load_dataset
from dotenv import load_dotenv

from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment

load_dotenv()

DATASET_REPO = "Paulescu/OHF-Voice-audio-20260504"

# The `llama-liquid-audio-server` runtime enforces a closed allow-list of
# system prompts (the strings in `liquid_audio_chat.py`'s SYSTEM_PROMPTS:
# `"Perform ASR."`, the various `"Perform TTS. ..."`, and the interleaved
# variant). Empty / missing / arbitrary system messages get rejected. To make
# the fine-tune effective at inference, we therefore have to train with the
# *exact* system prompt the server will be configured to send -- otherwise
# the inference-time prompt acts as a pretrained behavioral trigger
# ("Perform ASR." anchors the model in transcription mode) that overrides
# the fine-tune. Empirically verified on 2026-05-12: a 500-step run trained
# without a system prompt produced correct function calls in PyTorch (where
# the chat shape can match the training shape) but produced plain
# transcriptions through the GGUF server (which forces the prompt in).
SYSTEM_PROMPT = "Perform ASR."


class OHFVoiceIterator:
    """Yields one list-of-ChatMessage per row of the train split.

    Each emitted chat is prefixed with a `system` turn carrying the inference-
    time system prompt (`SYSTEM_PROMPT`). The rest of the messages come from
    the dataset's `audio_chat` field, which has only user + assistant turns.
    """

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        ds = load_dataset(DATASET_REPO, split="train")
        for row in ds:
            messages: list[ChatMessage] = [
                ChatMessage(
                    role="system",
                    content=[TextSegment(text=SYSTEM_PROMPT)],
                )
            ]
            for msg in row["audio_chat"]:
                segments: list[TextSegment | AudioSegment] = []
                for item in msg["content"]:
                    if item["modality"] == "audio" and item["audio"]:
                        segments.append(AudioSegment(audio=item["audio"]))
                    elif item["modality"] == "text" and item["text"]:
                        segments.append(TextSegment(text=item["text"]))
                messages.append(ChatMessage(role=msg["role"], content=segments))
            yield messages


def run_preprocessing(output_path: str, max_context_length: int, device: str) -> None:
    processor = LFM2AudioProcessor.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B", device=device
    ).eval()
    mapper = LFM2AudioChatMapper(processor)
    data = OHFVoiceIterator()
    preprocess_dataset(
        data=data,
        output_path=output_path,
        mapper=mapper,
        max_context_length=max_context_length,
    )


def run_on_modal(output_path: str, max_context_length: int) -> None:
    import modal

    app = modal.App("ohf-voice-preprocess")

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg")
        .pip_install_from_pyproject("pyproject.toml")
    )

    vol = modal.Volume.from_name("ohf-voice-data", create_if_missing=True)
    secrets = [
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}),
    ]

    @app.function(
        gpu="A100",
        image=image,
        volumes={"/output": vol},
        secrets=secrets,
        timeout=7200,
        serialized=True,
    )
    def remote_preprocess() -> None:
        run_preprocessing(
            output_path="/output/train",
            max_context_length=max_context_length,
            device="cuda",
        )

    with app.run():
        remote_preprocess.remote()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess OHF-Voice for LFM2.5-Audio fine-tuning."
    )
    parser.add_argument(
        "--output-path",
        default="data/ohf_voice/train",
        help="Output path for preprocessed data (default: data/ohf_voice/train).",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=512,
        help="Skip samples whose tokenised length exceeds this many tokens (default: 512).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for local preprocessing: cuda, mps, or cpu (default: cuda).",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Run preprocessing on Modal with an A100 (writes to the ohf-voice-data volume).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.modal:
        run_on_modal(args.output_path, args.max_context_length)
    else:
        run_preprocessing(args.output_path, args.max_context_length, args.device)
