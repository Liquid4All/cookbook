"""Classify one document with the fine-tuned checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from train import DocumentClassifier, load_config, model_reference, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_ref, local_only = model_reference(config)
    checkpoint = project_path(config["training"].get("output_dir", "local/classifier"))
    labels = tuple(config["dataset"]["labels"])
    text = args.text if args.text is not None else args.file.expanduser().read_text()

    metadata = json.loads((checkpoint / "run_metadata.json").read_text())
    thresholds = json.loads((checkpoint / "thresholds.json").read_text())["thresholds"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = DocumentClassifier.load_checkpoint(
        checkpoint,
        model_ref,
        local_files_only=local_only,
    ).eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=int(metadata["max_length"]),
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(**inputs).logits)[0].cpu().tolist()

    result = {
        "predicted_labels": [
            label
            for label, probability, threshold in zip(labels, probabilities, thresholds, strict=True)
            if probability >= threshold
        ],
        "probabilities": dict(zip(labels, probabilities, strict=True)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
