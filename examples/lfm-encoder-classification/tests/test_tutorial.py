import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast

from train import (
    DocumentClassifier,
    compute_metrics,
    document_text,
    load_config,
    multi_hot,
    tune_thresholds,
)

LABELS = ("account", "billing", "shipping", "technical")


def test_config_data_metrics_and_thresholds() -> None:
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    assert tuple(config["dataset"]["labels"]) == LABELS
    assert document_text(["First paragraph.", "Second paragraph."]) == (
        "First paragraph.\n\nSecond paragraph."
    )
    assert multi_hot(["billing", "technical"], LABELS) == [0.0, 1.0, 0.0, 1.0]

    targets = np.zeros((4, 4), dtype=int)
    targets[:2, 0] = 1
    probabilities = np.full((4, 4), 0.01)
    probabilities[:2, 0] = [0.4, 0.35]
    probabilities[2:, 0] = [0.2, 0.1]
    logits = np.log(probabilities / (1.0 - probabilities))
    assert compute_metrics(np.where(targets, 10.0, -10.0), targets, LABELS)["micro_f1"] == 1.0
    report = tune_thresholds(logits, targets, LABELS)
    assert report["fixed_0_5"]["metrics"]["micro_f1"] == 0.0
    assert report["selected"]["metrics"]["micro_f1"] == 1.0


def test_model_checkpoint_round_trip(tmp_path: Path) -> None:
    base = tmp_path / "base"
    checkpoint = tmp_path / "classifier"
    BertForMaskedLM(
        BertConfig(
            vocab_size=32,
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    ).save_pretrained(base)
    model = DocumentClassifier.from_base(str(base), ("a", "b"), True).eval()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    with torch.no_grad():
        expected = model(**inputs).logits
    model.save_pretrained(checkpoint)
    restored = DocumentClassifier.load_checkpoint(checkpoint, str(base), True).eval()
    with torch.no_grad():
        actual = restored(**inputs).logits
    assert torch.allclose(expected, actual)


def test_training_command_runs_offline_with_a_tiny_model(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    base_model = tmp_path / "base-model"
    output = tmp_path / "classifier"
    base_model.mkdir()

    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "account",
        "billing",
        "shipping",
        "technical",
        "problem",
    ]
    vocab_path = base_model / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(vocab_file=str(vocab_path)).save_pretrained(base_model)
    BertForMaskedLM(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            pad_token_id=0,
        )
    ).save_pretrained(base_model)

    rows = [
        {"text": "account problem", "labels": ["account"]},
        {"text": "billing problem", "labels": ["billing"]},
        {"text": "shipping problem", "labels": ["shipping"]},
        {"text": "technical problem", "labels": ["technical"]},
    ]
    data_files = {}
    for split, split_rows in {
        "train": rows * 2,
        "validation": rows,
        "test": rows,
    }.items():
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in split_rows))
        data_files[split] = str(path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"local_path": str(base_model)},
                "dataset": {
                    "source": {"type": "json", "data_files": data_files},
                    "text_column": "text",
                    "labels_column": "labels",
                    "labels": list(LABELS),
                },
                "training": {
                    "output_dir": str(output),
                    "max_length": 32,
                    "epochs": 1,
                    "train_batch_size": 2,
                    "eval_batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "precision": "fp32",
                },
            }
        )
    )

    completed = subprocess.run(
        [sys.executable, str(project_root / "train.py"), "--config", str(config_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "model.safetensors").is_file()
    assert (output / "thresholds.json").is_file()
    assert "selected" in json.loads((output / "validation_results.json").read_text())
