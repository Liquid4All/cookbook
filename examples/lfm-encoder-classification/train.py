"""Fine-tune LFM2.5-Encoder for multi-label document classification.

The tutorial is intentionally contained in one file. Read it top to bottom:
configuration and data, metrics, model, then training.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Lfm2BidirectionalConfig,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent


# 1. Configuration and data -------------------------------------------------


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("The configuration must contain a YAML mapping.")
    for section in ("model", "dataset", "training"):
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
    labels = config["dataset"].get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(x, str) for x in labels):
        raise ValueError("dataset.labels must be a non-empty list of strings.")
    if len(labels) != len(set(labels)):
        raise ValueError("dataset.labels must not contain duplicates.")
    config["_path"] = config_path
    return config


def model_reference(config: dict[str, Any]) -> tuple[str, bool]:
    """Return either an optional local model directory or the Hub model ID."""
    local_path = config["model"].get("local_path")
    if local_path:
        path = project_path(local_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Local model not found: {path}")
        return str(path), True
    return str(config["model"]["id"]), False


def load_splits(config: dict[str, Any]) -> Any:
    source = config["dataset"]["source"]
    if source["type"] == "json":
        files = {split: str(project_path(path)) for split, path in source["data_files"].items()}
        dataset = load_dataset("json", data_files=files)
    elif source["type"] == "huggingface":
        dataset = load_dataset(source["id"], source.get("name"))
    else:
        raise ValueError("dataset.source.type must be 'json' or 'huggingface'.")

    required = {"train", "validation", "test"}
    if missing := required - set(dataset):
        raise ValueError(f"Dataset is missing splits: {', '.join(sorted(missing))}")
    text_column = config["dataset"].get("text_column", "text")
    labels_column = config["dataset"].get("labels_column", "labels")
    for split in required:
        columns = dataset[split].column_names
        if text_column not in columns or labels_column not in columns:
            raise ValueError(f"{split} must contain '{text_column}' and '{labels_column}' columns.")
    return dataset


def document_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        paragraphs = list(value)
        if all(isinstance(paragraph, str) for paragraph in paragraphs):
            return "\n\n".join(paragraphs)
    raise ValueError("Each document must be a string or a list of paragraph strings.")


def multi_hot(
    values: Any,
    labels: tuple[str, ...],
    source_label_names: Sequence[str] | None = None,
) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("Each labels value must be a list.")

    normalized = []
    for value in values:
        if isinstance(value, str):
            normalized.append(value)
        elif isinstance(value, (int, np.integer)) and source_label_names is not None:
            index = int(value)
            if not 0 <= index < len(source_label_names):
                raise ValueError(f"Label ID {index} is outside the dataset's ClassLabel range.")
            normalized.append(source_label_names[index])
        else:
            raise ValueError(
                "Labels must be strings, or integer IDs backed by Hugging Face ClassLabel metadata."
            )

    unknown = sorted(set(normalized) - set(labels))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    selected = set(normalized)
    return [float(label in selected) for label in labels]


# 2. Metrics and threshold tuning -------------------------------------------


def _metric_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def compute_metrics(
    logits: Any,
    label_ids: Any,
    labels: tuple[str, ...],
    threshold: Any = 0.5,
) -> dict[str, float]:
    logits = np.asarray(logits)
    target = np.asarray(label_ids).astype(int)
    if logits.shape != target.shape or logits.ndim != 2:
        raise ValueError(
            f"Expected matching [examples, labels] arrays, got {logits.shape} and {target.shape}."
        )
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    predicted = (probabilities >= threshold).astype(int)

    micro = precision_recall_fscore_support(target, predicted, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(target, predicted, average="macro", zero_division=0)
    per_label = precision_recall_fscore_support(target, predicted, average=None, zero_division=0)
    per_label_ap = [
        average_precision_score(target[:, index], probabilities[:, index])
        if target[:, index].any()
        else 0.0
        for index in range(len(labels))
    ]
    metrics = {
        "micro_precision": float(micro[0]),
        "micro_recall": float(micro[1]),
        "micro_f1": float(micro[2]),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "exact_match_accuracy": float(accuracy_score(target, predicted)),
        "hamming_loss": float(hamming_loss(target, predicted)),
        "micro_average_precision": float(
            average_precision_score(target, probabilities, average="micro")
        ),
        "macro_average_precision": float(np.mean(per_label_ap)),
    }
    for index, label in enumerate(labels):
        key = _metric_key(label)
        metrics.update(
            {
                f"label_{key}_precision": float(per_label[0][index]),
                f"label_{key}_recall": float(per_label[1][index]),
                f"label_{key}_f1": float(per_label[2][index]),
                f"label_{key}_average_precision": float(per_label_ap[index]),
                f"label_{key}_support": float(target[:, index].sum()),
            }
        )
    return metrics


def tune_thresholds(
    logits: Any,
    label_ids: Any,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    """Select thresholds with validation data, never with test data."""
    logits = np.asarray(logits)
    target = np.asarray(label_ids).astype(int)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    candidates = np.arange(0.05, 0.951, 0.01)

    def binary_f1(predicted: Any, truth: Any) -> float:
        return float(
            precision_recall_fscore_support(
                truth.ravel(), predicted.ravel(), average="binary", zero_division=0
            )[2]
        )

    global_threshold = max(
        candidates,
        key=lambda value: (binary_f1(probabilities >= value, target), -abs(value - 0.5)),
    )
    per_label_thresholds = np.full(len(labels), 0.5)
    for index in range(len(labels)):
        if target[:, index].any():
            per_label_thresholds[index] = max(
                candidates,
                key=lambda value: (
                    binary_f1(probabilities[:, index] >= value, target[:, index]),
                    -abs(value - 0.5),
                ),
            )

    reports = {
        "fixed_0_5": {
            "thresholds": [0.5] * len(labels),
            "metrics": compute_metrics(logits, target, labels),
        },
        "global": {
            "thresholds": [float(global_threshold)] * len(labels),
            "metrics": compute_metrics(logits, target, labels, global_threshold),
        },
        "per_label": {
            "thresholds": per_label_thresholds.tolist(),
            "metrics": compute_metrics(logits, target, labels, per_label_thresholds),
        },
    }
    selected_name = max(
        ("global", "per_label"),
        key=lambda name: (
            reports[name]["metrics"]["micro_f1"],
            reports[name]["metrics"]["macro_f1"],
        ),
    )
    reports["selected"] = {"strategy": selected_name, **reports[selected_name]}
    return reports


# 3. Model ------------------------------------------------------------------


def load_base_model(
    model_ref: str,
    labels: tuple[str, ...],
    local_files_only: bool,
) -> torch.nn.Module:
    """Load the masked-LM checkpoint with Transformers' native classification head.

    The published encoder checkpoints still identify themselves as ``lfm2``. Constructing the
    bidirectional config explicitly keeps this example working until their Hub configs are updated
    to ``lfm2_bidirectional``.
    """
    config = Lfm2BidirectionalConfig.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
    )
    legacy_checkpoint = config.model_type != Lfm2BidirectionalConfig.model_type
    config.model_type = Lfm2BidirectionalConfig.model_type
    if legacy_checkpoint and hasattr(config, "auto_map"):
        del config.auto_map
    config.num_labels = len(labels)
    config.id2label = dict(enumerate(labels))
    config.label2id = {label: index for index, label in enumerate(labels)}
    config.problem_type = "multi_label_classification"
    config.architectures = ["Lfm2BidirectionalForSequenceClassification"]
    config.use_cache = False
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ref,
        config=config,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
        key_mapping={r"^lfm2\.": "model."} if legacy_checkpoint else None,
    )
    if legacy_checkpoint:
        # The input-only compatibility mapping must not be reversed when Trainer saves a checkpoint.
        model._weight_conversions = None
    return model


# 4. Training ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate test after the model and thresholds are final.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    config = load_config(args.config)
    labels = tuple(config["dataset"]["labels"])
    model_ref, local_only = model_reference(config)
    dataset = load_splits(config)
    training = config["training"]

    max_length = int(training.get("max_length", 512))
    epochs = float(training.get("epochs", 3))
    accumulation = int(training.get("gradient_accumulation_steps", 8))
    output = project_path(training.get("output_dir", "local/classifier"))

    if not 32 <= max_length <= 8192:
        raise ValueError("training.max_length must be between 32 and 8192.")
    precision = training.get("precision", "fp32")
    if precision in {"bf16", "fp16"} and not torch.cuda.is_available():
        raise RuntimeError(f"{precision} training requires CUDA; use fp32 on this machine.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        local_files_only=local_only,
    )
    text_column = config["dataset"].get("text_column", "text")
    labels_column = config["dataset"].get("labels_column", "labels")
    label_feature = dataset["train"].features[labels_column]
    source_label_names = getattr(getattr(label_feature, "feature", None), "names", None)

    def tokenize(split: Any) -> Any:
        def preprocess(batch: dict[str, list[Any]]) -> dict[str, Any]:
            encoded = tokenizer(
                [document_text(value) for value in batch[text_column]],
                truncation=True,
                max_length=max_length,
            )
            encoded["labels"] = [
                multi_hot(values, labels, source_label_names) for values in batch[labels_column]
            ]
            return encoded

        return split.map(preprocess, batched=True, remove_columns=split.column_names)

    tokenized_train = tokenize(dataset["train"])
    tokenized_validation = tokenize(dataset["validation"])
    tokenized_test = tokenize(dataset["test"]) if args.evaluate_test else None

    def collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        label_tensor = torch.tensor(
            [feature.pop("labels") for feature in features], dtype=torch.float32
        )
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = label_tensor
        return batch

    def trainer_metrics(prediction: EvalPrediction) -> dict[str, float]:
        logits = (
            prediction.predictions[0]
            if isinstance(prediction.predictions, tuple)
            else prediction.predictions
        )
        return compute_metrics(logits, prediction.label_ids, labels)

    model = load_base_model(model_ref, labels, local_only)
    model.accepts_loss_kwargs = False
    checkpointing = bool(training.get("gradient_checkpointing", False))
    if checkpointing:
        model.gradient_checkpointing_enable()
    elif max_length > 4096:
        print("Tip: enable gradient_checkpointing if long-context training runs out of memory.")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output,
            num_train_epochs=epochs,
            per_device_train_batch_size=int(training.get("train_batch_size", 1)),
            per_device_eval_batch_size=int(training.get("eval_batch_size", 1)),
            gradient_accumulation_steps=accumulation,
            learning_rate=float(training.get("learning_rate", 2e-5)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            warmup_steps=float(training.get("warmup_ratio", 0.1)),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="micro_average_precision",
            greater_is_better=True,
            bf16=precision == "bf16",
            fp16=precision == "fp16",
            dataloader_pin_memory=torch.cuda.is_available(),
            seed=int(training.get("seed", 42)),
            report_to=[],
        ),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=collate,
        compute_metrics=trainer_metrics,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    (output / "run_config.yaml").write_text(config["_path"].read_text())

    validation = trainer.predict(tokenized_validation, metric_key_prefix="validation")
    validation_logits = (
        validation.predictions[0]
        if isinstance(validation.predictions, tuple)
        else validation.predictions
    )
    threshold_report = tune_thresholds(validation_logits, validation.label_ids, labels)
    (output / "validation_results.json").write_text(json.dumps(threshold_report, indent=2) + "\n")
    (output / "thresholds.json").write_text(
        json.dumps(threshold_report["selected"], indent=2) + "\n"
    )
    (output / "run_metadata.json").write_text(
        json.dumps({"labels": labels, "max_length": max_length}, indent=2) + "\n"
    )
    selected = threshold_report["selected"]
    print(f"Validation micro-F1: {selected['metrics']['micro_f1']:.4f}")

    if args.evaluate_test:
        test = trainer.predict(tokenized_test, metric_key_prefix="test")
        test_logits = (
            test.predictions[0] if isinstance(test.predictions, tuple) else test.predictions
        )
        test_report = compute_metrics(
            test_logits,
            test.label_ids,
            labels,
            threshold=selected["thresholds"],
        )
        (output / "test_results.json").write_text(json.dumps(test_report, indent=2) + "\n")
        print(f"Test micro-F1: {test_report['micro_f1']:.4f}")

    print(f"Saved model and metrics to {output}")


if __name__ == "__main__":
    main()
