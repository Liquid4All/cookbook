#!/usr/bin/env python3
"""Prepare a Modal VLM SFT bundle with methane rows plus structured-output replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_schema(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("schema must be a mapping")
    return {str(key): str(value) for key, value in data.items()}


def build_system_prompt(schema: dict[str, str]) -> str:
    schema_yaml = yaml.safe_dump(schema, sort_keys=False, allow_unicode=True).strip()
    return f"Extract the following from the image:\n\n{schema_yaml}"


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")[:140] or "sample"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def parse_replay_arg(value: str) -> tuple[str, Path, int]:
    try:
        name, path, count = value.split(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "replay entries must use name:path:count"
        ) from exc
    return name, Path(path), int(count)


def extract_assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    content = messages[-1].get("content") if isinstance(messages[-1], dict) else None
    if isinstance(content, list):
        return " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    if isinstance(content, str):
        return content
    return ""


def validate_message_row(row: dict[str, Any], *, line_label: str) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{line_label}: missing messages")
    assistant_text = extract_assistant_text(row)
    if not assistant_text:
        raise ValueError(f"{line_label}: missing assistant text")
    try:
        json.loads(assistant_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{line_label}: assistant text is not JSON: {exc}") from exc


def rewrite_sample_id(row: dict[str, Any], prefix: str, index: int) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    sample_id = str(out.get("sample_id") or f"row_{index:06d}")
    out["sample_id"] = f"{prefix}:{sample_id}"
    return out


def minimal_training_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only columns the VLM loader needs.

    Hugging Face's JSON loader infers one Arrow schema for the full JSONL.
    Mixing methane metadata with replay metadata can introduce different nested
    structs or extra top-level columns, so the training file must stay minimal.
    """
    return {"sample_id": str(row["sample_id"]), "messages": row["messages"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--min-samples", type=int, default=10_000)
    parser.add_argument("--replay", action="append", type=parse_replay_arg, default=[])
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    schema_path = Path(args.schema).resolve()
    labels_path = Path(args.labels_jsonl).resolve()
    schema = load_schema(schema_path)
    schema_keys = list(schema)
    system_prompt = build_system_prompt(schema)

    dataset_dir = Path(args.bundle_root).resolve() / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_local_path = dataset_dir / "train.local.jsonl"
    train_remote_path = dataset_dir / "train.jsonl"
    manifest_path = dataset_dir / "manifest.json"

    rng = random.Random(args.seed)
    counts: dict[str, Counter[str]] = {
        "mix_source": Counter(),
        "augmentation": Counter(),
        "source_dataset": Counter(),
        "sensor": Counter(),
        "methane_plume_status": Counter(),
    }
    rows = 0
    staged_images = 0
    replay_details: list[dict[str, Any]] = []

    with labels_path.open(encoding="utf-8") as methane_f, train_local_path.open(
        "w", encoding="utf-8"
    ) as local_f, train_remote_path.open("w", encoding="utf-8") as remote_f:
        for line_no, line in enumerate(methane_f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = Path(row["image"]).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"line {line_no}: image not found: {image_path}")

            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            target = row.get("ground_truth")
            if set(target) != set(schema_keys):
                raise ValueError(f"line {line_no}: ground_truth keys do not match schema")

            sample_id = str(metadata.get("sample_id") or row.get("sample_id") or f"methane_{line_no:06d}")
            augmentation = str(metadata.get("augmentation") or "original")
            rel_name = f"{safe_name(sample_id)}{image_path.suffix.lower() or '.png'}"
            relative_image = Path("images") / "methane" / safe_name(augmentation) / rel_name
            staged_image = dataset_dir / relative_image
            link_or_copy(image_path, staged_image)
            staged_images += 1

            ordered_target = {key: target[key] for key in schema_keys}

            def make_messages(image_ref: str) -> dict[str, Any]:
                return {
                    "sample_id": f"methane:{sample_id}",
                    "messages": [
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "image", "image": image_ref}]},
                        {"role": "assistant", "content": [{"type": "text", "text": compact_json(ordered_target)}]},
                    ],
                }

            local_row = make_messages(str(staged_image))
            remote_row = make_messages(str(Path(args.remote_root) / relative_image))
            validate_message_row(remote_row, line_label=f"methane line {line_no}")
            local_f.write(compact_json(minimal_training_row(local_row)) + "\n")
            remote_f.write(compact_json(minimal_training_row(remote_row)) + "\n")

            rows += 1
            counts["mix_source"]["methane"] += 1
            counts["augmentation"][augmentation] += 1
            counts["source_dataset"][str(metadata.get("source_dataset") or "unknown")] += 1
            counts["sensor"][str(metadata.get("sensor") or "")] += 1
            counts["methane_plume_status"][str(ordered_target.get("methane_plume_status") or "unknown")] += 1

        for replay_name, replay_path, replay_count in args.replay:
            replay_rows = read_jsonl(replay_path)
            if len(replay_rows) < replay_count:
                raise ValueError(f"replay {replay_name}: {len(replay_rows)} rows < requested {replay_count}")
            selected = rng.sample(range(len(replay_rows)), replay_count)
            for out_index, source_index in enumerate(selected):
                replay_row = rewrite_sample_id(replay_rows[source_index], f"replay_{replay_name}", out_index)
                validate_message_row(replay_row, line_label=f"replay {replay_name} row {source_index}")
                local_f.write(compact_json(minimal_training_row(replay_row)) + "\n")
                remote_f.write(compact_json(minimal_training_row(replay_row)) + "\n")
                rows += 1
                counts["mix_source"][f"replay_{replay_name}"] += 1
            replay_details.append(
                {
                    "name": replay_name,
                    "path": str(replay_path),
                    "path_sha256": sha256_file(replay_path),
                    "available_rows": len(replay_rows),
                    "selected_rows": replay_count,
                }
            )

    if rows < args.min_samples:
        raise ValueError(f"prepared {rows} rows, below minimum {args.min_samples}")

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_dir": str(dataset_dir),
        "remote_root": args.remote_root,
        "schema": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "labels_jsonl": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "train_local_jsonl": str(train_local_path),
        "train_remote_jsonl": str(train_remote_path),
        "train_remote_sha256": sha256_file(train_remote_path),
        "rows": rows,
        "staged_methane_images": staged_images,
        "counts": {name: dict(counter) for name, counter in counts.items()},
        "replay": replay_details,
        "notes": [
            "Methane rows use the supplied flat schema in canonical key order.",
            "Replay rows are pre-existing strict-JSON structured-output VLM tasks and keep their own prompts/schemas.",
            "Replay image paths are expected to already exist under /outputs in the Modal volume.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
