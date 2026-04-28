import os
from pathlib import Path

import datasets
from datasets import Dataset, concatenate_datasets
from huggingface_hub import login
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_dataset(
    dataset_name: str,
    splits: list[str],
    sources: list[str] | None = None,
    n_samples: int | None = None,
    seed: int | None = 42,
    cache_dir: str = "/datasets",
) -> datasets.Dataset:
    """Loads a dataset from HuggingFace Hub with Modal volume caching."""
    cache_path = Path(cache_dir) / dataset_name.replace("/", "_")
    cache_path.mkdir(parents=True, exist_ok=True)

    dataset_list: list[Dataset] = []

    for split in splits:
        split_cache_path = cache_path / split

        if split_cache_path.exists():
            print(f"Loading cached dataset {dataset_name}, split={split} from {split_cache_path}...")
            try:
                dataset = Dataset.load_from_disk(str(split_cache_path))
                print(f"Loaded {len(dataset)} samples from cache")
            except Exception as e:
                print(f"Failed to load from cache: {e}")
                print(f"Downloading dataset {dataset_name}, split={split} from HuggingFace...")
                dataset = datasets.load_dataset(dataset_name, split=split, num_proc=1)
                print(f"Caching dataset to {split_cache_path}...")
                dataset.save_to_disk(str(split_cache_path))
        else:
            print(f"Downloading dataset {dataset_name}, split={split} from HuggingFace...")
            dataset = datasets.load_dataset(dataset_name, split=split, num_proc=1)
            print(f"Caching dataset to {split_cache_path}...")
            try:
                dataset.save_to_disk(str(split_cache_path))
                print("Dataset cached successfully")
            except Exception as e:
                print(f"Failed to cache dataset: {e}")

        dataset_list.append(dataset)

    if len(dataset_list) >= 1:
        dataset = concatenate_datasets(dataset_list)
    else:
        raise ValueError("No splits provided to load the dataset.")

    print(f"Shuffling dataset with seed {seed}...")
    dataset = dataset.shuffle(seed=seed)

    if sources is not None:
        print(f"Filtering dataset to sources: {sources}")
        indices = [i for i, s in enumerate(dataset["source"]) if s in sources]
        dataset = dataset.select(indices)
        print(f"Filtered to {len(dataset)} samples")

    if n_samples is not None:
        n_samples = min(n_samples, dataset.num_rows)
        dataset = dataset.select(range(n_samples))

    print(f"Dataset {dataset_name} loaded: {dataset.num_rows} rows")
    return dataset


def fix_model_type_in_config_json(model_path: str):
    """Fix config.json by replacing 'lfm2-vl' model_type with 'lfm2_vl'."""
    import json

    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    if config.get("model_type") == "lfm2-vl":
        print(f"Fixing config.json for model at {model_path}...")
        config["model_type"] = "lfm2_vl"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("config.json fixed successfully")


def load_model_and_processor(
    model_id: str,
    cache_dir: str = "/models",
    max_image_tokens: int = 256,
    min_image_tokens: int = 64,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Loads a model and processor from HuggingFace Hub with Modal volume caching."""
    model_cache_path = Path(cache_dir) / model_id.replace("/", "_")
    model_cache_path.mkdir(parents=True, exist_ok=True)

    processor_cache_path = model_cache_path / "processor"
    model_weights_cache_path = model_cache_path / "model"

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Logging in to HuggingFace Hub...")
        login(token=hf_token)
    else:
        print("No HF_TOKEN found, proceeding without authentication")

    if processor_cache_path.exists() and model_weights_cache_path.exists():
        print(f"Loading cached model and processor from {model_cache_path}...")
        try:
            try:
                fix_model_type_in_config_json(str(model_weights_cache_path))
            except Exception as e:
                print(f"Warning: could not fix config.json for cached model: {e}")

            processor = AutoProcessor.from_pretrained(
                str(processor_cache_path),
                max_image_tokens=max_image_tokens,
                min_image_tokens=min_image_tokens,
                do_image_splitting=True,
                local_files_only=True,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                str(model_weights_cache_path),
                torch_dtype="bfloat16",
                device_map="auto",
local_files_only=True,
            )
            print("Loaded model and processor from cache")
        except Exception as e:
            print(f"Failed to load from cache: {e}")
            print(f"Downloading model {model_id} from HuggingFace...")
            processor, model = _download_and_cache_model(
                model_id, hf_token, processor_cache_path, model_weights_cache_path, max_image_tokens, min_image_tokens
            )
    else:
        print(f"Downloading model {model_id} from HuggingFace...")
        processor, model = _download_and_cache_model(
            model_id, hf_token, processor_cache_path, model_weights_cache_path, max_image_tokens, min_image_tokens
        )

    print(f"Model loaded | vocab size: {len(processor.tokenizer)} | params: {model.num_parameters():,}")
    return model, processor


def load_model_from_checkpoint(
    checkpoint_path: str,
    max_image_tokens: int = 256,
    min_image_tokens: int = 64,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Loads a fine-tuned model and processor from a local checkpoint directory."""
    checkpoint_dir = Path(checkpoint_path)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading model from checkpoint: {checkpoint_path}")

    try:
        fix_model_type_in_config_json(str(checkpoint_dir))
    except Exception as e:
        print(f"Warning: could not fix config.json: {e}")

    processor = AutoProcessor.from_pretrained(
        str(checkpoint_dir),
        max_image_tokens=max_image_tokens,
        min_image_tokens=min_image_tokens,
        do_image_splitting=True,
        local_files_only=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(checkpoint_dir),
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    )

    print(f"Checkpoint loaded | vocab size: {len(processor.tokenizer)} | params: {model.num_parameters():,}")
    return model, processor


def _download_and_cache_model(
    model_id: str,
    hf_token: str | None,
    processor_cache_path: Path,
    model_weights_cache_path: Path,
    max_image_tokens: int = 256,
    min_image_tokens: int = 64,
) -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    processor = AutoProcessor.from_pretrained(
        model_id,
        max_image_tokens=max_image_tokens,
        min_image_tokens=min_image_tokens,
        do_image_splitting=True,
        token=hf_token,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation="sdpa",
        token=hf_token,
    )

    try:
        print(f"Caching processor to {processor_cache_path}...")
        processor.save_pretrained(str(processor_cache_path))
        print("Processor cached")
    except Exception as e:
        print(f"Failed to cache processor: {e}")

    try:
        print(f"Caching model to {model_weights_cache_path}...")
        model.save_pretrained(str(model_weights_cache_path))
        try:
            fix_model_type_in_config_json(str(model_weights_cache_path))
        except Exception as e:
            print(f"Warning: could not fix config.json: {e}")
        print("Model cached")
    except Exception as e:
        print(f"Failed to cache model: {e}")

    return processor, model
