
import os
import datasets
from datasets import concatenate_datasets, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from huggingface_hub import login


def load_dataset(
    dataset_name: str,
    splits: list[str],
    n_samples: int | None = None,
    seed: int | None = 42,
) -> datasets.Dataset:
    """
    Loads a dataset from the Hugging Face dataset hub.
    """
    dataset_list: list[Dataset] = []
    for split in splits:
        print(f"📚 Loading dataset {dataset_name}, split={split}...")
        dataset = datasets.load_dataset(dataset_name, split=split, num_proc=1)
        dataset_list.append(dataset)

    # Concatenate datasets
    if len(dataset_list) >= 1:
        dataset = concatenate_datasets(dataset_list)
    else:
        raise Exception("No splits provided to load the dataset.")

    # print(f"📚 Loading dataset {dataset_name}, split={split}...")
    # dataset = datasets.load_dataset(dataset_name, split=split, num_proc=1)

    # Shuffle the dataset
    dataset = dataset.shuffle(seed=seed)

    # Select a subset of the dataset
    if n_samples is not None:
        n_samples = min(n_samples, dataset.num_rows)
        dataset = dataset.select(range(n_samples))

    print(f"Dataset {dataset_name} loaded successfully: {dataset.num_rows} rows")

    return dataset


def fix_model_type_in_config_json(model_id: str):
    """Fix config.json by replacing 'lfm2-vl' model_type with 'lfm2_vl'."""
    import json
    import os
    from pathlib import Path

    config_path = Path(model_id) / "config.json"
    
    # Check if model_id is a local path
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Fix the model_type if needed
    if config.get("model_type") == "lfm2-vl":
        print(f"Fixing config.json for model {model_id}...")
        config["model_type"] = "lfm2_vl"
        
        # Write back the fixed config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print('config.json fixed successfully!')

def load_model_and_processor(
    model_id: str,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Loads a model and processor from the Hugging Face model hub.
    """
    # Login using HF_TOKEN from environment variables
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("🔐 Logging in to Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("⚠️ No HF_TOKEN found in environment variables")
    
    # TODO: hack hack hack
    try:
        fix_model_type_in_config_json(model_id)
    except Exception as e:
        print(f"Warning: could not fix config.json for model {model_id}: {e}")

    print(f"📚 Loading processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        # trust_remote_code=True,
        max_image_tokens=256,
        token=hf_token,
    )

    print(f"🧠 Loading model from {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="bfloat16",
        # trust_remote_code=True,
        device_map="auto",
        token=hf_token,
    )

    print("\n✅ Local model loaded successfully!")
    print(f"📖 Vocab size: {len(processor.tokenizer)}")
    # print(
    #     f"🖼️ Image processed in up to {processor.max_tiles} patches of size {processor.tile_size}"
    # )
    print(f"🔢 Parameters: {model.num_parameters():,}")
    print(f"💾 Model size: ~{model.num_parameters() * 2 / 1e9:.1f} GB (bfloat16)")

    return model, processor
