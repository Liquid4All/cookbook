import os
import random
import numpy as np
import datasets
from datasets import concatenate_datasets, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from huggingface_hub import login
import torch
from pathlib import Path


def load_dataset(
    dataset_name: str,
    splits: list[str],
    n_samples: int | None = None,
    seed: int | None = 42,
    cache_dir: str = "/datasets",
) -> datasets.Dataset:
    """
    Loads a dataset from the Hugging Face dataset hub with Modal volume caching.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        splits: List of dataset splits to load
        n_samples: Number of samples to select (optional)
        seed: Random seed for shuffling
        cache_dir: Directory to cache datasets (Modal volume mount point)
    """
    # Create cache directory structure
    cache_path = Path(cache_dir) / dataset_name.replace("/", "_")
    cache_path.mkdir(parents=True, exist_ok=True)
    
    dataset_list: list[Dataset] = []
    
    for split in splits:
        split_cache_path = cache_path / f"{split}"
        
        if split_cache_path.exists():
            print(f"📚 Loading cached dataset {dataset_name}, split={split} from {split_cache_path}...")
            try:
                # Load from cache
                dataset = Dataset.load_from_disk(str(split_cache_path))
                print(f"✅ Successfully loaded {len(dataset)} samples from cache")
            except Exception as e:
                print(f"❌ Failed to load from cache: {e}")
                print(f"📚 Downloading dataset {dataset_name}, split={split} from HuggingFace...")
                dataset = datasets.load_dataset(dataset_name, split=split, num_proc=1)
                
                # Save to cache
                print(f"💾 Caching dataset to {split_cache_path}...")
                dataset.save_to_disk(str(split_cache_path))
        else:
            print(f"📚 Downloading dataset {dataset_name}, split={split} from HuggingFace...")
            dataset = datasets.load_dataset(dataset_name, split=split, num_proc=1)
            
            # Save to cache
            print(f"💾 Caching dataset to {split_cache_path}...")
            try:
                dataset.save_to_disk(str(split_cache_path))
                print(f"✅ Dataset cached successfully")
            except Exception as e:
                print(f"⚠️ Failed to cache dataset: {e}")
        
        dataset_list.append(dataset)

    # Concatenate datasets
    if len(dataset_list) >= 1:
        dataset = concatenate_datasets(dataset_list)
    else:
        raise Exception("No splits provided to load the dataset.")

    # Shuffle the dataset
    print(f"Shuffling dataset with seed {seed}...")
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
    from pathlib import Path

    config_path = Path(model_id) / "config.json"

    # Check if model_id is a local path
    with open(config_path, "r") as f:
        config = json.load(f)

    # Fix the model_type if needed
    if config.get("model_type") == "lfm2-vl":
        print(f"Fixing config.json for model {model_id}...")
        config["model_type"] = "lfm2_vl"

        # Write back the fixed config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print("config.json fixed successfully!")


def load_model_and_processor(
    model_id: str,
    cache_dir: str = "/models",
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Loads a model and processor from the Hugging Face model hub with Modal volume caching.
    
    Args:
        model_id: HuggingFace model identifier 
        cache_dir: Directory to cache models (Modal volume mount point)
    """
    # Create cache directory structure
    model_cache_path = Path(cache_dir) / model_id.replace("/", "_")
    model_cache_path.mkdir(parents=True, exist_ok=True)
    
    processor_cache_path = model_cache_path / "processor"
    model_weights_cache_path = model_cache_path / "model"
    
    # Login using HF_TOKEN from environment variables
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("🔐 Logging in to Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("⚠️ No HF_TOKEN found in environment variables")

    # Check if model is cached
    if processor_cache_path.exists() and model_weights_cache_path.exists():
        print(f"📚 Loading cached model and processor from {model_cache_path}...")
        try:
            # Fix config if needed (for cached models)
            try:
                fix_model_type_in_config_json(str(model_weights_cache_path))
            except Exception as e:
                print(f"Warning: could not fix config.json for cached model: {e}")
            
            # Load processor from cache
            processor = AutoProcessor.from_pretrained(
                str(processor_cache_path),
                max_image_tokens=256,
                local_files_only=True,
            )
            
            # Load model from cache
            model = AutoModelForImageTextToText.from_pretrained(
                str(model_weights_cache_path),
                torch_dtype="bfloat16",
                device_map="auto",
                local_files_only=True,
            )
            
            print("✅ Successfully loaded model and processor from cache")
            
        except Exception as e:
            print(f"❌ Failed to load from cache: {e}")
            print(f"📚 Downloading model {model_id} from HuggingFace...")
            
            # Download and cache
            processor, model = _download_and_cache_model(model_id, hf_token, processor_cache_path, model_weights_cache_path)
    else:
        print(f"📚 Downloading model {model_id} from HuggingFace...")
        
        # Download and cache
        processor, model = _download_and_cache_model(model_id, hf_token, processor_cache_path, model_weights_cache_path)

    print("\n✅ Model loaded successfully!")
    print(f"📖 Vocab size: {len(processor.tokenizer)}")
    print(f"🔢 Parameters: {model.num_parameters():,}")
    print(f"💾 Model size: ~{model.num_parameters() * 2 / 1e9:.1f} GB (bfloat16)")

    return model, processor


def _download_and_cache_model(
    model_id: str, 
    hf_token: str, 
    processor_cache_path: Path, 
    model_weights_cache_path: Path
) -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    """Helper function to download and cache model and processor."""
    
    # Download processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        max_image_tokens=256,
        token=hf_token,
    )
    
    # Download model
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="bfloat16",
        device_map="auto", 
        token=hf_token,
    )
    
    # Cache processor
    try:
        print(f"💾 Caching processor to {processor_cache_path}...")
        processor.save_pretrained(str(processor_cache_path))
        print("✅ Processor cached successfully")
    except Exception as e:
        print(f"⚠️ Failed to cache processor: {e}")
    
    # Cache model  
    try:
        print(f"💾 Caching model to {model_weights_cache_path}...")
        model.save_pretrained(str(model_weights_cache_path))
        
        # Apply config fix after caching
        try:
            fix_model_type_in_config_json(str(model_weights_cache_path))
        except Exception as e:
            print(f"Warning: could not fix config.json for model {model_id}: {e}")
            
        print("✅ Model cached successfully")
    except Exception as e:
        print(f"⚠️ Failed to cache model: {e}")
    
    return processor, model
