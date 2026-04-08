from datetime import datetime
from pathlib import Path
from typing import Optional, Self, Union

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings

from .paths import get_path_to_configs


class BenchmarkConfig(BaseSettings):
    seed: int = 42

    # Model: use `model` when loading from HF Hub, `checkpoint_path` when loading
    # a fine-tuned checkpoint from the Modal volume at /model_checkpoints/{checkpoint_path}.
    # `model` is always required so we know which processor to load alongside a checkpoint.
    model: str
    checkpoint_path: Optional[str] = None

    # Generation
    max_image_tokens: int = 256
    min_image_tokens: int = 64

    # Prompt (required: must be explicit in every benchmark config)
    prompt_override: str

    # Dataset
    dataset: str = "Paulescu/defect-detection"
    split: str = "test"
    source: Optional[Union[str, list[str]]] = None  # filter by source, e.g. "VisA" or ["VisA", "GoodsAD"]
    n_samples: Optional[int] = None
    image_column: str = "query_image"
    answer_column: str = "answer"

    # Batch processing
    batch_size: int = 1

    # Weights and Biases
    wandb_project_name: str = "defect-detection-benchmark"
    config_file: Optional[str] = None

    @classmethod
    def from_yaml(cls, file_name: str) -> "BenchmarkConfig":
        file_path = str(Path(get_path_to_configs()) / file_name)
        print(f"Loading config from {file_path}")
        with open(file_path) as f:
            data = yaml.safe_load(f)
        return cls(**data, config_file=file_name)


class ApiBenchmarkConfig(BaseSettings):
    seed: int = 42

    # Model: API model name, e.g. "claude-sonnet-4-6"
    model: str

    # Prompt (required: must be explicit in every benchmark config)
    prompt_override: str

    # Dataset
    dataset: str = "Paulescu/defect-detection"
    split: str = "test"
    source: Optional[Union[str, list[str]]] = None
    n_samples: Optional[int] = None
    image_column: str = "query_image"
    answer_column: str = "answer"

    # Weights and Biases
    wandb_project_name: str = "defect-detection-benchmark"
    config_file: Optional[str] = None

    @classmethod
    def from_yaml(cls, file_name: str) -> "ApiBenchmarkConfig":
        file_path = str(Path(get_path_to_configs()) / file_name)
        print(f"Loading config from {file_path}")
        with open(file_path) as f:
            data = yaml.safe_load(f)
        return cls(**data, config_file=file_name)


class FineTuningConfig(BaseSettings):
    seed: int = 42
    use_wandb: bool = True

    # Model
    model_name: str = "LiquidAI/LFM2.5-VL-450M-new-chat-template-3"
    max_seq_length: int = 2048
    checkpoint_path: Optional[str] = None  # path within /model_checkpoints to resume from

    # Dataset
    dataset_name: str = "Paulescu/defect-detection"
    dataset_samples: Optional[int] = None
    dataset_source: Optional[Union[str, list[str]]] = None  # filter by source, e.g. "GoodsAD" or ["GoodsAD", "VisA"]
    # Prompt (required: must be explicit in every fine-tuning config)
    prompt_override: str

    dataset_image_column: str = "query_image"
    dataset_answer_column: str = "answer"

    # LoRA
    use_peft: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Union[list[str], str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # Training
    learning_rate: float = 5e-4
    num_train_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    optim: str = "adamw_8bit"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 100

    # Weights and Biases
    wandb_project_name: str = "defect-detection-finetuning"
    wandb_experiment_name: Optional[str] = None

    modal_app_name: str = "defect-detection-finetune"

    @classmethod
    def from_yaml(cls, file_name: str) -> Self:
        file_path = str(Path(get_path_to_configs()) / file_name)
        print(f"Loading config from {file_path}")
        with open(file_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @model_validator(mode="after")
    def set_experiment_name(self):
        if self.wandb_experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            dataset_short = self.dataset_name.split("/")[-1]
            self.wandb_experiment_name = f"{model_short}-{dataset_short}-{timestamp}"
        return self
