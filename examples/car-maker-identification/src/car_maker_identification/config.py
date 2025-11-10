""" """

from typing import Any, Optional

from datetime import datetime
from pathlib import Path
from typing import Self

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings

from .paths import get_path_to_configs


class FineTuningConfig(BaseSettings):
    seed: int = 23
    use_wandb: bool = True

    # Model configuration
    model_name: str = "LiquidAI/LFM2-VL-450M"  # or LiquidAI/LFM2-VL-1.6B
    max_seq_length: int = 2048
    checkpoint_path: Optional[str] = None  # useful to resume training from a checkpoint

    # Dataset configuration
    dataset_name: str
    dataset_samples: int
    dataset_image_column: str
    dataset_label_colum: str
    dataset_splits: list[str] = ["train"]
    label_mapping: Optional[dict[Any, str]] = None
    train_split_ratio: float
    preprocessing_workers: int = 2

    system_prompt: str
    user_prompt: str

    # LoRA-specific hyperparameters
    use_peft: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"  # unsloth: optimized lora kernel
    use_rslora: bool = False
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # General training hyperparameters
    learning_rate: float
    num_train_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    optim: str = "adamw_8bit"
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    eval_steps: int

    # max_steps: int = 10000  # increase!
    # save_steps: int = 1000  # increase!
    # eval_steps: int = 1000  # increase!
    # eval_sample_callback_enabled: bool = False

    # Weights and Biases configuration
    wandb_project_name: str = "car-maker-identification-fine-tuning"
    wandb_experiment_name: str | None = None
    skip_eval: bool = False
    output_dir: str = "outputs"

    modal_app_name: str

    @classmethod
    def from_yaml(cls, file_name: str) -> Self:
        """
        Loads configuration from a YAML file located in the configs directory.
        """
        file_path = str(Path(get_path_to_configs()) / file_name)
        print(f"Loading config from {file_path}")
        with open(file_path) as f:
            data = yaml.safe_load(f)

        # print('Loaded config:', data)

        return cls(**data)

    @model_validator(mode="after")
    def set_experiment_name(self):
        if self.wandb_experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.wandb_experiment_name = (
                f"{model_short}-{self.dataset_name}-{timestamp}"
            )

        return self


class EvaluationConfig(BaseSettings):
    seed: int = 23

    # Model parameters
    model: str
    structured_generation: bool

    # Dataset parameters
    dataset: str
    split: str
    n_samples: int
    system_prompt: str
    user_prompt: str
    image_column: str
    label_column: str
    label_mapping: Optional[dict] = None

    # Batch processing parameters
    batch_size: int = 1

    # Weights and Biases configuration
    wandb_project_name: str = "car-maker-identification-evals"

    @classmethod
    def from_yaml(cls, file_name: str) -> "EvaluationConfig":
        """
        Loads configuration from a YAML file located in the configs directory.
        """
        file_path = str(Path(get_path_to_configs()) / file_name)
        print(f"Loading config from {file_path}")
        with open(file_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)
