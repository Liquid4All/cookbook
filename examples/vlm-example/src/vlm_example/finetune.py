"""
Fine-tunes LFM2.5-VL on the defect-detection dataset using LoRA and SFT on Modal.

Usage:
    make finetune config=finetune_lfm25_vl_450M.yaml
"""

import os
from pathlib import Path

import wandb
from trl import SFTConfig, SFTTrainer

from .callbacks import ProcessorSaveCallback
from .config import FineTuningConfig
from .data_preparation import format_dataset_as_conversation
from .loaders import load_dataset, load_model_and_processor
from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)
from .paths import get_path_model_checkpoints_in_modal_volume

app = get_modal_app("defect-detection-finetune")
image = get_docker_image()
datasets_volume = get_volume("datasets")
models_volume = get_volume("models")
checkpoints_volume = get_volume("model-checkpoints")


def create_collate_fn(processor):
    def collate_fn(sample):
        batch = processor.apply_chat_template(
            sample, tokenize=True, return_dict=True, return_tensors="pt"
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/datasets": datasets_volume,
        "/models": models_volume,
        "/model_checkpoints": checkpoints_volume,
    },
    secrets=get_secrets(),
    timeout=1 * 60 * 60,
    retries=get_retries(max_retries=1),
    max_inputs=1,
)
def finetune(config: FineTuningConfig):
    """Fine-tunes a VL model on the defect-detection dataset using LoRA and SFT."""
    print("Starting fine-tuning job")

    if config.use_wandb:
        print(f"Initializing WandB experiment: {config.wandb_experiment_name}")
        wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_experiment_name,
            config=config.__dict__,
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"

    model, processor = load_model_and_processor(
        model_id=config.model_name, cache_dir="/models"
    )

    sources = [config.dataset_source] if isinstance(config.dataset_source, str) else config.dataset_source

    train_dataset_raw = load_dataset(
        dataset_name=config.dataset_name,
        splits=["train"],
        sources=sources,
        n_samples=config.dataset_samples,
        seed=config.seed,
        cache_dir="/datasets",
    )
    eval_dataset_raw = load_dataset(
        dataset_name=config.dataset_name,
        splits=["test"],
        sources=sources,
        n_samples=None,
        seed=config.seed,
        cache_dir="/datasets",
    )

    print("Formatting datasets as conversations...")
    train_dataset = format_dataset_as_conversation(
        train_dataset_raw,
        image_column=config.dataset_image_column,
        prompt=config.prompt_override,
        answer_column=config.dataset_answer_column,
    )
    eval_dataset = format_dataset_as_conversation(
        eval_dataset_raw,
        image_column=config.dataset_image_column,
        prompt=config.prompt_override,
        answer_column=config.dataset_answer_column,
    )

    print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    if config.use_peft:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            r=config.lora_r,
            bias="none",
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    collate_fn = create_collate_fn(processor)

    checkpoints_dir = get_path_model_checkpoints_in_modal_volume(
        config.wandb_experiment_name
    )
    print(f"Checkpoints will be saved to: {checkpoints_dir}")

    sft_config = SFTConfig(
        output_dir=checkpoints_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        optim=config.optim,
        gradient_checkpointing=True,
        max_length=config.max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="wandb" if config.use_wandb else None,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        per_device_eval_batch_size=config.batch_size,
        save_strategy="steps",
        save_steps=config.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
        callbacks=[ProcessorSaveCallback(processor)],
    )

    print("Starting SFT training...")
    if config.checkpoint_path is None:
        trainer.train()
    else:
        resume_path = str(Path("/model_checkpoints") / config.checkpoint_path)
        print(f"Resuming from checkpoint: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)

    if config.use_wandb:
        wandb.finish()


@app.local_entrypoint()
def main(config_file_name: str):
    """
    Fine-tunes a VL model on the defect-detection dataset using Modal serverless GPU.

    Args:
        config_file_name: Name of a YAML file in the configs/ directory.
    """
    config = FineTuningConfig.from_yaml(config_file_name)

    try:
        finetune.remote(config=config)
        print("Fine-tuning completed successfully")
    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        raise
