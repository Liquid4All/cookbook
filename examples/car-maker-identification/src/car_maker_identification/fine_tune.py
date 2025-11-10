import os

import wandb
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from .callbacks import ProcessorSaveCallback
from .config import FineTuningConfig
from .data_preparation import format_dataset_as_conversation, split_dataset
from .loaders import load_dataset, load_model_and_processor
from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)
from .paths import get_path_model_checkpoints_in_modal_volume

app = get_modal_app("car-maker-identification")
image = get_docker_image()
volume = get_volume("models")


def create_collate_fn(processor):
    """Create a collate function that prepares batch inputs for the processor."""

    def collate_fn(sample):
        batch = processor.apply_chat_template(
            sample, tokenize=True, return_dict=True, return_tensors="pt"
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        # print('Sample: ', sample)
        return batch

    return collate_fn


@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/model_checkpoints": volume,
    },
    secrets=get_secrets(),
    timeout=1 * 60 * 60,
    retries=get_retries(max_retries=1),
    max_inputs=1,  # Ensure we get a fresh container on retry
)
def fine_tune(
    config: FineTuningConfig,
):
    """Fine-tune an Image-text-to-Text model using LoRA and SFT."""
    print("Starting fine-tuning job")

    # Initialize wandb if enabled
    if config.use_wandb:
        print(
            f"Initializing WandB experiment {config.wandb_experiment_name or 'fine-tune-experiment'}"
        )
        wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_experiment_name,
            config=config.__dict__,
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"

    model, processor = load_model_and_processor(model_id=config.model_name)

    train_ds: Dataset = load_dataset(
        dataset_name=config.dataset_name,
        splits=config.dataset_splits,
        n_samples=config.dataset_samples,
        seed=config.seed,
    )

    print("Splitting the dataset into train and eval sets...")
    train_dataset, eval_dataset = split_dataset(
        train_ds, test_size=(1 - config.train_split_ratio), seed=config.seed
    )

    print("Formatting the datasets into a conversation format...")
    train_dataset = format_dataset_as_conversation(
        train_dataset,
        system_prompt=config.system_prompt,
        user_prompt=config.user_prompt,
        image_column=config.dataset_image_column,
        label_column=config.dataset_label_colum,
        label_mapping=config.label_mapping,
    )
    eval_dataset = format_dataset_as_conversation(
        eval_dataset,
        system_prompt=config.system_prompt,
        user_prompt=config.user_prompt,
        image_column=config.dataset_image_column,
        label_column=config.dataset_label_colum,
        label_mapping=config.label_mapping,
    )

    print("✅ SFT Dataset formatted:")
    print(f"📚 Train samples: {len(train_dataset)}")
    print(f"🧪 Eval samples: {len(eval_dataset)}")
    print("Train sample: ", train_dataset[0])
    print("Eval sample: ", eval_dataset[0])

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

    # Optional: model = get_peft_model(model, peft_config)

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
        max_length=512,  # TODO: use config.max_seq_length ?
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="wandb" if config.use_wandb else None,
        # Add these for step-based evaluation:
        eval_strategy="steps",  # Evaluate every N steps
        eval_steps=config.eval_steps,  # Evaluate every 1000 steps
        per_device_eval_batch_size=config.batch_size,  # Eval batch size
        save_strategy="steps",  # Save checkpoints every N steps
        save_steps=config.eval_steps,  # Save every 1000 steps
        load_best_model_at_end=True,  # Load best model after training
        metric_for_best_model="eval_loss",  # Metric to determine best model
    )

    # Create callbacks
    processor_callback = ProcessorSaveCallback(processor)
    # debug_callback = DebugPredictionCallback(eval_dataset, processor, num_samples=5)

    print("🏗️  Creating SFT trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
        callbacks=[
            processor_callback,
            # debug_callback
        ],
    )

    print("\n🚀 Starting SFT training...")
    from pathlib import Path

    if config.checkpoint_path is None:
        print("No checkpoint path provided, starting training from scratch.")
        trainer.train()
    else:
        print(f"Resuming training from checkpoint: {config.checkpoint_path}")
        trainer.train(
            resume_from_checkpoint=str(
                Path("/model_checkpoints") / config.checkpoint_path
            )
        )

    # print("Saving merged model")
    # if hasattr(model, 'peft_config'):
    #     print("🔄 Merging LoRA weights...")
    #     model = model.merge_and_unload()
    # model.save_pretrained(checkpoints_dir / "final")
    # processor.save_pretrained(checkpoints_dir / "final")
    # print("💾 Model saved to: {checkpoints_dir / 'final'}")

    # Finish wandb run if enabled
    if config.use_wandb:
        wandb.finish()


@app.local_entrypoint()
def main(config_file_name: str):
    """
    Fine-tunes a VL model on a given dataset using Modal serverless GPU
    acceleration.

    Args:
        config_file_name: The name of the configuration file to use
    """
    config = FineTuningConfig.from_yaml(config_file_name)

    try:
        fine_tune.remote(config=config)
        print("Fine-tuning job completed successfully!")
    except Exception as e:
        print(f"❌ Fine-tuning job failed: {e}")
        raise e
