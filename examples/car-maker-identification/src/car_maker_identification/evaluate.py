"""
Evaluates a VL model on a given dataset

Steps:
1. Download the dataset
2. Load the model
3. Loop through the dataset and compute model outputs
4. Compute accuracy as a binary score: 1 if the model output matches the ground truth, 0 otherwise
"""

import time
from tqdm import tqdm
import wandb
import tempfile
import matplotlib.pyplot as plt

from .config import EvaluationConfig
from .inference import get_model_output, get_structured_model_output
from .loaders import load_dataset, load_model_and_processor
from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)
from .report import EvalReport  # , save_predictions_to_disk
from .output_types import CarIdentificationOutputType
from .batching import create_batches

app = get_modal_app("car-maker-identification")
image = get_docker_image()
datasets_volume = get_volume("datasets")
models_volume = get_volume("models")


@app.function(
    image=image,
    gpu="L40S", # Need more power? Use `gpu="H100"`
    volumes={
        "/datasets": datasets_volume,
        "/models": models_volume,
    },
    secrets=get_secrets(),
    timeout=1 * 60 * 60,
    retries=get_retries(max_retries=1),
    max_inputs=1, # Ensure we get a fresh container on retry
)
def evaluate(
    config: EvaluationConfig,
) -> EvalReport:
    """
    Runs a model evaluation on a given dataset using Modal serverless GPU

    Args:
        config: The configuration for the evaluation

    Returns:
        EvalReport: The evaluation report
    """
    start_time = time.time()
    print(f"Starting evaluation of {config.model} on {config.dataset}")

    # Initialize wandb run
    wandb.init(
        project=config.wandb_project_name,
        config=config.model_dump(),
        tags=[config.model.split("/")[-1], config.dataset.split("/")[-1]],
    )

    dataset = load_dataset(
        dataset_name=config.dataset,
        splits=[config.split],
        n_samples=config.n_samples,
        seed=config.seed,
        cache_dir="/datasets",
    )

    model, processor = load_model_and_processor(model_id=config.model, cache_dir="/models")

    # Prepare evaluation report
    eval_report = EvalReport()

    # Create batches
    batches = create_batches(dataset, config)
    print(f"Processing {len(dataset)} samples in {len(batches)} batches of size {config.batch_size}")
    
    # Process batches
    accurate_predictions: int = 0
    for batch_images, batch_labels in tqdm(batches, desc="Processing batches"):
        
        if config.structured_generation:
            # Use structured generation with batching
            if len(batch_images) == 1:
                # Single image case
                model_outputs = get_structured_model_output(
                    model, processor, config.system_prompt, config.user_prompt, batch_images[0]
                )
                model_outputs = [model_outputs] if model_outputs is not None else [None]
            else:
                # Batch case
                model_outputs = get_structured_model_output(
                    model, processor, config.system_prompt, config.user_prompt, batch_images
                )
            
            # Process results
            if model_outputs is not None:
                for image, label, model_output in zip(batch_images, batch_labels, model_outputs):
                    if model_output is not None:
                        pred_class = model_output.pred_class
                        accurate_predictions += 1 if pred_class == label else 0
                        eval_report.add_record(image, label, pred_class)
            else:
                print(f"Skipping batch of {len(batch_images)} samples due to failed batch processing")
        
        else:
            # Process individually for non-structured generation
            for image, label in zip(batch_images, batch_labels):
                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": config.system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": config.user_prompt},
                        ],
                    },
                ]
                
                pred_class: str = get_model_output(model, processor, conversation)
                accurate_predictions += 1 if pred_class == label else 0
                eval_report.add_record(image, label, pred_class)

    accuracy = eval_report.get_accuracy()
    print(f"Accuracy: {accuracy:.2f}")

    # Log accuracy to wandb
    wandb.log({"accuracy": accuracy})

    # Generate and log confusion matrix
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        # Create confusion matrix plot
        eval_report.print_confusion_matrix()
        plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log confusion matrix as image artifact
        wandb.log({"confusion_matrix": wandb.Image(tmp_file.name)})

    end_time = time.time()
    total_time = end_time - start_time
    wandb.log({"total_execution_time_seconds": total_time})
    
    print(f"⏱️ Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print("✅ Evaluation completed successfully")

    # Finish wandb run
    wandb.finish()

    return eval_report


@app.local_entrypoint()
def main(
    config_file_name: str,
):
    """
    Evaluates a VL model on a given dataset using Modal serverless GPU
    acceleration and stores an evaluation report in the evals/ directory.

    Args:
        config_file_name: The name of the configuration file to use
    """
    config = EvaluationConfig.from_yaml(config_file_name)

    eval_report = evaluate.remote(config)

    output_path = eval_report.to_csv()
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    config = EvaluationConfig()

    print(f"Loading dataset {config.dataset}")
    dataset = load_dataset(
        dataset_name=config.dataset, n_samples=config.n_samples, seed=config.seed
    )
    print(f"Dataset loaded successfully: {dataset.num_rows} rows")

    # Naive evaluation loop without batching
    accurate_predictions: int = 0
    for sample in dataset:
        print("Extracting sample image and normalized label")
        image = sample[config.image_column]

        # breakpoint()

        try:
            label = config.label_mapping[sample[config.label_column]]
        except KeyError:
            print("Error mapping label: ", sample[config.label_column])
            breakpoint()

        print("--------------------------------")
