"""
Benchmarks a VL model on the defect-detection dataset.

Usage:
    make benchmark config=benchmark_lfm25_vl_450M_raw.yaml
"""

import tempfile
import time
from pathlib import Path

import wandb
from tqdm import tqdm

from .config import BenchmarkConfig
from .inference import get_model_output, parse_yes_no
from .loaders import load_dataset, load_model_and_processor, load_model_from_checkpoint
from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)
from .report import BenchmarkReport

app = get_modal_app("defect-detection-benchmark")
image = get_docker_image()
datasets_volume = get_volume("datasets")
models_volume = get_volume("models")
checkpoints_volume = get_volume("model-checkpoints")


@app.function(
    image=image,
    gpu="L40S",
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
def benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    """
    Runs model inference on the defect-detection dataset and logs results to WandB.

    When config.checkpoint_path is set, loads the fine-tuned model from the Modal
    volume at /model_checkpoints/{config.checkpoint_path} instead of from HuggingFace Hub.
    """
    start_time = time.time()

    model_label = (
        f"checkpoint:{config.checkpoint_path}"
        if config.checkpoint_path
        else config.model
    )
    print(f"Starting benchmark of {model_label} on {config.dataset}/{config.split}")

    wandb.init(
        project=config.wandb_project_name,
        config=config.model_dump(),
        tags=[
            config.model.split("/")[-1],
            config.dataset.split("/")[-1],
            *(["checkpoint"] if config.checkpoint_path else []),
            *([config.source] if isinstance(config.source, str) else (config.source or [])),
        ],
    )

    sources = [config.source] if isinstance(config.source, str) else config.source

    dataset = load_dataset(
        dataset_name=config.dataset,
        splits=[config.split],
        sources=sources,
        n_samples=config.n_samples,
        seed=config.seed,
        cache_dir="/datasets",
    )

    if config.checkpoint_path is not None:
        full_checkpoint_path = str(Path("/model_checkpoints") / config.checkpoint_path)
        model, processor = load_model_from_checkpoint(
            full_checkpoint_path,
            max_image_tokens=config.max_image_tokens,
            min_image_tokens=config.min_image_tokens,
        )
    else:
        model, processor = load_model_and_processor(
            model_id=config.model,
            cache_dir="/models",
            max_image_tokens=config.max_image_tokens,
            min_image_tokens=config.min_image_tokens,
        )

    report = BenchmarkReport()

    for sample in tqdm(dataset, desc="Benchmarking"):
        image_data = sample[config.image_column]
        raw_answer = sample[config.answer_column]
        if isinstance(raw_answer, int):
            ground_truth = dataset.features[config.answer_column].int2str(raw_answer)
        else:
            ground_truth = raw_answer
        user_prompt: str = config.prompt_override

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_data},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        raw = get_model_output(model, processor, conversation)
        predicted = parse_yes_no(raw)

        report.add_record(image_data, ground_truth, predicted)

    accuracy = report.get_accuracy()
    majority_class_accuracy = report.get_majority_class_accuracy()
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.1%})")
    print(f"Majority class baseline: {majority_class_accuracy:.4f} ({majority_class_accuracy:.1%})")

    wandb.log({"accuracy": accuracy, "majority_class_accuracy": majority_class_accuracy})

    fig = report.get_confusion_matrix_figure()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        wandb.log({"confusion_matrix": wandb.Image(tmp.name)})

    elapsed = time.time() - start_time
    wandb.log({"total_execution_time_seconds": elapsed})
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    wandb.finish()
    return report


@app.local_entrypoint()
def main(config_file_name: str):
    """
    Benchmarks a VL model on the defect-detection dataset using Modal serverless GPU.

    Args:
        config_file_name: Name of a YAML file in the configs/ directory.
    """
    config = BenchmarkConfig.from_yaml(config_file_name)

    report = benchmark.remote(config=config)

    output_path = report.to_csv()
    print(f"Predictions saved to {output_path}")
