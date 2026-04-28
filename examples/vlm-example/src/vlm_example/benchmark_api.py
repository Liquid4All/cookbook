"""
Benchmarks a frontier API model (e.g. Claude) on the defect-detection dataset.

Runs locally without GPU. Logs results to WandB.

Usage:
    make benchmark-api config=benchmark_claude_sonnet_goodsad.yaml
"""

import argparse
import os
import tempfile
import time

import anthropic
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

from .config import ApiBenchmarkConfig
from .inference import get_claude_output, parse_yes_no
from .loaders import load_dataset
from .report import BenchmarkReport

load_dotenv()


def benchmark(config: ApiBenchmarkConfig) -> BenchmarkReport:
    start_time = time.time()
    print(f"Starting benchmark of {config.model} on {config.dataset}/{config.split}")

    wandb.init(
        project=config.wandb_project_name,
        config=config.model_dump(),
        tags=[
            config.model,
            config.dataset.split("/")[-1],
            "api",
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
        cache_dir="./data",
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    report = BenchmarkReport()

    for sample in tqdm(dataset, desc="Benchmarking"):
        image_data = sample[config.image_column]
        raw_answer = sample[config.answer_column]
        if isinstance(raw_answer, int):
            ground_truth = dataset.features[config.answer_column].int2str(raw_answer)
        else:
            ground_truth = raw_answer
        user_prompt: str = config.prompt_override or sample[config.prompt_column]

        raw = get_claude_output(client, config.model, user_prompt, image_data)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file-name", required=True)
    args = parser.parse_args()

    config = ApiBenchmarkConfig.from_yaml(args.config_file_name)
    report = benchmark(config)

    output_path = report.to_csv()
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
