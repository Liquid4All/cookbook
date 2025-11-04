# Fine-tune LFM2-VL to identify car makers from images

[![GitHub](https://img.shields.io/badge/GitHub-Repository-purple?style=for-the-badge&logo=github)](https://github.com/Liquid4All/cookbook/tree/main/examples/car-maker-identification)

A comprehensive guide to fine-tuning Liquid Foundational Models for computer vision tasks, specifically car maker identification from images.

## Table of Contents

- [What's inside?](#whats-inside)
- [Quickstart](#quickstart)
- [Environment setup](#environment-setup)
  - [Install UV](#install-uv)
  - [Modal setup](#modal-setup)
  - [Weights & Biases setup](#weights--biases-setup)
  - [Install make](#install-make)
- [Understanding the main steps to fine tune good Vision Language Models](#understanding-the-main-steps-to-fine-tune-good-vision-language-models)
- [Step 1. Dataset preparation](#step-1-dataset-preparation)
- [Step 2. Evaluating LFM2-VL models](#step-2-evaluating-lfm2-vl-models)
- [Need help?](#need-help)

## What's inside?

In this example, you will learn how to:

- Build a model-agnostic **evaluation pipeline** for vision classification tasks
- **Use structured output generation** with Outlines to ensure consistent and reliable model responses
- **Prepare training data** for fine-tuning vision-language models on specific classification tasks
- **Improve model accuracy** with parameter efficient techniques like LoRA.
- **Run everything locally** on your machine without requiring cloud services, API keys, or sharing private data


## Quickstart

### 1. Clone the repository:
```sh
git clone https://github.com/Liquid4All/cookbook.git
cd cookbook/examples/car-maker-identification
```

### 2. Evaluate the models:
```sh
make evaluate config=eval_lfm_450M.yaml
make evaluate config=eval_lfm_1.6B.yaml
make evaluate config=eval_lfm_3B.yaml
```

### 3. Fine-tune the models:
```sh
make fine-tune config=finetune_lfm_450M.yaml
make fine-tune config=finetune_lfm_1.6B.yaml
make fine-tune config=finetune_lfm_3B.yaml
```


## Environment setup

You will need

- [uv](https://docs.astral.sh/uv/) to manage Python dependencies and run the application efficiently without creating virtual environments manually.

- [Modal](https://modal.com/) for GPU cloud compute. Fine-tuning a Vision Language Model without a GPU is too slow. One easy way to get access to a GPU and pay per usage is with Modal. It requires 0 infra setup and helps us get up and running with our fine-tuning experiment really fast.

- [Weights & Biases](https://wandb.ai/) (optional, but highly recommended) for experiment tracking and monitoring during fine-tuning

- [make](https://www.gnu.org/software/make/) (optional) to simplify the execution of the application and fine-tuning experiments.

Let's go one by one.

### Install UV

<details>
<summary>Click to see installation instructions for your platform</summary>

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

</details>

### Modal setup

<details>
<summary>Click to see installation instructions</summary>

1. Create an account at [modal.com](https://modal.com/)
2. Install the Modal Python package inside your virtual environment:
   ```bash
   uv add modal
   ```
3. Authenticate with Modal:
   ```bash
   uv run modal setup
   ```
   If the first command fails, try:
   ```bash
   uv run python -m modal setup
   ```
</details>

### Weights & Biases setup

<details>
<summary>Click to see installation instructions</summary>

1. Create an account at [wandb.ai](https://wandb.ai/)
2. Install the Weights & Biases Python package inside your virtual environment:
   ```bash
   uv add wandb
   ```
3. Authenticate with Weights & Biases:
   ```bash
   uv run wandb login
   ```
   This will open a browser window where you can copy your API key and paste it in the terminal.

</details>

### Install make

<details>
<summary>Click to see installation instructions</summary>

1. Install make:
   ```bash
   sudo apt-get install make
   ```
</details>

## Steps to fine-tune LFM2-VL models for car maker identification

Here's the systematic approach we follow to fine-tune LFM2-VL models for car maker identification:

1. **Prepare the dataset**. Collect an accurate and diverse dataset of (image, car_maker) pairs, that represents the entire distribution of inputs the model will be exposed to once deployed.

2. **Establish baseline performance**. Evaluate pre-trained models of different sizes (450M, 1.6B, 3B) to understand current capabilities. If the results are good enough for your use case, and the model fits your deployment environment constraints, there is no need to fine tune further. Otherwise, you need to fine-tune.

3. **Fine-tune with LoRA**. Apply parameter-efficient fine-tuning using Low-Rank Adaptation to improve model accuracy while keeping computational costs manageable.

4. **Evaluate improvements**. Compare fine-tuned model performance against baselines to measure the effectiveness of our customization.


Let's go one by one.


## Step 1. Dataset preparation

Dataset creation is one of the most critical parts of the whole project.

> [!NOTE]
> A fine tuned Language Model is as **good** as the dataset used to fine tune it

> [!TIP]
> **What does *good* mean in this case?**
>
> A good dataset for image classification needs to be:
>
> - **Accurate**: Labels must correctly match the images. For car maker identification, this means each car image is labeled with the correct manufacturer (e.g., a BMW X5 should be labeled as "BMW", not "Mercedes-Benz"). Mislabeled data will teach the model incorrect associations.
>
> - **Diverse**: The dataset should represent the full range of conditions the model will encounter in production. This includes:
>   - Different car models from each manufacturer
>   - Various angles, lighting conditions, and backgrounds
>   - Different image qualities and resolutions
>   - Cars from different years and in different conditions (new, old, dirty, clean)


In this guide we will be using the [Stanford Cars dataset](https://huggingface.co/datasets/Paulescu/stanford_cars) hosted on Hugging Face. 

The dataset contains:

- **Classes**: 49 unique car manufacturers.
- **Splits**: Train (6,860 images) and test (6,750 images) splits.

The dataset includes additional splits with various image corruptions (gaussian noise, motion blur, etc.) for robustness testing, making it ideal for evaluating model performance under different conditions. In this tutorial we will only use the train and test splits.


## Step 2. Evaluating LFM2-VL models

Before embarking into any fine-tuning experiment, we need to establish a baseline performance for existing models. In this case, we will evaluate the peformance of

- LFM2-VL-450M
- LFM2-VL-1.6B
- LFM2-VL-3B

To build a model-agnostic evaluation script, it is best practice to decouple the evaluation logic from the model and dataset specific code. Feel free to encapsulate the experiment parameters into any configuration file you like. One common approach is to use a YAML file, which is what we will do in this case.

In the `configs` directory you will find the 3 YAML files we use to evaluate these 3 models
on this task

```sh
configs/
   ├── eval_lfm_450M.yaml
   ├── eval_lfm_1.6B.yaml
   └── eval_lfm_3B.yaml
```

These YAML configs are loaded into our Python script using pydantic-settings, so we ensure type safety and validation of the configuration parameters.

```python
# src/car_maker_identification/config.py
from pydantic_settings import BaseSettings

class EvaluationConfig(BaseSettings):
   # to ensure reproducible runs
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
```


We use the same evaluation dataset of 50 samples, and the same system prompt and user prompt for the 3 models.

<details>
<summary>Click to see system and user prompts</summary>

```yaml
system_prompt: |
  You excel at identifying car makers from pictures.

user_prompt: |
  What car maker do you see in this picture?
  Pick one from the following list:

  - AM
  - Acura
  - Aston
  - Audi
  - BMW
  - Bentley
  - Bugatti
  - Buick
  - Cadillac
  - Chevrolet
  - Chrysler
  - Daewoo
  - Dodge
  - Eagle
  - FIAT
  - Ferrari
  - Fisker
  - Ford
  - GMC
  - Geo
  - HUMMER
  - Honda
  - Hyundai
  - Infiniti
  - Isuzu
  - Jaguar
  - Jeep
  - Lamborghini
  - Land
  - Lincoln
  - MINI
  - Maybach
  - Mazda
  - McLaren
  - Mercedes-Benz
  - Mitsubishi
  - Nissan
  - Plymouth
  - Porsche
  - Ram
  - Rolls-Royce
  - Scion
  - Spyker
  - Suzuki
  - Tesla
  - Toyota
  - Volkswagen
  - Volvo
  - smart
```
</details>

The only difference is the model name.

You can run evaluation for all the models with the following commands:
```sh
make evaluate config=eval_lfm_450M.yaml
make evaluate config=eval_lfm_1.6B.yaml
make evaluate config=eval_lfm_3B.yaml
```

The evaluation script starts as remove job inside a Docker container on Modal, using the Docker image



The results are:

| Model | Accuracy |
|-------|----------|
| LFM2-VL-450M | 60% |
| LFM2-VL-1.6B | 72% |
| LFM2-VL-3B | 80% |


Besides the overall accuracy,


![](./media/train_loss.png)
















## Step 1. Establish baseline accuracy with different models

Let's start by cloning the repository:

```sh
git clone https://github.com/Liquid4All/cookbook.git
cd cookbook/examples/car-maker-identification
```






## Need help?

Join the [Liquid AI Discord Community](https://discord.gg/DFU3WQeaYD) and ask.
[![Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.gg/DFU3WQeaYD)






## Step 1. Build a model-agnostic evaluation script


## Step 2. Generate baselines

```yaml
system_prompt: |
  You excel at identifying car makers from pictures.

user_prompt: |
  What car maker do you see in this picture?
  Pick one from the following list:

  - AM
  - Acura
  - Aston
  - Audi
  - BMW
  - Bentley
  - Bugatti
  - Buick
  - Cadillac
  - Chevrolet
  - Chrysler
  - Daewoo
  - Dodge
  - Eagle
  - FIAT
  - Ferrari
  - Fisker
  - Ford
  - GMC
  - Geo
  - HUMMER
  - Honda
  - Hyundai
  - Infiniti
  - Isuzu
  - Jaguar
  - Jeep
  - Lamborghini
  - Land
  - Lincoln
  - MINI
  - Maybach
  - Mazda
  - McLaren
  - Mercedes-Benz
  - Mitsubishi
  - Nissan
  - Plymouth
  - Porsche
  - Ram
  - Rolls-Royce
  - Scion
  - Spyker
  - Suzuki
  - Tesla
  - Toyota
  - Volkswagen
  - Volvo
  - smart

```

```sh
make evaluate config=eval_lfm_450M.yaml
make evaluate config=eval_lfm_1.6B.yaml
make evaluate config=eval_lfm_3B.yaml
```

The results are:

| Model | Accuracy |
|-------|----------|
| LFM2-VL-450M | 60% |
| LFM2-VL-1.6B | 72% |
| LFM2-VL-3B | 80% |

### Step 3.

```sh
make fine-tune config=finetune_lfm_450M.yaml
make fine-tune config=finetune_lfm_1.6B.yaml
make fine-tune config=finetune_lfm_3B.yaml
```


