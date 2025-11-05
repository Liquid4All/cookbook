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

<br>
Once you have installed these tools, you can git clone the repository and create the virtual environment with the following command:

```sh
git clone https://github.com/Liquid4All/cookbook.git
cd cookbook/examples/car-maker-identification
uv sync
```

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


## Step 2. Baseline performance of LFM2-VL models

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

You can run the evaluation for the 3 models with the following commands:
```sh
make evaluate config=eval_lfm_450M.yaml
make evaluate config=eval_lfm_1.6B.yaml
make evaluate config=eval_lfm_3B.yaml
```

The evaluation logic is encapsulated in the `evaluate.py` script, inside the `evaluate` function. This function does not run locally on your machine, but on a remote GPU thanks to the Modal `@app.function` decorator, and the `gpu="L40S"` argument.

```python
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
def evaluate(
    config: EvaluationConfig,
) -> EvalReport:
   """
   """
   # ...
```

The evaluationresults are the following:

| Model | Accuracy |
|-------|----------|
| LFM2-VL-450M | 60% |
| LFM2-VL-1.6B | 72% |
| LFM2-VL-3B | 81% |

Apart from the overall accuracy, it is highly recommended to inspect the misclassified images to understand the model's behavior.

You can do this by running the following command:
```sh
make report
```

This will open a Jupyter notebook that you can run to see:

- sample by sample comparison of the ground truth and predicted labels.
- confusion matrix of the predicted vs actual car makers.

For example, the confusion matrix for the LFM2-VL-3B model is this:

![](./media/confusion_matrix_lfm2_3b.png)

> [!NOTE]
> **Observations:**
> - The confusion matrix is mostly diagonal, meaning the model is good at identifying the correct car maker for most images.
> - Ford and Chevrolet are the most represented car makers in the eval set (7 samples each). However, while for Chevrolet the model is able to identify it correctly in most cases, for Ford it is not.

If you are happy with the results, you don't need to fine-tune the model further. Otherwise, you need to fine-tune the model to improve its performance.

## Step 3. Fine-tune LFM2-VL models

To fine-tune the model, we will use the LoRA technique. LoRA is a parameter-efficient fine-tuning technique that allows us to fine-tune the model by adding and tuning a small number of parameters.

![](./media/lora.webp)

We will fine-tune the 3 models with the following commands:
```sh
make fine-tune config=finetune_lfm_450M.yaml
make fine-tune config=finetune_lfm_1.6B.yaml
make fine-tune config=finetune_lfm_3B.yaml
```

This will fine-tune the models for 3 epochs and save the checkpoints in a remote Modal volume. You can monitor the training progress with Weights & Biases.

The train loss curves for the 3 models stabilize around very different loss values, where

- the LFM2-VL-3B model has the lowest loss, and
- the LFM2-VL-450M model has the highest loss.

![](./media/train_loss.png)

At this stage we know that for the same training dataset, the LFM2-VL-3B model is able to fit the data better than the other two models.

Apart from this, the train loss curves do not tell us much about the actual model peformance. Language Models are highly parametric neural networks, that can fit **anything** in the training dataset. This **anything** includes both actual patterns in the data, and noise, that does not generalize to the test set.

So, to get a better understanding of the model performance, we need to evaluate the model on a held-out dataset. This is what the evaluation loss curve tells us.

![](./media/eval_loss.png)

Again LFM2-VL-3B is able to fit the data better than the other two models.
When you look at its evaluation loss curve, you can see that it is stricly decreasing, meaning the model is learning epoch by epoch and still has not gotten to the point of overfitting.

The following table shows the evaluation loss at different checkpoints during fine-tuning:

| Checkpoint | Train Loss | Eval Loss |
|------------|------------|-----------|
| 100        | 5.82       | 5.46      |
| 200        | 0.16       | 0.20      |
| 300        | 0.07       | 0.10      |
| 500        | 0.03       | 0.03      |
| 1000        | 0.008       | 0.005      |

> [!TIP]
> **What is overfitting?**
>
> Overfitting is when a model learns the noise in the training data, and does not generalize to the test set. In other words, the training loss is decreasing, but the evaluation loss is increasing.

At this point, we can conclude that LFM2-VL-3B is the most promising model for our use case.

However, we still need to check its actual performance on the test set.

So, let's go back to the evaluation step.

## Step 4. Evaluate the fine-tuned model

To evaluate the fine-tuned model, we will use the `evaluate.py` script again, but this time we will use the last model checkpoint.

```sh
make evaluate config=eval_lfm_3B_checkpoint_1000.yaml
```

### Results

| Checkpoint | Accuracy |
|------------|----------|
| Base Model (LFM2-VL-3B) | 81% |
| checkpoint-1000 | 82% |

The confusion matrix for the fine-tuned model is the following:

![](./media/confusion_matrix_lfm2_3b_checkpoint_1000.png)

## What's next?

There are 2 things we should do next:

- Train the model for more epochs to see if we can improve the performance further. The latest checkpoint at 1000 steps is not overfitting, so why not?

- Inspect the dataset for wrong labels! We have seen that the model is not able to identify Ford correctly. Let's inspect the dataset to see if there are any wrong labels.