# Fine-tuning LFM2-VL-1.6B for browser control with GRPO and OpenEnv

This is the code snippet that started it all :-)
https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/browsergym.py

## Environment setup

Make sure you have `uv`. Then simply do:

```
uv sync
```

## Ideas

- Run the OpenEnv environment in Modal using a separte function that is accessible through HTTP.





## How does it work?

The main training function (`src/browser_control/training.py:main()`) implements **GRPO (Group Relative Policy Optimization) training** for a browser automation agent. Here's the step-by-step breakdown:

### 1. Dataset Creation
```python
train_dataset = Dataset.from_dict({"prompt": ["BrowserGym agent"] * DATASET_SIZE})
```
Creates a simple training dataset with `DATASET_SIZE` (default: 100) identical prompts. This serves as the starting point for rollout generation.

### 2. GRPO Configuration Setup
Configures the GRPO trainer with:
- **`vllm_mode="colocate"`**: Runs vLLM inference server alongside training
- **`use_vllm=True`**: Enables fast inference via vLLM
- **Training params**: Learning rate, batch size, epochs, gradient accumulation
- **Generation params**: Max tokens, temperature, generations per prompt
- **Logging**: TrackIO integration for experiment tracking
- **`reward_weights=[1.0, 0.0, 0.0]`**: Uses only the first reward function (`reward_sum`)

### 3. Trainer Initialization
```python
trainer = GRPOTrainer(
    model=MODEL_ID,                           # Vision-language model
    processing_class=processor,               # Multimodal processor
    reward_funcs=[reward_sum, reward_mean, reward_max],  # Multiple reward calculations
    train_dataset=train_dataset,
    args=grpo_config,
    rollout_func=rollout_func,               # Custom browser interaction function
)
```
Creates the GRPO trainer with:
- **Model**: Vision-language model (default: Qwen3-VL-2B-Instruct)
- **Processor**: Handles text + image inputs
- **Reward functions**: Three different reward aggregation methods
- **Rollout function**: Custom function that interacts with BrowserGym environment

### 4. Training Execution
```python
try:
    trainer.train()
finally:
    env.close()
```
Starts the training loop with proper environment cleanup.

### Key Components Integration

The main function orchestrates several key components:

1. **BrowserGym Environment**: Simulated web browser for training
2. **Vision-Language Model**: Processes screenshots + text instructions
3. **GRPO Algorithm**: Policy optimization using group-based rewards
4. **Rollout Function**: Executes browser actions and collects rewards
5. **Reward Functions**: Evaluate agent performance (sum, mean, max of step rewards)

The training loop will:
1. Generate rollouts using the current policy in BrowserGym
2. Calculate rewards based on task completion
3. Update the model using GRPO to improve performance
4. Repeat until convergence or max epochs reached

This creates an agent that learns to navigate web interfaces through reinforcement learning with visual and textual understanding.

## TODOs



