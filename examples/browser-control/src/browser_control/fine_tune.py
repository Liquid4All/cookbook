from datasets import Dataset
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import wandb

from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)

from .config import FineTuningConfig
from .paths import get_path_model_checkpoints

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = get_modal_app("browser-control-fine-tune-with-grpo")
image = get_docker_image()
hf_models_volume = get_volume("hf-model-cache")
model_checkpoints_volume = get_volume('browser-control-fine-tune-with-grpo')

# TODO: make sure the parameter to rollout_func match GRPOTrainer API
def rollout_func(
    prompts: list[str],
    trainer: GRPOTrainer,
    client: BrowserGymEnv,
    system_prompt: str,
    max_steps: int,
) -> dict[str, list]:
    """
    For each prompt in the batch, it executes a complete episode using the rollout_once
    function, collecting model outputs and rewards for GRPO optimization
    """
    episode_prompt_ids: list[list[int]] = []
    episode_completion_ids: list[list[int]] = []
    episode_logprobs: list[list[float]] = []
    completion_rewards: list[float] = []

    print(f"\n[DEBUG] rollout_func called with {len(prompts)} prompts (LLM mode, text-only)")

    for i, prompt_text in enumerate(prompts):
        print(f"[DEBUG] Processing prompt {i + 1}/{len(prompts)}")
        episode = rollout_once(
            trainer=trainer,
            env=client,
            tokenizer=trainer.processing_class,
            system_prompt=system_prompt,
            dataset_prompt=prompt_text,
            max_steps=max_steps,
        )
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        completion_rewards.append(episode["completion_reward"])

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "completion_reward": completion_rewards,
    }

def rollout_once(
    trainer: GRPOTrainer,
    env: BrowserGymEnv,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    dataset_prompt: str,
    max_steps: int,
) -> dict[str, list]:
    """Run one episode and collect training data (text-only, no screenshots)."""
    from trl.experimental.openenv import generate_rollout_completions

    result = env.reset()
    observation = result.observation

    print('Goal: ', observation.goal)
    print('axtree_txt: ', observation.axtree_txt)

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []
    completion_rewards: list[float] = []

    for step_num in range(max_steps):
        if result.done:
            break

        # Create prompt from observation (text-only using accessibility tree)
        goal = observation.goal or dataset_prompt
        axtree = observation.axtree_txt or ""
        error = observation.error if observation.last_action_error else ""

        user_prompt = make_user_prompt(goal, step_num, axtree, error)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Generate action with vLLM
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # Parse and execute action
        action_str = parse_action(completion_text)

        print(f"Step {step_num + 1}: {action_str}")

        # Take action in environment
        result = env.step(BrowserGymAction(action_str=action_str))
        observation = result.observation

        # Track rewards
        step_reward = float(result.reward or 0.0)
        step_rewards.append(step_reward)

        # Reward shaping: success is most important
        if result.done and step_reward > 0:
            completion_rewards.append(1.0)  # Task completed successfully
        elif result.done and step_reward == 0:
            completion_rewards.append(0.0)  # Task failed
        else:
            completion_rewards.append(step_reward)  # Intermediate reward

    # Final reward is based on task completion
    final_reward = completion_rewards[-1] if completion_rewards else 0.0

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "step_rewards": step_rewards,
        "completion_reward": final_reward,
    }

def make_user_prompt(goal: str, step_num: int, axtree: str, error: str = "") -> str:
    """Create user prompt from observation."""
    prompt_parts = [f"Step {step_num + 1}"]

    if goal:
        prompt_parts.append(f"Goal: {goal}")

    if error:
        prompt_parts.append(f"Previous action error: {error}")

    # Include accessibility tree (truncated for context)
    if axtree:
        max_len = 2000
        axtree_truncated = axtree[:max_len] + "..." if len(axtree) > max_len else axtree
        prompt_parts.append(f"Page structure:\n{axtree_truncated}")

    prompt_parts.append("What action do you take?")

    return "\n\n".join(prompt_parts)

def parse_action(response_text: str) -> str:
    """Parse BrowserGym action from model response."""
    # Extract first line that looks like an action
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if "(" in line and ")" in line:
            return line

    # Fallback to noop if no valid action found
    return "noop()"

def reward_completion(completions: list[str], **kwargs) -> list[float]:
    """Reward for task completion."""
    rewards = kwargs.get("completion_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]

def create_peft_config(config: FineTuningConfig) -> LoraConfig | None:
    """
    Creates LoRA configuration from FineTuningConfig.
    Returns None if use_peft is False.
    """
    if not config.use_peft:
        return None

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
        use_rslora=config.use_rslora,
    )

@app.function(
    image=image,
    gpu="A100", #"L40S",
    volumes={
        "/hf_model_cache": hf_models_volume,
        "/model_checkpoints": model_checkpoints_volume,
    },
    secrets=get_secrets(),
    timeout=2 * 60 * 60,  # 2 hours timeout for training
    retries=get_retries(max_retries=1),
    max_inputs=1,
)
def fine_tune(config: FineTuningConfig) -> None:
    """
    Fine tunes a Language Model using the the BrowserGym environment and the GRPO algorithm
    """
    if config.wandb_enabled:
        print(
            f"Initializing WandB experiment {config.wandb_experiment_name}"
        )
        wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_experiment_name,
            config=config.__dict__,
        )
    else:
        import os
        os.environ["WANDB_DISABLED"] = "true"

    print(f'Initializing connection with BrowserGym running on {config.browsergym_url}')
    client = BrowserGymEnv(base_url=config.browsergym_url)

    # load dataset
    dataset = Dataset.from_dict({"prompt": [config.default_goal] * config.dataset_size})

    # Path where we save intermediate and final model checkpoint
    output_dir = get_path_model_checkpoints(config.wandb_experiment_name)

    print('Creating GRPOConfig...')
    grpo_config = GRPOConfig(
        # num_train_epochs=1,
        max_steps=config.dataset_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,

        per_device_train_batch_size=config.per_device_train_batch_size,
        num_generations=config.num_generations,
        generation_batch_size=config.generation_batch_size,
        max_completion_length=config.max_completion_length,

        use_vllm=config.use_vllm,
        vllm_mode=config.vllm_mode,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,

        output_dir=output_dir,
        logging_steps=config.logging_steps,
        report_to="wandb",
    )

    print("Setting up the GRPOTrainer")

    # Create PEFT config if LoRA is enabled
    peft_config = create_peft_config(config)
    if peft_config:
        print(f"LoRA enabled: r={config.lora_r}, alpha={config.lora_alpha}")
        print(f"Target modules: {config.lora_target_modules}")

    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=[reward_completion],
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
        rollout_func=lambda prompts, trainer: rollout_func(
            prompts=prompts,
            trainer=trainer,
            client=client,
            system_prompt=config.system_prompt,
            max_steps=config.max_steps,
        ),
    )

    trainer_stats = trainer.train()

    # Save model (with LoRA adapters if enabled)
    print(f"Saving model to {output_dir}")
    if config.use_peft:
        print("Saving LoRA adapter weights (not full model)")
    trainer.save_model(output_dir)

    if config.push_to_hf:
        print("Pushing model to HuggingFace Hub")
        if config.use_peft:
            print("Note: Pushing LoRA adapters only (~5-10MB vs ~350MB full model)")
        trainer.push_to_hub()

@app.local_entrypoint()
def main(config_file_name: str):
    config = FineTuningConfig.from_yaml(file_name=config_file_name)

    try:
        fine_tune.remote(config=config)
        print("Fine-tuning job completed successfully!")
    except Exception as e:
        print(f"❌ Fine-tuning job failed: {e}")
        raise e

if __name__ == "__main__":
    main()