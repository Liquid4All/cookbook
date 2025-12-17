#!/usr/bin/env python3
"""
GRPO training for BrowserGym tasks using TRL's `GRPOTrainer` and the OpenEnv HTTP
environment interface.
"""

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
import os
import re
import textwrap
from typing import Dict, List, Optional

from datasets import Dataset
import numpy as np
from PIL import Image
import requests
from transformers import AutoProcessor
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import prepare_multimodal_messages

from envs.browsergym_env import BrowserGymAction, BrowserGymEnv

from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = get_modal_app("browser-control-training")
image = get_docker_image()
volume = get_volume("hf-model-cache")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/generate/")
VLLM_SERVER_PORT = VLLM_ENDPOINT.split(":")[-1].split("/")[0]
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "16"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
TOP_K = int(os.getenv("TOP_K", "10"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-6))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "8"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "20"))
PER_DEVICE_BATCH_SIZE = int(os.getenv("PER_DEVICE_BATCH_SIZE", "1"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "8"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))
DATASET_SIZE = int(os.getenv("DATASET_SIZE", "100"))
SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", "50"))
OUTPUT_DIR = f"outputs/browsergym-grpo-{MODEL_ID.replace('/', '-')}-{NOW}"
RUN_ID = f"run-{NOW}"
PROJECT_ID = f"browsergym-{MODEL_ID.replace('/', '-')}"
SPACE_ID = "BrowserGym-GRPO-vlm"

BROWSERGYM_BASE_URL = os.getenv("BROWSERGYM_BASE_URL", "http://localhost:8001")
FALLBACK_ACTION = os.getenv("FALLBACK_ACTION", "noop()")

ACTION_PREFIX_RE = re.compile(r"^(action|next action)\s*[:\-]\s*", re.IGNORECASE)
ACTION_PATTERN = re.compile(r"[A-Za-z_]+\s*\(.*\)", re.DOTALL)
CLICK_ACTION_RE = re.compile(r"click\('([^']+)'\)", re.IGNORECASE)
TEXT_ENTRY_ACTION_RE = re.compile(r"(type|fill)\('([^']+)'", re.IGNORECASE)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You control a web browser.
    Commands:
    - noop()
    - click('<BID>')
    - type('selector', 'text to enter')
    - fill('selector', 'text to enter')
    - send_keys('Enter')
    - scroll('down')
    """
).strip()

DEBUG = os.getenv("DEBUG", "0").lower() in {"1", "true", "yes"}
if DEBUG:
    print("=" * 100)
    print("DEBUG mode enabled")
    print("=" * 100)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = getattr(processor, "tokenizer", None)

if tokenizer is None:
    raise RuntimeError("Processor does not expose a tokenizer instance.")
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


env = BrowserGymEnv.from_hub(
    repo_id="burtenshaw/browsergym-env-95313e2",
    env_vars={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-test",
    },
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n... [truncated]"


def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-4:])


def extract_screenshot_image(observation) -> Optional[Image.Image]:
    screenshot = getattr(observation, "screenshot", None)
    array = np.array(screenshot, dtype=np.uint8)
    return Image.fromarray(array)


def extract_clickable_elements(observation) -> List[Dict[str, str]]:
    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    extra_props = obs_dict.get("extra_element_properties", {}) or {}

    clickables: List[Dict[str, str]] = []
    for bid, props in extra_props.items():
        if not props.get("clickable"):
            continue
        bbox = props.get("bbox") or []
        bbox_str = ", ".join(str(item) for item in bbox) if bbox else "?"
        clickables.append({"bid": str(bid), "bbox": bbox_str})

    clickables.sort(key=lambda item: item["bid"])
    return clickables


def build_user_prompt(step: int, observation, history: List[str]) -> str:
    goal = observation.goal or "(not provided)"
    url = observation.url or "(unknown)"
    error_note = "Yes" if observation.last_action_error else "No"

    clickables = extract_clickable_elements(observation)
    if clickables:
        actions_hint = "\n".join(f"    - {item['bid']} (bbox: {item['bbox']})" for item in clickables)
    else:
        actions_hint = "    (none detected)"

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Goal: {goal}
        Current URL: {url}
        Previous steps:
        {build_history_lines(history)}
        Last action error: {error_note}

        Available clickable element IDs: {actions_hint}

        Reply with exactly one BrowserGym action string.
        """
    ).strip()
    return prompt


def parse_model_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION

    lines = response_text.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        line = ACTION_PREFIX_RE.sub("", line)
        match = ACTION_PATTERN.search(line)
        if match:
            action = match.group(0).strip()
            action = re.sub(r"\s+", " ", action)
            return action

    match = ACTION_PATTERN.search(response_text)
    if match:
        action = match.group(0).strip()
        action = re.sub(r"\s+", " ", action)
        return action

    return FALLBACK_ACTION


def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def request_vllm_completion(prompt: str, args: GRPOConfig, images: Optional[List[Image.Image]] = None) -> Dict[str, List]:
    payload: Dict[str, object] = {
        "prompts": [prompt],
        "n": 1,
        "temperature": getattr(args, "temperature", TEMPERATURE),
        "max_tokens": getattr(args, "max_completion_length", MAX_NEW_TOKENS),
        "logprobs": True,
    }

    top_k = getattr(args, "top_k", None)
    if top_k is not None:
        payload["top_k"] = top_k

    top_p = getattr(args, "top_p", None)
    if top_p is not None:
        payload["top_p"] = top_p

    if images:
        payload["images"] = [pil_to_base64(img) for img in images]

    response = requests.post(VLLM_ENDPOINT, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    prompt_ids = data.get("prompt_ids") or data.get("prompt_token_ids") or [[]]
    completion_ids = data.get("completion_ids") or data.get("completion_token_ids") or [[]]
    logprobs = data.get("logprobs") or data.get("completion_logprobs") or [[]]
    texts = data.get("completions") or data.get("completion_texts") or data.get("texts")

    return {
        "prompt_ids": prompt_ids[0] if prompt_ids else [],
        "completion_ids": completion_ids[0] if completion_ids else [],
        "logprobs": [float(lp) for lp in (logprobs[0] if logprobs else [])],
        "text": (texts[0] if texts else None),
    }


# ---------------------------------------------------------------------------
# Rollout function
# ---------------------------------------------------------------------------


def rollout_once(
    env: BrowserGymEnv,
    processor: AutoProcessor,
    args: GRPOConfig,
    dataset_prompt: str,
) -> Dict[str, List]:
    del dataset_prompt  # Not used for BrowserGym rollouts

    result = env.reset()
    observation = result.observation

    prompt_ids: List[int] = []
    completion_ids: List[int] = []
    logprobs: List[float] = []
    history: List[str] = []
    step_rewards: List[float] = []

    tokenizer_inner = getattr(processor, "tokenizer", None)
    if tokenizer_inner is None:
        raise RuntimeError("Processor used in rollout does not expose a tokenizer.")

    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break

        user_prompt = build_user_prompt(step, observation, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        screenshot_image = extract_screenshot_image(observation)
        if screenshot_image is not None:
            structured_messages = prepare_multimodal_messages(messages, [screenshot_image])
            images = [screenshot_image]
        else:
            structured_messages = messages
            images = None

        prompt_text = processor.apply_chat_template(
            structured_messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        vllm_result = request_vllm_completion(prompt_text, args, images=images)

        prompt_ids.extend(vllm_result["prompt_ids"])
        completion_ids.extend(vllm_result["completion_ids"])
        logprobs.extend(vllm_result["logprobs"])

        completion_text = vllm_result.get("text") or tokenizer_inner.decode(
            vllm_result["completion_ids"], skip_special_tokens=True
        )
        action_str = parse_model_action(completion_text)

        result = env.step(BrowserGymAction(action_str=action_str))
        reward = float(result.reward or 0.0)
        step_rewards.append(reward)

        observation = result.observation
        history_line = f"Step {step}: {action_str} -> reward {reward:+.2f}"
        if observation.last_action_error:
            history_line += " ERROR"
        history.append(history_line)

        if result.done:
            break

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "step_rewards": step_rewards,
    }


def rollout_func(prompts: List[str], args: GRPOConfig, processing_class) -> Dict[str, List]:
    all_prompt_ids: List[List[int]] = []
    all_completion_ids: List[List[int]] = []
    all_logprobs: List[List[float]] = []
    per_step_rewards: List[List[float]] = []
    num_generations = args.num_generations or NUM_GENERATIONS

    for _ in range(num_generations):
        for prompt_text in prompts:
            rollout_stats = rollout_once(env, processing_class, args, prompt_text)
            all_prompt_ids.append(rollout_stats["prompt_ids"])
            all_completion_ids.append(rollout_stats["completion_ids"])
            all_logprobs.append(rollout_stats["logprobs"])
            per_step_rewards.append(rollout_stats.get("step_rewards", []))

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "step_rewards": per_step_rewards,
    }


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def reward_sum(completions: List[str], **kwargs: Optional[Dict]) -> List[float]:
    rewards = (kwargs or {}).get("step_rewards") or []
    rewards = [np.sum(reward) for reward in rewards]
    return rewards


def reward_mean(completions: List[str], **kwargs: Optional[Dict]) -> List[float]:
    rewards = (kwargs or {}).get("step_rewards") or []
    rewards = [np.mean(reward) for reward in rewards]
    return rewards


def reward_max(completions: List[str], **kwargs: Optional[Dict]) -> List[float]:
    rewards = (kwargs or {}).get("step_rewards") or []
    rewards = [np.max(reward) for reward in rewards]
    return rewards


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/model_cache": volume},
    secrets=get_secrets(),
    timeout=2 * 60 * 60,  # 2 hours timeout for training
    retries=get_retries(max_retries=1),
    max_inputs=1,
    # privileged=True,  # Required for Docker-in-Docker
)
def main() -> None:
    # Start Docker daemon
    import subprocess
    import time
    
    print("Starting Docker daemon...")
    subprocess.Popen(["/usr/local/bin/start-docker.sh"], shell=True)
    time.sleep(10)  # Wait for Docker to start
    print("Docker daemon started.")
    train_dataset = Dataset.from_dict({"prompt": ["BrowserGym agent"] * DATASET_SIZE})

    grpo_config = GRPOConfig(
        vllm_mode="colocate",
        use_vllm=True,
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,
        logging_steps=1,
        save_strategy="steps",
        save_steps=SAVE_INTERVAL,
        save_total_limit=None,
        report_to="trackio",
        trackio_space_id=SPACE_ID,
        run_name=RUN_ID,
        project=PROJECT_ID,
        reward_weights=[1.0, 0.0, 0.0],
    )

    grpo_config.temperature = TEMPERATURE
    grpo_config.top_k = TOP_K
    grpo_config.run_name = RUN_ID
    grpo_config.project = PROJECT_ID
    grpo_config.trackio_space_id = SPACE_ID

    trainer = GRPOTrainer(
        model=MODEL_ID,
        processing_class=processor,
        reward_funcs=[reward_sum, reward_mean, reward_max],
        train_dataset=train_dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print("Starting GRPO training with BrowserGym environment...")
    print(f"Using {NUM_GENERATIONS} rollouts per dataset prompt")

    try:
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()