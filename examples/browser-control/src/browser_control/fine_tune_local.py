"""
本地Docker训练入口脚本
移除Modal依赖，适配Docker Compose环境
使用 HTTP REST API（/reset、/step、/close）与BrowserGym交互，通用于训练和Android评估
基于 fine_tune.py 修改

vllm_mode="colocate"：vLLM 与训练共享同一 GPU 进程，TRL 内部管理，
不调用 init_communicator，避免 Blackwell CC 12.1 上 NCCL 挂起问题。
rollout_func 内部使用 trainer.model.generate() 直接推理。
"""

import os
import time
import requests
import torch

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from .config import FineTuningConfig
from .paths import get_path_model_checkpoints


# ---------------------------------------------------------------------------
# HTTP BrowserGym 客户端（无状态 + 有状态 session）
# ---------------------------------------------------------------------------


class BrowserGymHTTPClient:
    """
    同步 HTTP 客户端，封装 BrowserGym 服务器的 REST API。

    协议（有状态多步模式）：
      1. POST /reset  → 返回 {observation, session_id, ...}
      2. POST /step   → body 带 session_id，返回 {observation, reward, done, session_id}
      3. POST /close  → body 带 session_id，释放服务器端浏览器资源

    向后兼容：不传 session_id 时退化为无状态单步行为。
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # 绕过代理直连本地服务
        self._session = requests.Session()
        self._session.trust_env = False  # 忽略 http_proxy 环境变量

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, seed: int | None = None) -> dict:
        """
        POST /reset
        返回 dict，包含：
          observation: dict（含 text, goal, url 等字段）
          session_id:  str（传给 step/close 使用）
          reward:      None
          done:        False
        """
        body: dict = {}
        if seed is not None:
            body["seed"] = seed

        r = self._session.post(
            f"{self.base_url}/reset",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def step(self, session_id: str, action_str: str) -> dict:
        """
        POST /step
        返回 dict，包含：
          observation: dict
          reward:      float | None
          done:        bool
          session_id:  str
        """
        body = {
            "action": {"action_str": action_str},
            "session_id": session_id,
        }
        r = self._session.post(
            f"{self.base_url}/step",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def close(self, session_id: str) -> None:
        """POST /close — 释放服务器端浏览器 session。"""
        try:
            self._session.post(
                f"{self.base_url}/close",
                json={"session_id": session_id},
                timeout=10,
            )
        except Exception:
            pass  # best-effort


# ---------------------------------------------------------------------------
# Reward function：从 rollout_func 传来的 extra_fields["reward"] 中提取
# ---------------------------------------------------------------------------


import re as _re


def browsergym_reward_func(
    prompts: list[str],
    completions: list[str],
    completion_ids=None,
    **kwargs,
) -> list[float | None]:
    """
    TRL reward_func 接口。
    奖励分级：
      1.0  = BrowserGym 任务成功（环境返回 reward=1）
      0.3  = 最后一个非空行格式正确（click('数字') 或 noop()）
      0.1  = 输出中任意位置包含 click('数字') 或 noop()
      0.0  = 完全没有正确格式
    """
    env_rewards = kwargs.get("reward", None)
    print(f"[reward_func] completions={[c[:60] for c in completions]}")
    print(f"[reward_func] env_rewards={env_rewards}")

    _action_pat = _re.compile(r"click\('\d+'\)|noop\(\)", _re.IGNORECASE)

    results = []
    for i, completion in enumerate(completions):
        # 取环境奖励
        if env_rewards is not None:
            if isinstance(env_rewards, (list, tuple)):
                env_r = float(env_rewards[i]) if i < len(env_rewards) else 0.0
            else:
                env_r = float(env_rewards)
        else:
            env_r = 0.0

        # 任务成功：直接用环境奖励
        if env_r >= 1.0:
            results.append(1.0)
            continue

        # 取最后一个非空行
        lines = [l.strip() for l in completion.splitlines() if l.strip()]
        last_line = lines[-1] if lines else ""

        # 最后一行完全匹配正确格式 → 0.3
        if _re.fullmatch(r"click\('\d+'\)|noop\(\)", last_line, _re.IGNORECASE):
            results.append(0.3)
        # 输出中任意位置包含正确格式 → 0.1（有格式意识但位置不对）
        elif _action_pat.search(completion):
            results.append(0.1)
        else:
            results.append(0.0)

    return results


# ---------------------------------------------------------------------------
# Rollout 函数（colocate 模式，直接用 trainer.model.generate()）
# ---------------------------------------------------------------------------


def rollout_func(
    prompts: list[str],
    trainer,
    browsergym_url: str,
    system_prompt: str,
    max_steps: int,
    _state: dict,
) -> dict[str, list]:
    """
    TRL 0.28.0 rollout_func 接口实现（colocate 模式）。

    签名：(prompts, trainer) — GRPOTrainer 内部包装为 self.rollout_func(prompts, self)。

    colocate 模式下 TRL 不调用 init_communicator，vLLM 与训练共享 GPU 进程。
    rollout_func 使用 trainer.model.generate() 做文本生成，避免 NCCL 依赖。

    持久 session 策略：
      _state["session_id"] 跨调用复用同一 BrowserGym session。
      每次 rollout 开始时调用 /reset（复用 session），而非在结束时 /close。
      只有发生不可恢复错误时才放弃 session（清除 _state["session_id"]），
      下次调用再重新建立。这样 Playwright event loop 始终在同一线程存活，
      避免 "no running event loop" 500 错误。
    """
    processing_class = trainer.processing_class
    args = trainer.args
    model = trainer.model
    gym_client = _state.setdefault(
        "gym_client", BrowserGymHTTPClient(base_url=browsergym_url)
    )

    episode_prompt_ids: list[list[int]] = []
    episode_completion_ids: list[list[int]] = []
    episode_logprobs: list[list[float]] = []
    completion_rewards: list[float] = []

    print(f"\n[rollout_func] {len(prompts)} prompts")

    for i, prompt_text in enumerate(prompts):
        print(f"[rollout_func] prompt {i + 1}/{len(prompts)}")
        episode = rollout_once(
            gym_client=gym_client,
            processing_class=processing_class,
            model=model,
            max_completion_length=args.max_completion_length,
            system_prompt=system_prompt,
            task_prompt=prompt_text,
            max_steps=max_steps,
            persistent_state=_state,
        )
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        completion_rewards.append(episode["reward"])
        print(f"[rollout_func] episode {i + 1} reward={episode['reward']:.4f}")

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "reward": completion_rewards,
    }


def rollout_once(
    gym_client: BrowserGymHTTPClient,
    processing_class,
    model,
    max_completion_length: int,
    system_prompt: str,
    task_prompt: str,
    max_steps: int,
    persistent_state: dict,
) -> dict:
    """
    单个 episode：HTTP reset → loop(model.generate → step)。
    不调用 /close，通过 persistent_state["session_id"] 跨 episode 复用 session，
    避免 Playwright event loop 被销毁导致下次 /reset 500。

    返回 {prompt_ids, completion_ids, logprobs, reward}。
    使用 trainer.model.generate() 直接推理，不依赖 vLLM HTTP API 或 NCCL。

    持久 session 策略：
      - 调用 /reset 前检查 persistent_state["session_id"]
      - /reset 成功后更新 persistent_state["session_id"]
      - 发生不可恢复错误时清除 session_id，下次重建
      - 整个训练只在 fine_tune() 退出时（trainer_cleanup）调用 /close
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    # 初始 prompt token ids
    prompt_ids: list[int] = processing_class.apply_chat_template(
        conversation=messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    # --- reset（复用已有 session 或建立新 session）---
    # 如果服务端支持传 session_id，则传入以复用同一 Playwright 线程；
    # 若服务端不支持（忽略该字段），也只是每次建新 session，不会报错。
    reset_body: dict = {}
    existing_session_id = persistent_state.get("session_id")
    if existing_session_id:
        reset_body["session_id"] = existing_session_id

    try:
        r = gym_client._session.post(
            f"{gym_client.base_url}/reset",
            json=reset_body,
            timeout=gym_client.timeout,
        )
        r.raise_for_status()
        reset_resp = r.json()
    except Exception as e:
        print(f"[ERROR] /reset error: {e}，尝试关闭旧 session 后重试")
        # 先尝试关闭可能遗留的 session（包括刚才 /reset 时服务端可能创建的）
        if existing_session_id:
            gym_client.close(existing_session_id)
        persistent_state.pop("session_id", None)
        # 也尝试关闭无 session_id 的默认 session（兼容服务端无 session_id 字段的情况）
        try:
            gym_client._session.post(
                f"{gym_client.base_url}/close",
                json={},
                timeout=10,
            )
        except Exception:
            pass
        import time as _time

        _time.sleep(1)  # 给服务端一点时间释放 session
        # 不带 session_id 重试一次
        r = gym_client._session.post(
            f"{gym_client.base_url}/reset",
            json={},
            timeout=gym_client.timeout,
        )
        r.raise_for_status()
        reset_resp = r.json()

    session_id = reset_resp.get("session_id", "")
    persistent_state["session_id"] = session_id

    obs = reset_resp.get("observation", {})
    observation_text = obs.get("text") or obs.get("axtree_txt") or ""
    goal = obs.get("goal", "")

    print(f"[rollout_once] session={session_id}, goal={goal!r:.60}")
    # --- 调试：打印 reset 返回的观察详情 ---
    print(f"[DEBUG /reset] obs keys: {list(obs.keys())}")
    print(f"[DEBUG /reset] url={obs.get('url', '(none)')!r}")
    print(f"[DEBUG /reset] observation_text:\n{observation_text!r}")

    # 使用与 Android PromptFormatter.kt 完全一致的格式：
    # Step N
    #
    # Goal: <goal>
    #
    # Page structure:
    # <axtree>
    #
    # What action do you take?
    step_num = 1
    obs_message = (
        f"Step {step_num}\n\n"
        f"Goal: {goal}\n\n"
        f"Page structure:\n{observation_text}\n\n"
        f"What action do you take?"
    )
    messages.append({"role": "user", "content": obs_message})

    done = False
    step = 0
    reward = 0.0
    completion_ids: list[int] = []
    logprobs: list[float] = []

    while not done and step < max_steps:
        # 构造当前 prompt
        current_prompt_text: str = processing_class.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # tokenize
        enc = processing_class(
            current_prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        current_prompt_ids: list[int] = input_ids[0].tolist()

        # 推理（用 model.generate，不依赖 vLLM）
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_completion_length,
                do_sample=True,
                temperature=1.0,
                pad_token_id=processing_class.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # 提取生成的 completion token ids（去掉 prompt 部分）
        gen_ids = out.sequences[0][input_ids.shape[1] :]
        completion_ids = gen_ids.tolist()

        # 从 scores 计算 logprobs（每步 top-1 log softmax）
        logprobs = []
        for score, tok_id in zip(out.scores, completion_ids):
            lp = torch.nn.functional.log_softmax(score[0], dim=-1)
            logprobs.append(lp[tok_id].item())

        # 更新 prompt_ids 为本步实际用的 prompt
        prompt_ids = current_prompt_ids

        action_str = processing_class.decode(
            completion_ids, skip_special_tokens=True
        ).strip()

        # 尝试从输出中提取最后一个有效动作行（去掉模型的解释文字）
        _lines = [l.strip() for l in action_str.splitlines() if l.strip()]
        # 大小写不敏感匹配 click('<digits>') 或 noop()
        _action_re = _re.compile(r"click\('\d+'\)|noop\(\)", _re.IGNORECASE)
        # 优先取最后一行完整匹配的；否则取任意位置包含该模式的行并规范化大小写
        _matched = [l for l in _lines if _action_re.fullmatch(l)]
        if _matched:
            action_for_gym = _matched[-1].lower()  # 规范化为小写
        else:
            # 尝试从任意行中提取动作模式
            _found = None
            for l in reversed(_lines):
                m = _action_re.search(l)
                if m:
                    _found = m.group(0).lower()
                    break
            action_for_gym = _found if _found else "noop()"

        print(f"[rollout_once] step {step + 1}, action={action_str!r:.80}")
        if action_for_gym != action_str:
            print(f"[rollout_once] extracted action: {action_for_gym!r}")

        try:
            step_resp = gym_client.step(
                session_id=session_id, action_str=action_for_gym
            )
        except requests.HTTPError as e:
            print(f"[ERROR] /step error: {e}，清除 session_id")
            persistent_state.pop("session_id", None)
            reward = -1.0
            break

        step_obs = step_resp.get("observation", {})
        observation_text = step_obs.get("text") or step_obs.get("axtree_txt") or ""
        reward = float(step_resp.get("reward") or 0.0)
        done = bool(step_resp.get("done", False))

        # --- 调试：打印 step 完整响应 ---
        last_action_error = step_obs.get("last_action_error", "")
        print(f"[DEBUG /step] raw keys: {list(step_resp.keys())}")
        print(
            f"[DEBUG /step] reward={step_resp.get('reward')!r}, done={step_resp.get('done')!r}"
        )
        print(f"[DEBUG /step] last_action_error={last_action_error!r}")
        print(f"[DEBUG /step] observation_text:\n{observation_text!r}")

        messages.append({"role": "assistant", "content": action_str})
        if not done:
            # 后续步骤：与 Android PromptFormatter.kt 格式一致，step_num 递增
            step_num += 1
            obs_message = (
                f"Step {step_num}\n\n"
                f"Goal: {goal}\n\n"
                f"Page structure:\n{observation_text}\n\n"
                f"What action do you take?"
            )
            if last_action_error:
                obs_message = (
                    f"Previous action error: {last_action_error}\n\n" + obs_message
                )
            messages.append({"role": "user", "content": obs_message})

        step += 1

    print(f"[rollout_once] done after {step} steps, reward={reward:.4f}")

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "reward": reward,
    }


# ---------------------------------------------------------------------------
# 连通性验证
# ---------------------------------------------------------------------------


def verify_browsergym(url: str, max_retries: int = 3) -> str:
    """验证 BrowserGym HTTP 服务可达，返回确认过的 base URL。"""
    client = BrowserGymHTTPClient(base_url=url)
    for attempt in range(max_retries):
        if client.health():
            print(f"BrowserGym HTTP connection verified at {url}")
            return url
        if attempt < max_retries - 1:
            print(
                f"BrowserGym not ready (attempt {attempt + 1}/{max_retries}), retrying in 10s..."
            )
            time.sleep(10)
        else:
            raise ConnectionError(f"Cannot reach BrowserGym at {url}/health")
    return url  # unreachable


# ---------------------------------------------------------------------------
# 训练主函数
# ---------------------------------------------------------------------------


def fine_tune(config: FineTuningConfig) -> None:
    """
    使用 BrowserGym 环境和 GRPO 算法微调语言模型。
    本地 Docker 版本（移除 Modal 依赖，使用 HTTP REST API，colocate vLLM 模式）。
    """

    log_dir = f"/workspace/logs/{config.wandb_experiment_name}"
    print(f"TensorBoard logging enabled: {log_dir}")

    print(f"Verifying BrowserGym HTTP server at {config.browsergym_url}")
    browsergym_url = verify_browsergym(config.browsergym_url)

    # Dataset
    dataset = Dataset.from_dict({"prompt": [config.default_goal] * config.dataset_size})

    # Checkpoint path
    output_dir = get_path_model_checkpoints(config.wandb_experiment_name or "local-run")

    print("Creating GRPOConfig...")
    grpo_config = GRPOConfig(
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
        torch_compile=False,
        output_dir=output_dir,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=max(1, config.dataset_size // 5),
        seed=config.seed,
        # TensorBoard：GRPOTrainer 在每个 logging_step 写入 step-level 指标（loss/reward 等）
        report_to="tensorboard",
        logging_dir=log_dir,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir="/hf_model_cache",
    )

    peft_config = None
    if config.use_peft:
        print(f"Using LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type="CAUSAL_LM",
            target_modules=config.lora_target_modules,
            use_rslora=config.use_rslora,
        )

    print("Creating GRPOTrainer (colocate mode)...")
    # 持久 session 状态字典，通过闭包传给 rollout_func，跨 episode 共用
    _rollout_state: dict = {}
    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=[browsergym_reward_func],
        train_dataset=dataset,
        processing_class=tokenizer,
        args=grpo_config,
        peft_config=peft_config,
        rollout_func=lambda prompts, trainer: rollout_func(
            prompts=prompts,
            trainer=trainer,
            browsergym_url=browsergym_url,
            system_prompt=config.system_prompt,
            max_steps=config.max_steps,
            _state=_rollout_state,
        ),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    if config.use_peft:
        print("Saving LoRA adapter weights")
    trainer.save_model(output_dir)

    if config.push_to_hf:
        print("push_to_hf skipped in local training mode")

    print("Training completed successfully!")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Browser Control 本地 Docker 训练")
    parser.add_argument(
        "--config-file-name",
        type=str,
        required=True,
        help="配置文件名（例如：lfm2_350m_local.yaml）",
    )
    args = parser.parse_args()

    config = FineTuningConfig.from_yaml(file_name=args.config_file_name)

    print("=" * 80)
    print("Browser Control 本地 Docker 训练")
    print(f"配置文件: {args.config_file_name}")
    print(f"实验名称: {config.wandb_experiment_name}")
    print(f"模型: {config.model_name}")
    print(f"BrowserGym URL: {config.browsergym_url}")
    print("=" * 80)

    try:
        fine_tune(config=config)
        print("\n训练任务完成！")
    except Exception as e:
        print(f"\n训练任务失败: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
