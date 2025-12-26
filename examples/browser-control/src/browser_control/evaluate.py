from envs.browsergym_env import BrowserGymEnv, BrowserGymAction
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from PIL import Image
import os

from .config import FineTuningConfig
from .paths import get_path_to_media

config = FineTuningConfig.from_yaml(file_name='lfm2_350m_debugging.yaml')
system_prompt = config.system_prompt
max_steps = config.max_steps
dataset_prompt = config.default_goal
model_name = "Paulescu/LFM2-350M-browsergym-20251224-013119"


def parse_action(response_text: str) -> str:
    """Parse BrowserGym action from model response."""
    # Extract first line that looks like an action
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if "(" in line and ")" in line:
            return line

    # Fallback to noop if no valid action found
    return "noop()"

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

def save_screenshot(screenshot, episode: int, step: int) -> str:
    """Save screenshot to media directory and return the path."""
    media_dir = get_path_to_media()
    screenshot_array = np.array(screenshot, dtype=np.uint8)
    screenshot_image = Image.fromarray(screenshot_array)
    screenshot_path = os.path.join(media_dir, f'episode_{episode}_step_{step}.png')
    screenshot_image.save(screenshot_path)
    return screenshot_path

def test_click_in_browsergym(
    env,
    model,
    tokenizer,
    episodes: int
):
    
    for episode in range(episodes):

        print(f'Episode {episode}')

        result = env.reset()
        observation = result.observation
        screenshot = observation.screenshot

        # save screenshot to media/ dir
        screenshot_path = save_screenshot(screenshot, episode, step=0)
        print(f'Saved screenshot to {screenshot_path}')

        for step_num in range(max_steps):
            if result.done:
                break

            # Create prompt from observation (text-only using accessibility tree)
            goal = observation.goal or dataset_prompt
            axtree = observation.axtree_txt or ""
            error = observation.error if observation.last_action_error else ""

            user_prompt = make_user_prompt(goal, step_num, axtree, error)
            breakpoint()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

            # Decode and extract model response
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

            action_str = parse_action(generated_text)
            print(f"Step {step_num + 1}: {action_str}")

            # Take action in environment
            result = env.step(BrowserGymAction(action_str=action_str))
            observation = result.observation

if __name__ == '__main__':
    client = BrowserGymEnv(base_url=config.browsergym_url)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_click_in_browsergym(client, model, tokenizer, episodes=10)