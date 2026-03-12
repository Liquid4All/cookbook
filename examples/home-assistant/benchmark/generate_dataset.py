import sys
import copy
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from app.agent import run_agent, SYSTEM_PROMPT
from app.state import home_state
from app.tools.schemas import TOOL_SCHEMAS
from benchmark.tasks import TASKS

_DEFAULT_STATE = copy.deepcopy(home_state)

TASK_PARAPHRASES = {
    1:  ["Kitchen lights on please",
         "Can you switch on the kitchen lights?",
         "Light up the kitchen",
         "Kitchen light on"],
    2:  ["Lock up the front door",
         "Front door locked please",
         "Make sure the front door is locked"],
    3:  ["Set the heat to 72",
         "Warm the house up to 72 degrees",
         "72 degrees heating please"],
    4:  ["Show me all device states",
         "What's the status of everything?",
         "Give me a full home status"],
    5:  ["Put on movie night",
         "Enable movie night",
         "Set up for movie night"],
    6:  ["Open the garage door",
         "Unlock garage",
         "The garage door should be unlocked"],
    7:  ["Is the bedroom light on?",
         "Bedroom light status?",
         "Check if bedroom lights are on"],
    8:  ["Cool it down to 74",
         "Set AC to 74 degrees",
         "I want it cooler, 74 degrees"],
    9:  ["I'm leaving the house for the day",
         "I'm going out, set up the house",
         "Leaving now, prepare the house"],
    10: ["Lock the back door and switch off the office lights",
         "Back door locked and office lights off please",
         "Secure the back door, kill the office lights"],
    11: ["Turn all lights on",
         "All lights on please",
         "Switch on every light in the house"],
    12: ["Turn it off",
         "Off please",
         "Switch it off please"],
    13: ["Actually leave the hallway light off",
         "Keep hallway dark",
         "Hallway light should stay off"],
    14: ["Make it 2 degrees warmer",
         "Increase temperature by 2",
         "Bump the heat up 2 degrees"],
    15: ["Unlock the first door",
         "Undo the first one",
         "The first door, unlock it"],
}


def collect_example(task, prompt, backend="openai"):
    home_state.clear()
    home_state.update(copy.deepcopy(_DEFAULT_STATE))

    tool_calls_seen = []
    messages_out = []

    def capture(name, args, result):
        tool_calls_seen.append({"name": name, "args": args})

    run_agent(prompt, history=task.history, backend=backend,
              on_tool_call=capture, messages_out=messages_out)

    final_state = copy.deepcopy(home_state)
    result = task.verifier(tool_calls_seen, 0.0, final_state)

    if not result.passed:
        return None

    return {
        "task_id":    task.id,
        "difficulty": task.difficulty,
        "messages":   messages_out,
        "tools":      TOOL_SCHEMAS,
    }


def generate_dataset(runs=20, backend="openai"):
    examples = []
    stats = {}

    for task in TASKS:
        prompts = [task.prompt] + TASK_PARAPHRASES.get(task.id, [])
        passed = failed = 0

        for prompt in prompts:
            for _ in range(runs):
                ex = collect_example(task, prompt, backend)
                if ex:
                    examples.append(ex)
                    passed += 1
                else:
                    failed += 1

        stats[task.id] = (task.name, len(prompts), passed, failed)

    return examples, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate golden SFT dataset using gpt-4o-mini.")
    parser.add_argument("--runs",    type=int, default=20, help="Runs per prompt (default 20)")
    parser.add_argument("--backend", default="openai", choices=["local", "openai"])
    args = parser.parse_args()

    print(f"Generating dataset: {args.runs} runs per prompt, backend={args.backend}")
    examples, stats = generate_dataset(runs=args.runs, backend=args.backend)

    print(f"\n{'ID':<4} {'Task':<45} {'Prompts':<8} {'Passed':<8} {'Failed'}")
    print("-" * 78)
    for tid, (name, n_prompts, passed, failed) in stats.items():
        print(f"{tid:<4} {name:<45} {n_prompts:<8} {passed:<8} {failed}")
    total = sum(p + f for _, _, p, f in stats.values())
    print("-" * 78)
    print(f"Total attempts: {total}   Saved examples: {len(examples)}   "
          f"Pass rate: {100 * len(examples) / total:.0f}%")

    Path("benchmark/datasets").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path("benchmark/datasets") / f"{timestamp}_golden_dataset.jsonl"
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nDataset saved to {out_path}")
