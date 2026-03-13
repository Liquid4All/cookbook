import sys
import copy
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from app.agent import run_agent, SYSTEM_PROMPT
from app.state import home_state, randomize_state
from app.tools.schemas import TOOL_SCHEMAS
from benchmark.tasks import TASKS, _find_last_call, _find_all_calls

_DEFAULT_STATE = copy.deepcopy(home_state)

# Tasks where initial state should be randomized (no fixed history dependency)
_RANDOMIZED_TASKS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13}

# ---------------------------------------------------------------------------
# Paraphrases (template-based for tasks 1-8; hand-crafted for complex tasks)
# ---------------------------------------------------------------------------

TASK_PARAPHRASES = {
    1: [
        "Kitchen lights on please",
        "Can you switch on the kitchen lights?",
        "Light up the kitchen",
        "Kitchen light on",
        "Please turn the kitchen lights on",
        "Get the kitchen lights on",
        "Could you put the kitchen lights on?",
        "Switch the kitchen lights on",
        "I'd like the kitchen lights on",
        "Kitchen lights need to be on",
    ],
    2: [
        "Lock up the front door",
        "Front door locked please",
        "Make sure the front door is locked",
        "Can you lock the front door?",
        "Please lock the front door",
        "Secure the front door",
        "Could you lock the front door for me?",
        "Lock front",
        "Make the front door secure",
        "Front door, please lock it",
    ],
    3: [
        "Warm the house up to 72 degrees",
        "72 degrees heating please",
        "Set the heat to 72",
        "Please heat the house to 72",
        "Can you set heating to 72 degrees?",
        "Turn on the heat at 72",
        "I want it at 72, heating mode",
        "Heat mode, 72 degrees",
        "Set thermostat to 72 for heat",
        "Could you warm it up to 72?",
    ],
    4: [
        "What's the status of everything?",
        "Give me a full home status",
        "Show me all device states",
        "What are all my devices doing?",
        "Status of all devices please",
        "Can you check all the devices?",
        "What's going on with everything?",
        "Give me a summary of the house",
        "Check everything for me",
        "Full status report",
    ],
    5: [
        "Enable movie night",
        "Set up for movie night",
        "Put on movie night",
        "Activate movie night please",
        "Can you put on movie night mode?",
        "Start movie night",
        "Movie night time",
        "I want movie night mode",
        "Switch to movie night",
        "Set the movie night scene",
    ],
    6: [
        "Unlock garage",
        "The garage door should be unlocked",
        "Open the garage door",
        "Can you unlock the garage?",
        "Please unlock the garage door",
        "I need the garage open",
        "Unlock the garage for me",
        "Garage door unlock please",
        "Open up the garage",
        "Let me into the garage",
    ],
    7: [
        "Bedroom light status?",
        "Check if bedroom lights are on",
        "Are the bedroom lights on?",
        "What's the status of the bedroom lights?",
        "Is the bedroom light on?",
        "Can you check the bedroom lights?",
        "Tell me about the bedroom lights",
        "Status of bedroom lighting",
        "Are the bedroom lights currently on?",
        "Check bedroom lights for me",
    ],
    8: [
        "Set AC to 74 degrees",
        "I want it cooler, 74 degrees",
        "Cool it down to 74",
        "Set cooling to 74",
        "Can you set the AC to 74?",
        "74 degrees cooling mode",
        "I need it cooled to 74",
        "Cool the house to 74 degrees",
        "Set the temperature to 74 cool",
        "AC at 74 please",
    ],
    9: [
        "I'm leaving the house for the day",
        "I'm going out, set up the house",
        "Leaving now, prepare the house",
        "Set everything up for me leaving",
        "I'm heading out",
    ],
    10: [
        "Lock the back door and switch off the office lights",
        "Back door locked and office lights off please",
        "Secure the back door, kill the office lights",
        "Back door lock and office lights out",
    ],
    11: [
        "Turn all lights on",
        "All lights on please",
        "Switch on every light in the house",
        "Lights on everywhere",
        "Can you turn on all the lights?",
    ],
    12: [
        "Turn it off",
        "Off please",
        "Switch it off please",
        "Can you turn it off?",
        "Please switch it off",
    ],
    13: [
        "Actually leave the hallway light off",
        "Keep hallway dark",
        "Hallway light should stay off",
        "Don't turn on the hallway light",
        "Actually the hallway one should be off",
    ],
    14: [
        "Make it 2 degrees warmer",
        "Increase temperature by 2",
        "Bump the heat up 2 degrees",
        "Add 2 degrees to the thermostat",
        "Raise it 2 degrees please",
    ],
    15: [
        "Unlock the first door",
        "Undo the first one",
        "The first door, unlock it",
        "Go back and unlock the first one",
        "Unlock the one I mentioned first",
    ],
    16: [
        "Set the lights to half brightness",
        "Can you dim the bedroom lights?",
        "Lower the brightness to 40%",
        "Dim the kitchen lights please",
        "Reduce the hallway lights to 30%",
        "Make the lights dimmer",
        "Can you dim the lights in the living room?",
    ],
    17: [
        "Can you order some food?",
        "I want to order a pizza",
        "Call a cab for me",
        "Book a restaurant for tonight",
        "Play some music please",
        "What's the weather like?",
        "Set a timer for 30 minutes",
    ],
    18: [
        "Switch it on",
        "Can you put it on?",
        "Turn that on",
        "Switch on the thing",
        "I need it on",
        "Turn that device on",
    ],
    19: [
        "I want it to be comfortable",
        "Set it up nicely",
        "Make the temperature perfect",
        "Can you adjust it?",
        "Make the house feel better",
        "Fix the temperature",
    ],
}

# ---------------------------------------------------------------------------
# History variants for tasks with pronoun/back-reference resolution
# ---------------------------------------------------------------------------

TASK_HISTORY_VARIANTS = {
    12: [
        {
            "history": [
                {"role": "user",      "content": "switch on the bedroom light"},
                {"role": "assistant", "content": "The bedroom light has been turned on."},
            ],
            "target_room": "bedroom",
        },
        {
            "history": [
                {"role": "user",      "content": "turn on the office light"},
                {"role": "assistant", "content": "The office light has been turned on."},
            ],
            "target_room": "office",
        },
        {
            "history": [
                {"role": "user",      "content": "kitchen lights on please"},
                {"role": "assistant", "content": "The kitchen light has been turned on."},
            ],
            "target_room": "kitchen",
        },
        {
            "history": [
                {"role": "user",      "content": "switch on the hallway light"},
                {"role": "assistant", "content": "The hallway light has been turned on."},
            ],
            "target_room": "hallway",
        },
    ],
    15: [
        {
            "history": [
                {"role": "user",      "content": "lock the front door"},
                {"role": "assistant", "content": "The front door has been locked."},
                {"role": "user",      "content": "and the garage too"},
                {"role": "assistant", "content": "The garage door has been locked."},
            ],
            "target_door": "front",
        },
        {
            "history": [
                {"role": "user",      "content": "lock the back door"},
                {"role": "assistant", "content": "The back door has been locked."},
                {"role": "user",      "content": "and the side door as well"},
                {"role": "assistant", "content": "The side door has been locked."},
            ],
            "target_door": "back",
        },
        {
            "history": [
                {"role": "user",      "content": "can you lock the garage door"},
                {"role": "assistant", "content": "The garage door has been locked."},
                {"role": "user",      "content": "also lock the front door"},
                {"role": "assistant", "content": "The front door has been locked."},
            ],
            "target_door": "garage",
        },
    ],
}

# ---------------------------------------------------------------------------
# Call-quality filter
# ---------------------------------------------------------------------------

_VALID_ENUMS = {
    "toggle_lights": {
        "room":  {"living_room", "bedroom", "kitchen", "bathroom", "office", "hallway"},
        "state": {"on", "off"},
    },
    "set_thermostat": {
        "mode": {"heat", "cool", "auto"},
    },
    "lock_door": {
        "door":  {"front", "back", "garage", "side"},
        "state": {"lock", "unlock"},
    },
    "get_device_status": {
        "device_type": {"lights", "thermostat", "door", "all"},
    },
    "set_scene": {
        "scene": {"movie_night", "bedtime", "morning", "away", "party"},
    },
    "intent_unclear": {
        "reason": {"ambiguous", "off_topic", "incomplete", "unsupported_device"},
    },
}

_TASK_MAX_CALLS = {
    1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3,
    9: 4, 10: 5, 11: 9, 12: 4, 13: 9, 14: 4, 15: 4,
    16: 2, 17: 2, 18: 2, 19: 2,
}


def _check_call_quality(tool_calls: list[dict], task_id: int) -> bool:
    """Return False if the trace has too many calls or hallucinated enum values."""
    max_calls = _TASK_MAX_CALLS.get(task_id, 10)
    if len(tool_calls) > max_calls:
        return False
    for call in tool_calls:
        valid_enums = _VALID_ENUMS.get(call["name"], {})
        for param, valid_vals in valid_enums.items():
            val = call["args"].get(param)
            if val is not None and val not in valid_vals:
                return False
    return True


# ---------------------------------------------------------------------------
# Verifier factories for history variants
# ---------------------------------------------------------------------------

def _make_pronoun_verifier(room: str):
    """Verifier for task 12 variants: 'it' refers to the given room."""
    def verifier(tool_calls, duration, state):
        from benchmark.run import TaskResult
        call = _find_last_call(tool_calls, "toggle_lights")
        passed = state["lights"][room]["state"] == "off"
        return TaskResult(12, "Turn off light (pronoun reference)", "hard",
                          passed, call["name"] if call else None, passed, duration)
    return verifier


def _make_first_door_verifier(door: str):
    """Verifier for task 15 variants: unlock the first door mentioned in history."""
    def verifier(tool_calls, duration, state):
        from benchmark.run import TaskResult
        door_calls = [c for c in _find_all_calls(tool_calls, "lock_door")
                      if c["args"].get("door") == door]
        call = door_calls[-1] if door_calls else None
        passed = state["doors"][door] == "unlocked"
        return TaskResult(15, "Unlock first door (back-reference)", "hard",
                          passed, call["name"] if call else None, passed, duration)
    return verifier


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def collect_example(task, prompt, backend="openai",
                    history_override=None, verifier_override=None):
    home_state.clear()
    home_state.update(copy.deepcopy(_DEFAULT_STATE))

    if task.id in _RANDOMIZED_TASKS:
        randomize_state()

    initial_state = copy.deepcopy(home_state)

    tool_calls_seen = []
    messages_out = []

    def capture(name, args, result):
        tool_calls_seen.append({"name": name, "args": args})

    history = history_override if history_override is not None else task.history
    run_agent(prompt, history=history, backend=backend,
              on_tool_call=capture, messages_out=messages_out)

    final_state = copy.deepcopy(home_state)
    verifier = verifier_override if verifier_override is not None else task.verifier

    # Task 14 verifier accepts initial_state for relative temperature check
    if task.id == 14 and verifier_override is None:
        result = verifier(tool_calls_seen, 0.0, final_state, initial_state)
    else:
        result = verifier(tool_calls_seen, 0.0, final_state)

    if not result.passed:
        return None

    if not _check_call_quality(tool_calls_seen, task.id):
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
        variants = TASK_HISTORY_VARIANTS.get(task.id) or [None]
        prompts = [task.prompt] + TASK_PARAPHRASES.get(task.id, [])
        passed = failed = 0

        for variant in variants:
            history_override = variant["history"] if variant else None
            if variant and task.id == 12:
                verifier_override = _make_pronoun_verifier(variant["target_room"])
            elif variant and task.id == 15:
                verifier_override = _make_first_door_verifier(variant["target_door"])
            else:
                verifier_override = None

            for prompt in prompts:
                for _ in range(runs):
                    ex = collect_example(task, prompt, backend,
                                         history_override=history_override,
                                         verifier_override=verifier_override)
                    if ex:
                        examples.append(ex)
                        passed += 1
                    else:
                        failed += 1

        stats[task.id] = (task.name, len(prompts) * len(variants), passed, failed)

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
