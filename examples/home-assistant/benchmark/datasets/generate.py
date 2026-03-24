#!/usr/bin/env python3
"""
benchmark/datasets/generate.py

Generates synthetic SFT training data for the home assistant using a
multi-layer contamination prevention pipeline and cross-validation via
gpt-4o-mini.

Usage:
    uv run python benchmark/datasets/generate.py
    uv run python benchmark/datasets/generate.py --count 500 --output benchmark/datasets/sft_data.jsonl
    uv run python benchmark/datasets/generate.py --dry-run
"""

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from benchmark.tasks import TASKS  # noqa: E402
from app.agent import SYSTEM_PROMPT, run_agent  # noqa: E402
from app.tools.schemas import TOOL_SCHEMAS  # noqa: E402
import app.state as _state_module  # noqa: E402

_gen_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
GEN_MODEL = "gpt-4o-mini"

BATCH_SIZE = 20
MAX_BATCHES_PER_CELL = 10
DEFAULT_COUNT = 500
DEFAULT_OUTPUT = ROOT / "benchmark" / "datasets" / "sft_data.jsonl"


# ---------------------------------------------------------------------------
# Prompt building blocks
# ---------------------------------------------------------------------------

PHRASING_DESC = {
    "imperative": "a direct command",
    "colloquial": "casual or informal phrasing",
    "implicit": "context-driven (user describes a situation, not a direct command)",
    "question": "an interrogative or question form",
}

DEPTH_DESC = {
    "literal": "words map directly to the tool and its arguments",
    "semantic": "meaning is clear but requires inference or translation",
    "boundary": "the request cannot be fulfilled by any available tool",
}


# ---------------------------------------------------------------------------
# Generation spec: one entry per taxonomy cell
# ---------------------------------------------------------------------------

GENERATION_SPEC = [
    # LIGHTS (120 total)
    {"capability": "lights", "phrasing": "imperative", "depth": "literal", "quota": 20,
     "tool_name": "toggle_lights",
     "arg_spec": "room: one of [bedroom, bathroom, office, hallway, kitchen, living_room], state: 'on' or 'off'",
     "desc": "Direct command to turn a specific room's lights on or off"},
    {"capability": "lights", "phrasing": "imperative", "depth": "semantic", "quota": 12,
     "tool_name": "toggle_lights",
     "arg_spec": "room: one of [bedroom, bathroom, office, hallway, kitchen, living_room], state: 'on' or 'off'",
     "desc": "Imply a light change through related activity (e.g. 'get the bedroom ready for reading')"},
    {"capability": "lights", "phrasing": "colloquial", "depth": "literal", "quota": 18,
     "tool_name": "toggle_lights",
     "arg_spec": "room: one of [bedroom, bathroom, office, hallway, kitchen, living_room], state: 'on' or 'off'",
     "desc": "Casual, informal request to toggle room lights"},
    {"capability": "lights", "phrasing": "colloquial", "depth": "semantic", "quota": 10,
     "tool_name": "toggle_lights",
     "arg_spec": "room: one of [bedroom, bathroom, office, hallway, kitchen, living_room], state: 'on' or 'off'",
     "desc": "Casual phrasing implying a light state change (e.g. 'the kitchen is wasting electricity')"},
    {"capability": "lights", "phrasing": "implicit", "depth": "literal", "quota": 18,
     "tool_name": "toggle_lights",
     "arg_spec": "room: one of [bedroom, bathroom, office, hallway, kitchen, living_room], state: 'on' or 'off'",
     "desc": "User enters or leaves a room, implying lights should be toggled"},
    {"capability": "lights", "phrasing": "implicit", "depth": "semantic", "quota": 12,
     "tool_name": "toggle_lights",
     "arg_spec": "room: one of [bedroom, bathroom, office, hallway, kitchen, living_room], state: 'on' or 'off'",
     "desc": "Situational description implying a lighting change (e.g. 'it is dark in here', 'done for the day')"},
    {"capability": "lights", "phrasing": "question", "depth": "literal", "quota": 15,
     "tool_name": "toggle_lights",
     "arg_spec": "room: one of [bedroom, bathroom, office, hallway, kitchen, living_room], state: 'on' or 'off'",
     "desc": "Question asking to turn a specific room's lights on or off"},
    {"capability": "lights", "phrasing": "question", "depth": "semantic", "quota": 15,
     "tool_name": "toggle_lights",
     "arg_spec": "room: one of [bedroom, bathroom, office, hallway, kitchen, living_room], state: 'on' or 'off'",
     "desc": "Question implying a desired lighting outcome (e.g. 'can you make the office brighter?')"},

    # THERMOSTAT (80 total)
    {"capability": "thermostat", "phrasing": "imperative", "depth": "literal", "quota": 18,
     "tool_name": "set_thermostat",
     "arg_spec": "temperature: integer 60-80, mode: 'heat', 'cool', or 'auto'",
     "desc": "Direct command to set the thermostat to a specific temperature and mode"},
    {"capability": "thermostat", "phrasing": "imperative", "depth": "semantic", "quota": 10,
     "tool_name": "set_thermostat",
     "arg_spec": "temperature: integer 60-80, mode: 'heat' or 'cool'",
     "desc": "Imply heating or cooling without specifying an exact temperature (e.g. 'make it warmer')"},
    {"capability": "thermostat", "phrasing": "colloquial", "depth": "literal", "quota": 12,
     "tool_name": "set_thermostat",
     "arg_spec": "temperature: integer 60-80, mode: 'heat', 'cool', or 'auto'",
     "desc": "Casual command to set the thermostat (e.g. 'AC to 74', '65 heat mode')"},
    {"capability": "thermostat", "phrasing": "colloquial", "depth": "semantic", "quota": 10,
     "tool_name": "set_thermostat",
     "arg_spec": "temperature: integer 60-80, mode: 'heat' or 'cool'",
     "desc": "Informal temperature complaint implying a mode change (e.g. 'it is freezing, warm it up')"},
    {"capability": "thermostat", "phrasing": "implicit", "depth": "literal", "quota": 10,
     "tool_name": "set_thermostat",
     "arg_spec": "temperature: integer 60-80, mode: 'heat', 'cool', or 'auto'",
     "desc": "User states a desired temperature without a direct command"},
    {"capability": "thermostat", "phrasing": "implicit", "depth": "semantic", "quota": 10,
     "tool_name": "set_thermostat",
     "arg_spec": "temperature: integer 60-80, mode: 'heat' or 'cool'",
     "desc": "User describes discomfort implying a temperature mode (e.g. 'feels like a sauna')"},
    {"capability": "thermostat", "phrasing": "question", "depth": "literal", "quota": 10,
     "tool_name": "set_thermostat",
     "arg_spec": "temperature: integer 60-80, mode: 'heat', 'cool', or 'auto'",
     "desc": "Question asking to set the thermostat to specific values"},

    # DOORS (80 total)
    {"capability": "doors", "phrasing": "imperative", "depth": "literal", "quota": 18,
     "tool_name": "lock_door",
     "arg_spec": "door: one of [front, back, garage, side], state: 'lock' or 'unlock'",
     "desc": "Direct command to lock or unlock a specific door"},
    {"capability": "doors", "phrasing": "imperative", "depth": "semantic", "quota": 10,
     "tool_name": "lock_door",
     "arg_spec": "door: one of [front, back, garage, side], state: 'lock' or 'unlock'",
     "desc": "Phrase implying a door state without explicit lock/unlock (e.g. 'make the garage secure')"},
    {"capability": "doors", "phrasing": "colloquial", "depth": "literal", "quota": 12,
     "tool_name": "lock_door",
     "arg_spec": "door: one of [front, back, garage, side], state: 'lock' or 'unlock'",
     "desc": "Casual command to lock or unlock a door (e.g. 'front door, lock it')"},
    {"capability": "doors", "phrasing": "colloquial", "depth": "semantic", "quota": 10,
     "tool_name": "lock_door",
     "arg_spec": "door: one of [front, back, garage, side], state: 'lock' or 'unlock'",
     "desc": "Casual phrasing implying a door action (e.g. 'let me into the garage')"},
    {"capability": "doors", "phrasing": "implicit", "depth": "literal", "quota": 10,
     "tool_name": "lock_door",
     "arg_spec": "door: one of [front, back, garage, side], state: 'lock' or 'unlock'",
     "desc": "User states a need without explicit lock/unlock (e.g. 'I need the front door locked')"},
    {"capability": "doors", "phrasing": "implicit", "depth": "semantic", "quota": 10,
     "tool_name": "lock_door",
     "arg_spec": "door: one of [front, back, garage, side], state: 'lock' or 'unlock'",
     "desc": "Situational context implying a door action (e.g. 'I am leaving through the side')"},
    {"capability": "doors", "phrasing": "question", "depth": "literal", "quota": 10,
     "tool_name": "lock_door",
     "arg_spec": "door: one of [front, back, garage, side], state: 'lock' or 'unlock'",
     "desc": "Question asking to lock or unlock a specific door"},

    # STATUS (50 total)
    {"capability": "status", "phrasing": "imperative", "depth": "literal", "quota": 12,
     "tool_name": "get_device_status",
     "arg_spec": "device_type: one of [lights, thermostat, door, all], room: optional room name for lights",
     "desc": "Direct command to check device status"},
    {"capability": "status", "phrasing": "colloquial", "depth": "literal", "quota": 10,
     "tool_name": "get_device_status",
     "arg_spec": "device_type: one of [lights, thermostat, door, all], room: optional room name for lights",
     "desc": "Casual status check (e.g. 'Thermostat status', 'Door status please')"},
    {"capability": "status", "phrasing": "question", "depth": "literal", "quota": 15,
     "tool_name": "get_device_status",
     "arg_spec": "device_type: one of [lights, thermostat, door, all], room: optional room name for lights",
     "desc": "Question about current device state (e.g. 'Are the bedroom lights on?')"},
    {"capability": "status", "phrasing": "question", "depth": "semantic", "quota": 13,
     "tool_name": "get_device_status",
     "arg_spec": "device_type: one of [lights, thermostat, door, all]",
     "desc": "Question implying a status check (e.g. 'Is the house secure?', 'Which lights are on?')"},

    # SCENE (50 total)
    {"capability": "scene", "phrasing": "imperative", "depth": "literal", "quota": 20,
     "tool_name": "set_scene",
     "arg_spec": "scene: one of [movie_night, bedtime, morning, away, party]",
     "desc": "Direct command to activate a named home scene"},
    {"capability": "scene", "phrasing": "colloquial", "depth": "literal", "quota": 10,
     "tool_name": "set_scene",
     "arg_spec": "scene: one of [movie_night, bedtime, morning, away, party]",
     "desc": "Casual command to activate a scene"},
    {"capability": "scene", "phrasing": "implicit", "depth": "semantic", "quota": 20,
     "tool_name": "set_scene",
     "arg_spec": "scene: one of [movie_night, bedtime, morning, away, party]",
     "desc": "User describes an activity that maps to a scene (e.g. 'we are about to watch a film')"},

    # REJECTION (60 total)
    {"capability": "rejection", "phrasing": "imperative", "depth": "boundary", "quota": 12,
     "tool_name": "intent_unclear",
     "arg_spec": "reason: one of [ambiguous, off_topic, incomplete, unsupported_device]",
     "desc": "Command for something not supported: unsupported device, out-of-range value, or off-topic request"},
    {"capability": "rejection", "phrasing": "colloquial", "depth": "boundary", "quota": 15,
     "tool_name": "intent_unclear",
     "arg_spec": "reason: one of [ambiguous, off_topic, incomplete, unsupported_device]",
     "desc": "Casual request that is ambiguous, off-topic, incomplete, or for an unsupported feature"},
    {"capability": "rejection", "phrasing": "question", "depth": "boundary", "quota": 15,
     "tool_name": "intent_unclear",
     "arg_spec": "reason: one of [ambiguous, off_topic, incomplete, unsupported_device]",
     "desc": "Question for something the home assistant cannot handle"},
    {"capability": "rejection", "phrasing": "implicit", "depth": "boundary", "quota": 18,
     "tool_name": "intent_unclear",
     "arg_spec": "reason: one of [ambiguous, off_topic, incomplete, unsupported_device]",
     "desc": "Vague or ambiguous statement the assistant cannot act on"},

    # MULTI_TOOL (60 total)
    {"capability": "multi_tool", "phrasing": "imperative", "depth": "literal", "quota": 15,
     "tool_name": "toggle_lights + lock_door",
     "arg_spec": "toggle_lights(room, state) AND lock_door(door, state). Combine a lights action with a door action",
     "desc": "Direct compound command combining lights and door control"},
    {"capability": "multi_tool", "phrasing": "imperative", "depth": "literal", "quota": 10,
     "tool_name": "toggle_lights + set_thermostat",
     "arg_spec": "toggle_lights(room, state) AND set_thermostat(temperature, mode). Combine a lights action with thermostat control",
     "desc": "Direct compound command combining lights and thermostat control"},
    {"capability": "multi_tool", "phrasing": "imperative", "depth": "semantic", "quota": 10,
     "tool_name": "toggle_lights + lock_door",
     "arg_spec": "toggle_lights(room, state) AND lock_door(door, state)",
     "desc": "Implicit compound request combining lights and door actions (e.g. 'lock up and get the bedroom ready')"},
    {"capability": "multi_tool", "phrasing": "colloquial", "depth": "literal", "quota": 10,
     "tool_name": "toggle_lights + lock_door",
     "arg_spec": "toggle_lights(room, state) AND lock_door(door, state)",
     "desc": "Casual compound request for lights and door control"},
    {"capability": "multi_tool", "phrasing": "colloquial", "depth": "semantic", "quota": 8,
     "tool_name": "toggle_lights + lock_door",
     "arg_spec": "toggle_lights(room, state) AND lock_door(door, state)",
     "desc": "Casual phrase implying both lights and door actions"},
    {"capability": "multi_tool", "phrasing": "implicit", "depth": "literal", "quota": 7,
     "tool_name": "toggle_lights + lock_door",
     "arg_spec": "toggle_lights(room, state) AND lock_door(door, state)",
     "desc": "Situational statement implying specific lights and door control (e.g. 'I am going to bed, lock the back door and turn off the hallway light')"},
]

TOTAL_QUOTA = sum(c["quota"] for c in GENERATION_SPEC)
assert TOTAL_QUOTA == 500, f"Expected total quota of 500, got {TOTAL_QUOTA}"


# ---------------------------------------------------------------------------
# Contamination guard
# ---------------------------------------------------------------------------

class ContaminationGuard:
    """
    Three-layer check to prevent benchmark data leakage into training.

    Layer 1: exact match (case-insensitive)
    Layer 2: substring containment (either direction)
    Layer 3: character trigram Jaccard similarity > 0.5
    """

    def __init__(self) -> None:
        self._prompts: list[str] = [t.prompt for t in TASKS]

    def _trigrams(self, s: str) -> set[str]:
        s = s.lower()
        return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else set()

    def is_contaminated(self, utterance: str) -> bool:
        u = utterance.lower()
        u_tri = self._trigrams(utterance)
        for bp in self._prompts:
            bp_l = bp.lower()
            if u == bp_l:
                return True
            if bp_l in u or u in bp_l:
                return True
            if u_tri:
                bp_tri = self._trigrams(bp)
                if bp_tri:
                    inter = len(u_tri & bp_tri)
                    union = len(u_tri | bp_tri)
                    if inter / union > 0.5:
                        return True
        return False


# ---------------------------------------------------------------------------
# State save/restore (handlers mutate module-level home_state)
# ---------------------------------------------------------------------------

def _save_state() -> dict:
    return copy.deepcopy(_state_module.home_state)


def _restore_state(saved: dict) -> None:
    _state_module.home_state.clear()
    _state_module.home_state.update(saved)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _benchmark_utterances(capability: str, phrasing: str, depth: str) -> list[str]:
    return [
        t.prompt for t in TASKS
        if t.capability == capability and t.phrasing == phrasing and t.depth == depth
    ]


def _generate_batch(cell: dict, k: int = BATCH_SIZE) -> list[dict]:
    """Call gpt-4o-mini to produce k (utterance, tool_calls) pairs for a cell."""
    blocklist = _benchmark_utterances(cell["capability"], cell["phrasing"], cell["depth"])
    blocklist_str = "\n".join(f"  - {u}" for u in blocklist) if blocklist else "  (none)"

    is_multi = "+" in cell.get("tool_name", "")
    n_calls = "exactly 2 objects" if is_multi else "exactly 1 object"

    prompt = f"""You are creating diverse training examples for a home assistant AI.

System context the model operates under:
{SYSTEM_PROMPT}

Generate {k} varied user messages for this scenario:
  Capability: {cell["capability"]}
  Tool(s): {cell["tool_name"]}
  Phrasing style: {cell["phrasing"]} — {PHRASING_DESC[cell["phrasing"]]}
  Inference depth: {cell["depth"]} — {DEPTH_DESC[cell["depth"]]}
  Scenario: {cell["desc"]}
  Valid arguments: {cell["arg_spec"]}

For each message, produce a JSON object with:
  "utterance": the user message
  "tool_calls": an array of {n_calls}, each with "tool_name" and "arguments"

Rules:
- Every utterance must feel distinct in wording and structure from all others.
- Do NOT reproduce any of these exact phrases from the benchmark:
{blocklist_str}
- Utterances must be natural English a real user would say to a voice assistant.
- All argument values must be valid according to the arg_spec above.

Return a JSON object with one key "examples" containing an array of {k} objects, nothing else."""

    try:
        resp = _gen_client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.9,
            max_tokens=2000,
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("examples", [])
    except Exception as e:
        print(f"  [warn] Generation error: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def _extract_all_tool_calls(messages: list[dict]) -> list[dict]:
    """Collect all tool calls across all assistant turns."""
    calls = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                calls.append({
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"]),
                })
    return calls


def _first_tool_call_message(messages: list[dict]) -> dict | None:
    """Return the first assistant message that contains tool_calls."""
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return msg
    return None


def _tool_calls_match(expected: list[dict], actual: list[dict]) -> bool:
    """
    Single-tool: actual[0] must match expected[0] (name + args).
    Multi-tool: all expected calls must appear in actual (any order).
    """
    if not expected or not actual:
        return False

    if len(expected) == 1:
        return (
            actual[0]["name"] == expected[0]["tool_name"]
            and actual[0]["args"] == expected[0].get("arguments", {})
        )

    actual_set = {
        (a["name"], json.dumps(a["args"], sort_keys=True))
        for a in actual
    }
    return all(
        (e["tool_name"], json.dumps(e.get("arguments", {}), sort_keys=True)) in actual_set
        for e in expected
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_record(utterance: str, assistant_msg: dict, cell: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": utterance},
            {
                "role": "assistant",
                "content": assistant_msg.get("content"),
                "tool_calls": assistant_msg["tool_calls"],
            },
        ],
        "tools": TOOL_SCHEMAS,
        "_meta": {
            "capability": cell["capability"],
            "phrasing": cell["phrasing"],
            "depth": cell["depth"],
            "source": "synthetic",
        },
    }


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------

def generate(count: int, output: Path) -> None:
    guard = ContaminationGuard()
    output.parent.mkdir(parents=True, exist_ok=True)

    scale = count / TOTAL_QUOTA

    stats: dict[str, int] = {
        "total_generated": 0,
        "contaminated": 0,
        "validation_rejected": 0,
        "accepted": 0,
    }
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    results: list[dict] = []

    for cell in GENERATION_SPEC:
        quota = max(1, round(cell["quota"] * scale))
        accepted_for_cell = 0
        batches_done = 0
        label = f"{cell['capability']}/{cell['phrasing']}/{cell['depth']}"
        print(f"  Generating {label} (target={quota})")

        while accepted_for_cell < quota and batches_done < MAX_BATCHES_PER_CELL:
            batch = _generate_batch(cell, k=BATCH_SIZE)
            batches_done += 1
            stats["total_generated"] += len(batch)

            for example in batch:
                if accepted_for_cell >= quota:
                    break

                utterance = example.get("utterance", "").strip()
                if not utterance:
                    continue

                expected = example.get("tool_calls", [])

                # Layers 2+3: contamination guard
                if guard.is_contaminated(utterance):
                    stats["contaminated"] += 1
                    continue

                # Layer 4: cross-validation via agent
                saved = _save_state()
                messages_out: list[dict] = []
                agent_error = False
                try:
                    run_agent(
                        utterance,
                        backend="openai",
                        messages_out=messages_out,
                        temperature=0.0,
                    )
                except Exception as e:
                    print(f"    [warn] Agent error: {e}", file=sys.stderr)
                    agent_error = True
                finally:
                    _restore_state(saved)

                if agent_error:
                    stats["validation_rejected"] += 1
                    continue

                actual_calls = _extract_all_tool_calls(messages_out)
                assistant_msg = _first_tool_call_message(messages_out)

                if not assistant_msg or not _tool_calls_match(expected, actual_calls):
                    stats["validation_rejected"] += 1
                    continue

                results.append(_format_record(utterance, assistant_msg, cell))
                accepted_for_cell += 1
                stats["accepted"] += 1
                coverage[cell["capability"]][cell["depth"]] += 1

        print(f"    accepted {accepted_for_cell}/{quota}")

    with open(output, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    _print_summary(stats, coverage, output)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_summary(stats: dict, coverage: dict, output: Path) -> None:
    total = stats["total_generated"]
    contaminated = stats["contaminated"]
    rejected = stats["validation_rejected"]
    accepted = stats["accepted"]

    def pct(n: int) -> str:
        return f"{100 * n / total:.1f}%" if total else "0.0%"

    print("\n" + "=" * 62)
    print("Generation complete")
    print("=" * 62)
    print(f"  Total candidates generated  : {total}")
    print(f"  Rejected by contamination   : {contaminated} ({pct(contaminated)})")
    print(f"  Rejected by cross-validation: {rejected} ({pct(rejected)})")
    print(f"  Final accepted              : {accepted}")
    print(f"  Output                      : {output}")

    capabilities = ["lights", "thermostat", "doors", "status", "scene", "rejection", "multi_tool"]
    depths = ["literal", "semantic", "boundary"]

    print("\nCoverage matrix (capability x depth):")
    col_w = 10
    header = f"{'Capability':<15}" + "".join(f"{d:>{col_w}}" for d in depths) + f"{'Total':>{col_w}}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for cap in capabilities:
        row = coverage.get(cap, {})
        total_row = sum(row.values())
        print(f"{cap:<15}" + "".join(f"{row.get(d, 0):>{col_w}}" for d in depths) + f"{total_row:>{col_w}}")
    all_totals = {d: sum(coverage.get(c, {}).get(d, 0) for c in capabilities) for d in depths}
    grand = sum(all_totals.values())
    print(sep)
    print(f"{'TOTAL':<15}" + "".join(f"{all_totals[d]:>{col_w}}" for d in depths) + f"{grand:>{col_w}}")


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run() -> None:
    capabilities = ["lights", "thermostat", "doors", "status", "scene", "rejection", "multi_tool"]
    cap_totals: dict[str, int] = defaultdict(int)

    print("Generation plan (dry run)")
    print("=" * 62)
    for cell in GENERATION_SPEC:
        cap_totals[cell["capability"]] += cell["quota"]
        label = f"{cell['capability']}/{cell['phrasing']}/{cell['depth']}"
        print(f"  {label:<40} quota={cell['quota']:>3}  tool={cell['tool_name']}")

    print("\nPer-capability totals:")
    for cap in capabilities:
        print(f"  {cap:<15}: {cap_totals[cap]}")
    print(f"\nTotal target : {TOTAL_QUOTA}")
    print("No API calls made.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic SFT training data for the home assistant."
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help=f"Total examples to generate (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print generation plan without calling the API",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    print(f"Generating {args.count} training examples...")
    print(f"Output: {args.output}\n")
    generate(args.count, args.output)


if __name__ == "__main__":
    main()
