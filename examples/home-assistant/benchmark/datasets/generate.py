#!/usr/bin/env python3
"""
benchmark/datasets/generate.py

Generates synthetic SFT training data for the home assistant using a
multi-layer contamination prevention pipeline and cross-validation via
gpt-4o-mini.

Each run writes a timestamped folder containing:
  data.jsonl      training examples in OpenAI format
  config.yaml     full record of every parameter used

Usage:
    # No flags: uniform distribution, 500 examples
    uv run python benchmark/datasets/generate.py --dry-run --count 500

    # Fully automatic: derive weights from a baseline benchmark run
    uv run python benchmark/datasets/generate.py --dry-run \\
        --from-results benchmark/results/2026-03-27_12-01-50_LFM2-350M-Q8_0.gguf.md

    uv run python benchmark/datasets/generate.py --count 500 \\
        --from-results benchmark/results/2026-03-27_12-01-50_LFM2-350M-Q8_0.gguf.md

    # Auto weights + manual boost on top (e.g. double rejection examples)
    uv run python benchmark/datasets/generate.py --count 500 \\
        --from-results benchmark/results/2026-03-27_12-01-50_LFM2-350M-Q8_0.gguf.md \\
        --capability-weights rejection=2

    # Manual weights only, no results file
    uv run python benchmark/datasets/generate.py --count 500 \\
        --capability-weights thermostat=5 rejection=8 lights=0.5 \\
        --depth-weights boundary=6
"""

import argparse
import copy
import json
import math
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

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


# ---------------------------------------------------------------------------
# Taxonomy constants
# ---------------------------------------------------------------------------

CAPABILITIES = ["lights", "thermostat", "doors", "status", "scene", "rejection", "multi_tool"]
PHRASINGS    = ["imperative", "colloquial", "implicit", "question"]
DEPTHS       = ["literal", "semantic", "boundary"]


def _default_output() -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return ROOT / "benchmark" / "datasets" / f"sft_data_{ts}"


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
# Per-cell worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def _process_cell(cell: dict, quota: int) -> dict:
    """Process one taxonomy cell and return picklable results."""
    guard = ContaminationGuard()

    local_stats: dict[str, int] = {
        "total_generated": 0,
        "contaminated": 0,
        "validation_rejected": 0,
        "accepted": 0,
    }
    local_coverage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    local_records: list[dict] = []

    accepted_for_cell = 0
    batches_done = 0
    label = f"{cell['capability']}/{cell['phrasing']}/{cell['depth']}"

    while accepted_for_cell < quota and batches_done < MAX_BATCHES_PER_CELL:
        batch = _generate_batch(cell, k=BATCH_SIZE)
        batches_done += 1
        local_stats["total_generated"] += len(batch)

        for example in batch:
            if accepted_for_cell >= quota:
                break

            utterance = example.get("utterance", "").strip()
            if not utterance:
                continue

            expected = example.get("tool_calls", [])

            # Layers 2+3: contamination guard
            if guard.is_contaminated(utterance):
                local_stats["contaminated"] += 1
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
                local_stats["validation_rejected"] += 1
                continue

            actual_calls = _extract_all_tool_calls(messages_out)
            assistant_msg = _first_tool_call_message(messages_out)

            if not assistant_msg or not _tool_calls_match(expected, actual_calls):
                local_stats["validation_rejected"] += 1
                continue

            local_records.append(_format_record(utterance, assistant_msg, cell))
            accepted_for_cell += 1
            local_stats["accepted"] += 1
            local_coverage[cell["capability"]][cell["depth"]] += 1

    return {
        "records": local_records,
        "stats": local_stats,
        "coverage": {cap: dict(depths) for cap, depths in local_coverage.items()},
        "label": label,
        "accepted": accepted_for_cell,
        "quota": quota,
    }


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def _parse_cell_scores(path: Path) -> tuple[dict, dict]:
    """
    Parse a benchmark result file.

    Returns:
        cell_scores: {(capability, phrasing, depth): (n_pass, n_total)}
        marginals: {"capability": {name: rate}, "phrasing": {name: rate}, "depth": {name: rate}}
    """
    text = path.read_text()
    lines = text.splitlines()

    cell_scores: dict[tuple, tuple[int, int]] = {}
    marginals: dict[str, dict[str, float]] = {
        "capability": {},
        "phrasing": {},
        "depth": {},
    }

    # Parse ## Tasks code block
    in_tasks = False
    in_code = False
    past_header = False
    for line in lines:
        stripped = line.strip()
        if stripped == "## Tasks":
            in_tasks = True
            continue
        if not in_tasks:
            continue
        if stripped.startswith("```"):
            if not in_code:
                in_code = True
            else:
                break  # closing fence: done with Tasks section
            continue
        if not in_code:
            continue
        # Skip the header row and separator row
        if stripped.startswith("#") or all(c == "-" for c in stripped):
            past_header = True
            continue
        if not past_header or not stripped:
            continue
        # Data row: split on 2+ spaces to isolate fixed-width columns
        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 6:
            continue
        cap, phrasing, depth, pass_fail = parts[2], parts[3], parts[4], parts[5]
        key = (cap, phrasing, depth)
        n_pass, n_total = cell_scores.get(key, (0, 0))
        n_total += 1
        if pass_fail == "PASS":
            n_pass += 1
        cell_scores[key] = (n_pass, n_total)

    # Parse ## Breakdown section
    in_breakdown = False
    current_dim: str | None = None
    rate_re = re.compile(r"^\s+(\w+)\s+([\d.]+)%")
    for line in lines:
        if line.strip() == "## Breakdown":
            in_breakdown = True
            continue
        if not in_breakdown:
            continue
        if "**By capability:**" in line:
            current_dim = "capability"
        elif "**By phrasing:**" in line:
            current_dim = "phrasing"
        elif "**By inference depth:**" in line:
            current_dim = "depth"
        elif current_dim:
            m = rate_re.match(line)
            if m:
                marginals[current_dim][m.group(1)] = float(m.group(2)) / 100.0

    return cell_scores, marginals


def _cell_weights_from_results(path: Path, epsilon: float = 0.1) -> dict[tuple, float]:
    """Derive inverse-score cell weights from a benchmark result file.

    weight = 1 / (score + epsilon)

    Cells not covered by the benchmark use the mean of their capability,
    phrasing, and depth marginal rates as a fallback score.
    """
    cell_scores, marginals = _parse_cell_scores(path)
    weights: dict[tuple, float] = {}
    for cell in GENERATION_SPEC:
        cap = cell["capability"]
        phrasing = cell["phrasing"]
        depth = cell["depth"]
        key = (cap, phrasing, depth)
        if key in cell_scores:
            n_pass, n_total = cell_scores[key]
            score = n_pass / n_total if n_total > 0 else 0.0
        else:
            cap_rate = marginals["capability"].get(cap, 0.5)
            phr_rate = marginals["phrasing"].get(phrasing, 0.5)
            dep_rate = marginals["depth"].get(depth, 0.5)
            score = (cap_rate + phr_rate + dep_rate) / 3.0
        weights[key] = 1.0 / (score + epsilon)
    return weights


def _parse_weights(args_list: list[str], valid_keys: list[str]) -> dict[str, float]:
    """Parse ["key=value", ...] CLI tokens into {key: float}.

    Only explicitly provided keys are included. Missing keys default to 1.0
    at the point of use in _compute_quotas. Exits with a clear error on
    unknown keys or non-numeric values.
    """
    if not args_list:
        return {}
    result: dict[str, float] = {}
    for token in args_list:
        if "=" not in token:
            print(f"Error: expected key=value format, got {token!r}", file=sys.stderr)
            sys.exit(1)
        key, _, val = token.partition("=")
        if key not in valid_keys:
            print(
                f"Error: unknown key {key!r}. Valid keys: {valid_keys}",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            result[key] = float(val)
        except ValueError:
            print(
                f"Error: non-numeric value for {key!r}: {val!r}",
                file=sys.stderr,
            )
            sys.exit(1)
    return result


def _compute_quotas(
    count: int,
    cell_weights: dict[tuple, float] | None,
    cap_w: dict[str, float],
    phrasing_w: dict[str, float],
    depth_w: dict[str, float],
) -> list[tuple[dict, int]]:
    """Compute per-cell generation quotas from combined weights.

    When cell_weights is None, each entry's GENERATION_SPEC quota is used as
    its base weight, preserving the original distribution exactly. When
    cell_weights is provided (from a results file), per-cell inverse scores
    are used instead.

    For each entry:
        w = base_weight * cap_multiplier * phrasing_multiplier * depth_multiplier
        quota = max(1, round(w / total_weight * count))
    """
    ws = []
    for cell in GENERATION_SPEC:
        cap, phr, dep = cell["capability"], cell["phrasing"], cell["depth"]
        if cell_weights is None:
            base_w = float(cell["quota"])
        else:
            base_w = cell_weights.get((cap, phr, dep), 1.0)
        w = base_w * cap_w.get(cap, 1.0) * phrasing_w.get(phr, 1.0) * depth_w.get(dep, 1.0)
        ws.append(w)
    total = sum(ws)
    return [
        (cell, max(1, round(w / total * count)))
        for cell, w in zip(GENERATION_SPEC, ws)
    ]


def _write_config(
    output_dir: Path,
    args_used: dict,
    cell_jobs: list[tuple[dict, int]],
) -> None:
    """Write config.yaml into output_dir recording all generation parameters."""
    results_file = args_used.get("results_file")
    if results_file is not None:
        try:
            results_file_str = str(Path(results_file).relative_to(ROOT))
        except ValueError:
            results_file_str = str(results_file)
    else:
        results_file_str = None

    cell_quotas = {
        f"{cell['capability']}/{cell['phrasing']}/{cell['depth']}": quota
        for cell, quota in cell_jobs
    }
    config = {
        "timestamp": args_used["timestamp"],
        "count": args_used["count"],
        "workers": args_used.get("workers"),
        "results_file": results_file_str,
        "epsilon": args_used.get("epsilon"),
        "capability_weights": args_used.get("capability_weights") or None,
        "phrasing_weights": args_used.get("phrasing_weights") or None,
        "depth_weights": args_used.get("depth_weights") or None,
        "cell_quotas": cell_quotas,
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------

def generate(
    count: int,
    output: Path,
    workers: int = 8,
    results_file: Path | None = None,
    epsilon: float = 0.1,
    cap_weights: dict[str, float] | None = None,
    phrasing_weights: dict[str, float] | None = None,
    depth_weights: dict[str, float] | None = None,
) -> None:
    start_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output.mkdir(parents=True, exist_ok=True)

    cell_weights = _cell_weights_from_results(results_file, epsilon) if results_file else None

    cell_jobs = _compute_quotas(
        count,
        cell_weights,
        cap_weights or {},
        phrasing_weights or {},
        depth_weights or {},
    )

    stats: dict[str, int] = {
        "total_generated": 0,
        "contaminated": 0,
        "validation_rejected": 0,
        "accepted": 0,
    }
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    data_path = output / "data.jsonl"
    with open(data_path, "w") as jsonl_file:
        with ProcessPoolExecutor(max_workers=min(workers, len(GENERATION_SPEC))) as executor:
            future_to_label = {
                executor.submit(_process_cell, cell, quota): f"{cell['capability']}/{cell['phrasing']}/{cell['depth']}"
                for cell, quota in cell_jobs
            }

            total_quota = sum(quota for _, quota in cell_jobs)
            bar = tqdm(
                as_completed(future_to_label),
                total=len(future_to_label),
                desc="Cells",
                unit="cell",
                dynamic_ncols=True,
            )
            for future in bar:
                label = future_to_label[future]
                exc = future.exception()
                if exc is not None:
                    tqdm.write(f"  [error] {label}: {exc}", file=sys.stderr)
                    continue

                result = future.result()
                for record in result["records"]:
                    jsonl_file.write(json.dumps(record) + "\n")
                jsonl_file.flush()
                for key in stats:
                    stats[key] += result["stats"][key]
                for cap, depths in result["coverage"].items():
                    for depth, n in depths.items():
                        coverage[cap][depth] += n

                tqdm.write(f"  + {label:<45} accepted={result['accepted']}/{result['quota']}")
                bar.set_postfix(accepted=f"{stats['accepted']}/{total_quota}")

    _write_config(output, {
        "timestamp": start_ts,
        "count": count,
        "workers": workers,
        "results_file": results_file,
        "epsilon": epsilon,
        "capability_weights": cap_weights,
        "phrasing_weights": phrasing_weights,
        "depth_weights": depth_weights,
    }, cell_jobs)

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
    print(f"  Output folder               : {output}")

    print("\nCoverage matrix (capability x depth):")
    col_w = 10
    header = f"{'Capability':<15}" + "".join(f"{d:>{col_w}}" for d in DEPTHS) + f"{'Total':>{col_w}}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for cap in CAPABILITIES:
        row = coverage.get(cap, {})
        total_row = sum(row.values())
        print(f"{cap:<15}" + "".join(f"{row.get(d, 0):>{col_w}}" for d in DEPTHS) + f"{total_row:>{col_w}}")
    all_totals = {d: sum(coverage.get(c, {}).get(d, 0) for c in CAPABILITIES) for d in DEPTHS}
    grand = sum(all_totals.values())
    print(sep)
    print(f"{'TOTAL':<15}" + "".join(f"{all_totals[d]:>{col_w}}" for d in DEPTHS) + f"{grand:>{col_w}}")


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(
    count: int = DEFAULT_COUNT,
    results_file: Path | None = None,
    epsilon: float = 0.1,
    cap_weights: dict[str, float] | None = None,
    phrasing_weights: dict[str, float] | None = None,
    depth_weights: dict[str, float] | None = None,
) -> None:
    cell_weights = _cell_weights_from_results(results_file, epsilon) if results_file else None

    cell_jobs = _compute_quotas(
        count,
        cell_weights,
        cap_weights or {},
        phrasing_weights or {},
        depth_weights or {},
    )

    cap_totals: dict[str, int] = defaultdict(int)
    phrasing_totals: dict[str, int] = defaultdict(int)
    depth_totals: dict[str, int] = defaultdict(int)

    print("Generation plan (dry run)")
    print("=" * 62)

    if results_file is not None and cell_weights is not None:
        print(f"  Weights derived from: {results_file}")
        cell_scores, _ = _parse_cell_scores(results_file)
        print("\n  Cell weights (score -> weight):")
        for cell in GENERATION_SPEC:
            cap, phrasing, depth = cell["capability"], cell["phrasing"], cell["depth"]
            key = (cap, phrasing, depth)
            w = cell_weights.get(key, 1.0)
            if key in cell_scores:
                n_pass, n_total = cell_scores[key]
                score = n_pass / n_total if n_total else 0.0
                score_str = f"score={score:.2f}"
            else:
                score_str = "score=est "
            label = f"{cap}/{phrasing}/{depth}"
            print(f"    {label:<45} {score_str}  weight={w:.2f}")
        print()

    for cell, quota in cell_jobs:
        cap_totals[cell["capability"]] += quota
        phrasing_totals[cell["phrasing"]] += quota
        depth_totals[cell["depth"]] += quota
        label = f"{cell['capability']}/{cell['phrasing']}/{cell['depth']}"
        print(f"  {label:<40} quota={quota:>4}  tool={cell['tool_name']}")

    print("\nPer-capability totals:")
    for cap in CAPABILITIES:
        print(f"  {cap:<15}: {cap_totals.get(cap, 0)}")

    print("\nPer-phrasing totals:")
    for phr in PHRASINGS:
        print(f"  {phr:<15}: {phrasing_totals.get(phr, 0)}")

    print("\nPer-depth totals:")
    for dep in DEPTHS:
        print(f"  {dep:<15}: {depth_totals.get(dep, 0)}")

    total_scaled = sum(quota for _, quota in cell_jobs)
    print(f"\nTotal target : {total_scaled}")

    total_batches = sum(math.ceil(quota / BATCH_SIZE) for _, quota in cell_jobs)
    est_input_tokens = total_batches * 600
    est_output_tokens = total_batches * 1500
    est_cost = (est_input_tokens / 1_000_000) * 0.15 + (est_output_tokens / 1_000_000) * 0.60
    print(f"\nCost estimate ({GEN_MODEL}, optimistic):")
    print(f"  API batches  : ~{total_batches}")
    print(f"  Input tokens : ~{est_input_tokens:,}")
    print(f"  Output tokens: ~{est_output_tokens:,}")
    print(f"  Estimated cost: ~${est_cost:.2f}  (actual may be higher if examples are rejected)")
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
        "--output", type=Path, default=None,
        help="Output directory path (default: benchmark/datasets/sft_data_<timestamp>/)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel worker processes (default: 8)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print generation plan without calling the API",
    )
    parser.add_argument(
        "--from-results", type=Path, default=None, metavar="PATH",
        help="Benchmark result file to derive cell weights from",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1,
        help="Inverse-weight smoothing floor (default: 0.1)",
    )
    parser.add_argument(
        "--capability-weights", nargs="*", default=[], metavar="KEY=VALUE",
        help="Optional multipliers per capability (e.g. rejection=2 lights=0.5)",
    )
    parser.add_argument(
        "--phrasing-weights", nargs="*", default=[], metavar="KEY=VALUE",
        help="Optional multipliers per phrasing (e.g. question=2)",
    )
    parser.add_argument(
        "--depth-weights", nargs="*", default=[], metavar="KEY=VALUE",
        help="Optional multipliers per depth (e.g. boundary=3)",
    )
    args = parser.parse_args()

    cap_weights = _parse_weights(args.capability_weights or [], CAPABILITIES)
    phrasing_weights = _parse_weights(args.phrasing_weights or [], PHRASINGS)
    depth_weights = _parse_weights(args.depth_weights or [], DEPTHS)

    if args.dry_run:
        dry_run(
            count=args.count,
            results_file=args.from_results,
            epsilon=args.epsilon,
            cap_weights=cap_weights or None,
            phrasing_weights=phrasing_weights or None,
            depth_weights=depth_weights or None,
        )
        return

    output = args.output if args.output is not None else _default_output()
    print(f"Generating {args.count} training examples...")
    print(f"Output folder: {output}\n")
    generate(
        args.count,
        output,
        workers=args.workers,
        results_file=args.from_results,
        epsilon=args.epsilon,
        cap_weights=cap_weights or None,
        phrasing_weights=phrasing_weights or None,
        depth_weights=depth_weights or None,
    )


if __name__ == "__main__":
    main()
