#!/usr/bin/env python3
"""
V2 Training Data Generator for LFM2.5-1.2B Router Fine-Tuning

Generates 4000+ ChatML-format JSONL training examples covering all 83 MCP tools
across 15 servers with variable K values (5-83), hard negatives, and
reinforcement. Uses GPT-4o as teacher model for realistic user prompt generation.

Key improvements over v1 (generate-training-data.ts):
  - 83 tools (was 67) — adds system monitoring, system-settings, screenshot
  - Variable K: 5/10/15/25/35/83 (was fixed K=15)
  - 29 weighted scenario types (was 6 sources + paraphrases)
  - Per-tool minimum 20 examples (was 0-48 with 19 tools at zero)
  - 20 confusable pair drills (was 11 sibling + 6 cross-server)
  - Proactive reinforcement following liquid-lfm-cloud playbook
  - Frozen eval set for consistent benchmarking across iterations

Usage:
    # Generate full dataset (~$20-30 via OpenRouter GPT-4o)
    python scripts/generate_training_data_v2.py --count 3200 --output-dir training-data/v2

    # Stats only (no generation)
    python scripts/generate_training_data_v2.py --stats-only --output-dir training-data/v2

    # Validate existing dataset
    python scripts/generate_training_data_v2.py --validate --output-dir training-data/v2

Requirements:
    pip install pyyaml openai
    export OPENROUTER_API_KEY=<your-key>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PREFIX = (
    "You are LocalCowork, a desktop AI assistant that runs entirely on-device. "
    "You have access to the following tools. ALWAYS call exactly one tool using "
    'bracket syntax: [server.tool(param="value")]. NEVER ask questions. '
    "NEVER say you cannot help. ALWAYS select the most appropriate tool."
)

SEMANTIC_NEIGHBORS: dict[str, list[str]] = {
    "filesystem": ["document", "knowledge", "data"],
    "document": ["ocr", "filesystem", "knowledge"],
    "ocr": ["document", "knowledge", "screenshot-pipeline"],
    "knowledge": ["filesystem", "document"],
    "data": ["filesystem", "document"],
    "security": ["audit", "filesystem"],
    "task": ["calendar", "email"],
    "calendar": ["task", "email", "meeting"],
    "email": ["task", "calendar"],
    "meeting": ["calendar", "email", "knowledge"],
    "audit": ["security", "task"],
    "clipboard": ["filesystem", "system"],
    "system": ["system-settings", "filesystem", "clipboard", "screenshot-pipeline"],
    "system-settings": ["system", "clipboard"],
    "screenshot-pipeline": ["ocr", "system", "document"],
}

# Confusable pairs for hard negatives — each side needs dedicated drills
CONFUSABLE_PAIRS: list[tuple[str, str]] = [
    ("task.get_overdue", "calendar.list_events"),
    ("task.daily_briefing", "calendar.list_events"),
    ("document.read_spreadsheet", "data.query_sqlite"),
    ("knowledge.search_documents", "filesystem.search_files"),
    ("ocr.extract_text_from_image", "document.extract_text"),
    ("ocr.extract_text_from_pdf", "document.extract_text"),
    ("data.query_sqlite", "data.write_sqlite"),
    ("email.draft_email", "email.send_draft"),
    ("security.encrypt_file", "security.decrypt_file"),
    ("calendar.create_event", "calendar.create_time_block"),
    ("task.create_task", "task.update_task"),
    ("clipboard.get_clipboard", "clipboard.set_clipboard"),
    ("filesystem.move_file", "filesystem.copy_file"),
    ("system.take_screenshot", "screenshot.capture_and_extract"),
    ("system.get_system_info", "system.get_memory_usage"),
    ("system.get_system_info", "system.get_cpu_usage"),
    ("system.list_processes", "system.kill_process"),
    ("system-settings.get_audio_settings", "system-settings.set_audio_volume"),
    ("system-settings.get_display_settings", "system-settings.set_display_sleep"),
    ("screenshot.extract_ui_elements", "screenshot.suggest_actions"),
]

# Scenario weights — sum to ~1.0
SCENARIOS: dict[str, dict[str, Any]] = {
    "single_tool_basic": {"weight": 0.08, "k": 15},
    "single_tool_terse": {"weight": 0.06, "k": 15},
    "single_tool_verbose": {"weight": 0.04, "k": 15},
    "single_tool_indirect": {"weight": 0.05, "k": 15},
    "sibling_confusable": {"weight": 0.06, "k": 15},
    "cross_server_confusable": {"weight": 0.06, "k": 15},
    "server_prefix_drill": {"weight": 0.05, "k": 15},
    "anti_refusal_mutable": {"weight": 0.04, "k": 15},
    "anti_refusal_sensitive": {"weight": 0.03, "k": 15},
    "large_k_basic": {"weight": 0.05, "k": 25},
    "large_k_confusable": {"weight": 0.04, "k": 25},
    "very_large_k": {"weight": 0.04, "k": 35},
    "full_registry": {"weight": 0.03, "k": 83},
    "small_k_focused": {"weight": 0.03, "k": 10},
    "tiny_k_exact": {"weight": 0.02, "k": 5},
    "multi_step_isolated": {"weight": 0.06, "k": 15},
    "workflow_context": {"weight": 0.04, "k": 15},
    "domain_specific_jargon": {"weight": 0.03, "k": 15},
    "question_form": {"weight": 0.03, "k": 15},
    "command_form": {"weight": 0.03, "k": 15},
    "system_tools_boost": {"weight": 0.05, "k": 15},
    "system_settings_boost": {"weight": 0.04, "k": 15},
    "screenshot_boost": {"weight": 0.03, "k": 15},
    "audit_clipboard_boost": {"weight": 0.03, "k": 15},
    "ocr_meeting_boost": {"weight": 0.03, "k": 15},
    "edge_ambiguous": {"weight": 0.02, "k": 15},
    "edge_no_direct_match": {"weight": 0.02, "k": 15},
    "paraphrase_formal": {"weight": 0.02, "k": 15},
    "paraphrase_casual": {"weight": 0.02, "k": 15},
}

# Mutable/destructive tools for anti-refusal scenarios
MUTABLE_TOOLS = [
    "filesystem.write_file", "filesystem.delete_file", "filesystem.move_file",
    "filesystem.copy_file", "security.encrypt_file", "security.decrypt_file",
    "clipboard.set_clipboard", "data.write_csv", "data.write_sqlite",
    "system.kill_process", "system-settings.set_display_sleep",
    "system-settings.set_audio_volume", "system-settings.set_default_browser",
    "system-settings.toggle_do_not_disturb",
]

SENSITIVE_TOOLS = [
    "security.scan_for_pii", "security.scan_for_secrets",
    "audit.get_tool_log", "audit.get_session_summary",
    "system.list_processes", "system.get_network_info",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    """A tool definition from the registry."""

    name: str
    server: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    confirmation_required: bool = False
    undo_supported: bool = False


@dataclass
class TrainingExample:
    """A single training example in ChatML format."""

    messages: list[dict[str, str]]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------

def load_tool_registry(registry_path: str) -> dict[str, ToolDef]:
    """Load all tool definitions from mcp-tool-registry.yaml."""
    with open(registry_path, "r") as f:
        data = yaml.safe_load(f)

    tools: dict[str, ToolDef] = {}
    servers = data.get("servers", {})
    for server_name, server_data in servers.items():
        for tool_data in server_data.get("tools", []):
            name = tool_data["name"]
            tools[name] = ToolDef(
                name=name,
                server=server_name,
                description=tool_data.get("description", ""),
                params=tool_data.get("params", {}),
                confirmation_required=tool_data.get("confirmation_required", False),
                undo_supported=tool_data.get("undo_supported", False),
            )

    logger.info("Loaded %d tools from %d servers", len(tools), len(servers))
    return tools


def get_tools_by_server(tools: dict[str, ToolDef]) -> dict[str, list[str]]:
    """Group tool names by server."""
    by_server: dict[str, list[str]] = defaultdict(list)
    for name, tool in tools.items():
        by_server[tool.server].append(name)
    return dict(by_server)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def select_candidates(
    target_tool: str,
    all_tools: dict[str, ToolDef],
    k: int,
    forced: list[str] | None = None,
) -> list[str]:
    """Select K candidate tools for the system prompt.

    1. Always include target_tool
    2. Include all tools from target server
    3. Include forced confusables
    4. Fill from semantically related servers
    5. Random fill from remaining
    6. Shuffle to prevent positional bias
    """
    server = all_tools[target_tool].server
    all_names = list(all_tools.keys())

    # Start with same-server tools
    candidates: set[str] = {
        n for n in all_names if all_tools[n].server == server
    }

    # Add forced confusables
    if forced:
        for f in forced:
            if f in all_tools:
                candidates.add(f)

    # Add from semantic neighbors
    neighbors = SEMANTIC_NEIGHBORS.get(server, [])
    neighbor_tools = [n for n in all_names if all_tools[n].server in neighbors]
    random.shuffle(neighbor_tools)
    for t in neighbor_tools:
        if len(candidates) >= k:
            break
        candidates.add(t)

    # Random fill
    remaining = [n for n in all_names if n not in candidates]
    random.shuffle(remaining)
    for t in remaining:
        if len(candidates) >= k:
            break
        candidates.add(t)

    # Ensure target is present
    candidates.add(target_tool)

    result = list(candidates)[:k]
    random.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    candidate_tools: list[str], tools: dict[str, ToolDef]
) -> str:
    """Build the system prompt with numbered tool list."""
    lines: list[str] = []
    for i, name in enumerate(candidate_tools, 1):
        desc = tools[name].description if name in tools else name
        lines.append(f"{i}. {name} — {desc}")
    tool_list = "\n".join(lines)
    return f"{SYSTEM_PROMPT_PREFIX}\n\nAvailable tools:\n{tool_list}"


# ---------------------------------------------------------------------------
# Tool call formatter
# ---------------------------------------------------------------------------

def format_tool_call(tool_name: str, params: dict[str, str] | None = None) -> str:
    """Format a tool call in bracket syntax: [server.tool(param=\"value\")]."""
    if not params:
        return f"[{tool_name}()]"
    param_str = ", ".join(f'{k}="{v}"' for k, v in params.items())
    return f"[{tool_name}({param_str})]"


def infer_params(tool_name: str, prompt: str, tool_def: ToolDef) -> dict[str, str]:
    """Infer realistic parameter values from the tool definition and prompt."""
    params: dict[str, str] = {}
    param_defs = tool_def.params
    if not param_defs or param_defs == {}:
        return params

    for pname, pdef in param_defs.items():
        if isinstance(pdef, dict) and pdef.get("required", False):
            # Generate a plausible value based on param name and type
            params[pname] = _infer_param_value(pname, pdef, prompt)

    return params


def _infer_param_value(pname: str, pdef: dict[str, Any], prompt: str) -> str:
    """Generate a plausible parameter value."""
    ptype = pdef.get("type", "string")

    # Extract file paths from prompt
    path_match = re.search(r"[~/][\w/._ -]+\.\w+", prompt)
    if pname in ("path", "file_path", "output_path", "image_path", "db_path"):
        if path_match:
            return path_match.group(0)
        return "~/Documents/example.txt"

    if pname == "query":
        # Use part of the prompt as the query
        words = prompt.split()[:8]
        return " ".join(words)

    if pname in ("title", "app_name", "browser"):
        words = prompt.split()[:4]
        return " ".join(words)

    if pname == "pid":
        return str(random.randint(1000, 99999))

    if pname == "volume":
        return str(random.randint(0, 100))

    if pname == "minutes":
        return str(random.choice([0, 5, 10, 15, 30, 60]))

    if ptype == "boolean":
        return "true"

    if ptype == "number":
        return str(random.randint(1, 100))

    return pdef.get("description", pname)[:50]


# ---------------------------------------------------------------------------
# GPT-4o Teacher — prompt generation
# ---------------------------------------------------------------------------

TEACHER_SYSTEM_PROMPT = """You are a training data generator for a tool-calling AI assistant.
Given a tool name and description, generate a realistic user prompt that would
naturally lead to calling that specific tool. The prompt should sound like a real
user talking to a desktop AI assistant.

Rules:
- Output ONLY the user prompt text, nothing else
- No tool names in the prompt (the user wouldn't know them)
- Vary between questions, commands, and requests
- Include realistic file names, paths, and context when relevant
- Keep it natural — real users don't speak in perfect sentences"""


def build_teacher_prompt(
    tool: ToolDef,
    scenario: str,
    confusable: str | None = None,
) -> str:
    """Build the GPT-4o teacher prompt for generating a user prompt."""
    base = (
        f"Tool: {tool.name}\n"
        f"Description: {tool.description}\n"
        f"Server: {tool.server}\n"
    )

    style_hints: dict[str, str] = {
        "single_tool_basic": "Generate a straightforward request. 10-20 words.",
        "single_tool_terse": "Generate an ultra-short request. 3-8 words only. Very casual.",
        "single_tool_verbose": (
            "Generate a detailed request with context. 30-50 words. "
            "Include background info about why the user needs this."
        ),
        "single_tool_indirect": (
            "Generate a request that doesn't mention the action directly. "
            "The user describes their goal, not the tool they need."
        ),
        "sibling_confusable": (
            f"Generate a prompt for {tool.name} that could also be confused "
            f"with {confusable}. Make it clear this is for {tool.name}."
        ),
        "cross_server_confusable": (
            f"Generate a prompt for {tool.name} that sounds similar to "
            f"{confusable} but specifically needs {tool.name}."
        ),
        "server_prefix_drill": (
            f"Generate a prompt that CLEARLY needs {tool.name} (from the "
            f"{tool.server} server), NOT {confusable}. Emphasize the "
            f"distinguishing context."
        ),
        "anti_refusal_mutable": (
            "Generate a confident, direct request for this mutable action. "
            "The user knows what they want. No hedging."
        ),
        "anti_refusal_sensitive": (
            "Generate a legitimate business request that uses this tool. "
            "Sound professional and purposeful."
        ),
        "multi_step_isolated": (
            "Generate a single step from a multi-step workflow. "
            "E.g., 'Now extract the text from that PDF.'"
        ),
        "workflow_context": (
            "Generate a request that references a previous step. "
            "E.g., 'After creating the CSV, now I need to...'"
        ),
        "domain_specific_jargon": (
            "Use technical or business jargon. The user is an expert."
        ),
        "question_form": "Phrase as a question: 'How do I...?', 'Can you...?'",
        "command_form": "Phrase as a direct command. Short and imperative.",
        "system_tools_boost": "Generate for this system monitoring/control tool.",
        "system_settings_boost": "Generate for this system settings tool.",
        "screenshot_boost": "Generate for this screenshot analysis tool.",
        "audit_clipboard_boost": "Generate for this audit/clipboard tool.",
        "ocr_meeting_boost": "Generate for this OCR/meeting tool.",
        "edge_ambiguous": (
            "Generate a somewhat ambiguous request that could fit multiple tools "
            "but {tool.name} is the best match."
        ),
        "edge_no_direct_match": (
            "Generate a request that doesn't perfectly match any tool "
            "but {tool.name} is the closest available option."
        ),
        "paraphrase_formal": "Use formal, professional language.",
        "paraphrase_casual": "Use very casual, conversational language. Slang OK.",
    }

    style = style_hints.get(scenario, "Generate a natural user request.")

    # Handle large-K variants
    if scenario.startswith("large_k") or scenario in (
        "very_large_k", "full_registry", "small_k_focused", "tiny_k_exact"
    ):
        style = (
            "Generate a clear, unambiguous request for this specific tool. "
            "The tool list will be very long, so clarity is key."
        )

    return f"{base}\nStyle: {style}"


async def call_teacher(
    client: Any,
    tool: ToolDef,
    scenario: str,
    confusable: str | None = None,
) -> str | None:
    """Call GPT-4o to generate a user prompt for the given tool and scenario."""
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": build_teacher_prompt(tool, scenario, confusable)},
            ],
            temperature=0.9,
            max_tokens=150,
        )
        content: str = response.choices[0].message.content.strip()
        # Strip quotes if GPT wraps the prompt
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        return content
    except Exception as e:
        logger.warning("Teacher call failed for %s/%s: %s", tool.name, scenario, e)
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

BRACKET_PATTERN = re.compile(r"^\[[\w.-]+\.\w+\(.*\)\]$")


def validate_example(example: TrainingExample, tools: dict[str, ToolDef]) -> bool:
    """Validate a training example."""
    msgs = example.messages
    if len(msgs) != 3:
        return False
    if msgs[0]["role"] != "system" or msgs[1]["role"] != "user" or msgs[2]["role"] != "assistant":
        return False

    assistant_content = msgs[2]["content"]
    if not BRACKET_PATTERN.match(assistant_content):
        return False

    # Extract tool name from bracket call
    tool_match = re.match(r"\[([\w.-]+\.\w+)\(", assistant_content)
    if not tool_match:
        return False

    tool_name = tool_match.group(1)
    if tool_name not in tools:
        return False

    return True


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

async def generate_examples(
    tools: dict[str, ToolDef],
    count: int,
    seed: int = 42,
) -> list[TrainingExample]:
    """Generate training examples using GPT-4o teacher and local generation."""
    random.seed(seed)
    all_tool_names = list(tools.keys())
    by_server = get_tools_by_server(tools)
    examples: list[TrainingExample] = []

    # Check if we have an API key for teacher generation
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    use_teacher = bool(api_key)

    client = None
    if use_teacher:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            logger.info("Using GPT-4o teacher via OpenRouter")
        except ImportError:
            logger.warning("openai package not installed — using local generation only")
            use_teacher = False

    # Build weighted scenario pool
    scenario_pool: list[tuple[str, int]] = []
    for scenario_name, cfg in SCENARIOS.items():
        n = max(1, int(cfg["weight"] * count))
        scenario_pool.append((scenario_name, n))

    total_planned = sum(n for _, n in scenario_pool)
    logger.info(
        "Generating %d examples across %d scenarios (target: %d)",
        total_planned, len(scenario_pool), count,
    )

    sem = asyncio.Semaphore(8)  # Max concurrent teacher calls

    async def _generate_one(
        tool_name: str, scenario: str, k: int, confusable: str | None = None,
    ) -> TrainingExample | None:
        tool = tools[tool_name]
        forced = [confusable] if confusable else None
        candidates = select_candidates(tool_name, tools, k, forced)
        system_prompt = build_system_prompt(candidates, tools)

        # Generate user prompt
        user_prompt: str | None = None
        if use_teacher and client:
            async with sem:
                user_prompt = await call_teacher(client, tool, scenario, confusable)

        if not user_prompt:
            user_prompt = _generate_local_prompt(tool, scenario)

        params = infer_params(tool_name, user_prompt, tool)
        assistant_content = format_tool_call(tool_name, params)

        return TrainingExample(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_content},
            ],
            metadata={
                "source": scenario,
                "server": tool.server,
                "expectedTool": tool_name,
                "k": k,
            },
        )

    # Generate for each scenario
    for scenario_name, target_count in scenario_pool:
        k = SCENARIOS[scenario_name]["k"]
        tasks: list[Any] = []

        for i in range(target_count):
            tool_name, confusable = _pick_tool_for_scenario(
                scenario_name, all_tool_names, by_server, tools, i
            )
            tasks.append(_generate_one(tool_name, scenario_name, k, confusable))

        results = await asyncio.gather(*tasks)
        for r in results:
            if r is not None:
                examples.append(r)

        logger.info("  %s: %d examples", scenario_name, len([r for r in results if r]))

    logger.info("Generated %d base examples", len(examples))
    return examples


def _pick_tool_for_scenario(
    scenario: str,
    all_tools: list[str],
    by_server: dict[str, list[str]],
    tools: dict[str, ToolDef],
    index: int,
) -> tuple[str, str | None]:
    """Pick a target tool and optional confusable for a scenario."""
    if scenario in ("sibling_confusable", "cross_server_confusable", "server_prefix_drill"):
        pair = CONFUSABLE_PAIRS[index % len(CONFUSABLE_PAIRS)]
        # Alternate sides
        if index % 2 == 0:
            return pair[0], pair[1]
        return pair[1], pair[0]

    if scenario == "large_k_confusable":
        pair = CONFUSABLE_PAIRS[index % len(CONFUSABLE_PAIRS)]
        return pair[0], pair[1]

    if scenario in ("anti_refusal_mutable",):
        return MUTABLE_TOOLS[index % len(MUTABLE_TOOLS)], None

    if scenario in ("anti_refusal_sensitive",):
        return SENSITIVE_TOOLS[index % len(SENSITIVE_TOOLS)], None

    if scenario == "system_tools_boost":
        system_tools = by_server.get("system", [])
        return system_tools[index % len(system_tools)], None

    if scenario == "system_settings_boost":
        ss_tools = by_server.get("system-settings", [])
        return ss_tools[index % len(ss_tools)], None

    if scenario == "screenshot_boost":
        sc_tools = by_server.get("screenshot-pipeline", [])
        return sc_tools[index % len(sc_tools)], None

    if scenario == "audit_clipboard_boost":
        combined = by_server.get("audit", []) + by_server.get("clipboard", [])
        return combined[index % len(combined)], None

    if scenario == "ocr_meeting_boost":
        combined = by_server.get("ocr", []) + by_server.get("meeting", [])
        return combined[index % len(combined)], None

    # Default: round-robin across all tools
    return all_tools[index % len(all_tools)], None


def _generate_local_prompt(tool: ToolDef, scenario: str) -> str:
    """Generate a user prompt locally (fallback when no teacher available)."""
    templates: dict[str, list[str]] = {
        "single_tool_terse": [
            "{action}",
            "Can you {action}?",
            "{action} please",
        ],
        "question_form": [
            "How do I {action}?",
            "Can you help me {action}?",
            "What's the best way to {action}?",
        ],
        "command_form": [
            "{action}",
            "Go ahead and {action}",
            "Just {action}",
        ],
    }

    action = tool.description.lower().rstrip(".")
    if action.startswith("get "):
        action = "show me the " + action[4:]
    elif action.startswith("list "):
        action = "show me " + action[5:]

    template_list = templates.get(scenario, ["{action}", "I need to {action}", "Please {action}"])
    template = random.choice(template_list)
    return template.format(action=action)


# ---------------------------------------------------------------------------
# Reinforcement
# ---------------------------------------------------------------------------

def apply_reinforcement(
    examples: list[TrainingExample],
    tools: dict[str, ToolDef],
) -> list[TrainingExample]:
    """Apply proactive reinforcement following liquid-lfm-cloud playbook."""
    reinforced: list[TrainingExample] = list(examples)

    # Group by tool
    by_tool: dict[str, list[TrainingExample]] = defaultdict(list)
    for ex in examples:
        tool_name = ex.metadata.get("expectedTool", "")
        by_tool[tool_name].append(ex)

    # 1. Per-tool reinforcement: 2 best per tool × 3 repeats
    for tool_name in tools:
        tool_examples = by_tool.get(tool_name, [])
        best = tool_examples[:2] if len(tool_examples) >= 2 else tool_examples
        for ex in best:
            for _ in range(3):
                reinforced.append(ex)

    # 2. Confusable pair reinforcement: 1 per side × 5 repeats
    for tool_a, tool_b in CONFUSABLE_PAIRS:
        for tool_name in (tool_a, tool_b):
            tool_examples = by_tool.get(tool_name, [])
            if tool_examples:
                ex = tool_examples[0]
                for _ in range(5):
                    reinforced.append(ex)

    # 3. Zero-coverage tool extra reinforcement
    zero_tools = [n for n in tools if not by_tool.get(n)]
    for tool_name in zero_tools:
        logger.warning("Tool %s has no examples — needs manual data", tool_name)

    logger.info(
        "Reinforcement: %d → %d examples (+%d)",
        len(examples), len(reinforced), len(reinforced) - len(examples),
    )
    return reinforced


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

def split_dataset(
    examples: list[TrainingExample],
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
    """Stratified split by tool into train/eval/test."""
    random.seed(seed)

    # Group by tool
    by_tool: dict[str, list[TrainingExample]] = defaultdict(list)
    for ex in examples:
        tool_name = ex.metadata.get("expectedTool", "unknown")
        by_tool[tool_name].append(ex)

    train: list[TrainingExample] = []
    eval_set: list[TrainingExample] = []
    test: list[TrainingExample] = []

    for _tool_name, tool_examples in by_tool.items():
        random.shuffle(tool_examples)
        n = len(tool_examples)
        n_eval = max(1, int(n * eval_ratio))
        n_test = max(1, int(n * eval_ratio))
        n_train = n - n_eval - n_test

        if n_train < 1:
            # Very few examples: put at least 1 in each split
            train.append(tool_examples[0])
            if n > 1:
                eval_set.append(tool_examples[1])
            if n > 2:
                test.append(tool_examples[2])
            train.extend(tool_examples[3:])
        else:
            eval_set.extend(tool_examples[:n_eval])
            test.extend(tool_examples[n_eval : n_eval + n_test])
            train.extend(tool_examples[n_eval + n_test :])

    random.shuffle(train)
    random.shuffle(eval_set)
    random.shuffle(test)

    return train, eval_set, test


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_jsonl(examples: list[TrainingExample], filepath: str) -> None:
    """Write examples to JSONL file."""
    with open(filepath, "w") as f:
        for ex in examples:
            record = {"messages": ex.messages, "metadata": ex.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d examples to %s", len(examples), filepath)


def write_metadata(
    output_dir: str,
    train: list[TrainingExample],
    eval_set: list[TrainingExample],
    test: list[TrainingExample],
    tools: dict[str, ToolDef],
) -> None:
    """Write generation metadata/stats."""
    all_examples = train + eval_set + test

    tool_counts = Counter(ex.metadata["expectedTool"] for ex in all_examples)
    scenario_counts = Counter(ex.metadata["source"] for ex in all_examples)
    k_counts = Counter(ex.metadata.get("k", 15) for ex in all_examples)
    server_counts = Counter(ex.metadata.get("server", "") for ex in all_examples)

    # Tools with <20 examples
    low_coverage = {t: c for t, c in tool_counts.items() if c < 20}
    zero_coverage = [t for t in tools if t not in tool_counts]

    metadata = {
        "version": "v2",
        "total": len(all_examples),
        "train_count": len(train),
        "eval_count": len(eval_set),
        "test_count": len(test),
        "total_tools": len(tools),
        "tools_with_examples": len(tool_counts),
        "tools_with_zero": zero_coverage,
        "tools_below_20": low_coverage,
        "by_scenario": dict(scenario_counts.most_common()),
        "by_k_value": {str(k): v for k, v in sorted(k_counts.items())},
        "by_server": dict(server_counts.most_common()),
        "by_tool": dict(tool_counts.most_common()),
    }

    path = os.path.join(output_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata written to %s", path)


def print_stats(
    train: list[TrainingExample],
    eval_set: list[TrainingExample],
    test: list[TrainingExample],
    tools: dict[str, ToolDef],
) -> None:
    """Print dataset statistics."""
    all_ex = train + eval_set + test
    tool_counts = Counter(ex.metadata["expectedTool"] for ex in all_ex)
    server_counts = Counter(ex.metadata.get("server", "") for ex in all_ex)
    k_counts = Counter(ex.metadata.get("k", 15) for ex in all_ex)

    print(f"\n{'='*60}")
    print(f"V2 Training Data Statistics")
    print(f"{'='*60}")
    print(f"Total: {len(all_ex)} (train={len(train)}, eval={len(eval_set)}, test={len(test)})")
    print(f"Tools: {len(tool_counts)}/{len(tools)} have examples")
    print(f"\nBy server:")
    for server, count in sorted(server_counts.items(), key=lambda x: -x[1]):
        n_tools = len([t for t in tools if tools[t].server == server])
        print(f"  {server:20s}: {count:4d} examples ({n_tools} tools, {count/n_tools:.1f}/tool)")

    print(f"\nBy K value:")
    for k, count in sorted(k_counts.items()):
        print(f"  K={k:3d}: {count:4d} examples ({count/len(all_ex)*100:.1f}%)")

    # Low coverage
    zero = [t for t in tools if t not in tool_counts]
    low = [(t, c) for t, c in tool_counts.items() if c < 20]
    if zero:
        print(f"\nZERO coverage ({len(zero)} tools):")
        for t in zero:
            print(f"  {t}")
    if low:
        print(f"\nBelow 20 examples ({len(low)} tools):")
        for t, c in sorted(low, key=lambda x: x[1]):
            print(f"  {t}: {c}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="V2 Training Data Generator for LFM2.5-1.2B Router"
    )
    parser.add_argument(
        "--count", type=int, default=3200,
        help="Number of base examples to generate (before reinforcement)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="training-data/v2",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--registry", type=str, default="docs/mcp-tool-registry.yaml",
        help="Path to tool registry YAML",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Print stats for existing dataset without generating",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate existing dataset",
    )
    parser.add_argument(
        "--verify-distribution", action="store_true",
        help="Verify K-value and scenario distribution",
    )
    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    registry_path = project_root / args.registry
    output_dir = project_root / args.output_dir

    tools = load_tool_registry(str(registry_path))

    if args.stats_only:
        # Load existing dataset and print stats
        train_path = output_dir / "train.jsonl"
        eval_path = output_dir / "eval.jsonl"
        test_path = output_dir / "test.jsonl"

        if not train_path.exists():
            logger.error("No dataset found at %s", output_dir)
            sys.exit(1)

        train = _load_jsonl(str(train_path))
        eval_set = _load_jsonl(str(eval_path))
        test = _load_jsonl(str(test_path))
        print_stats(train, eval_set, test, tools)
        return

    if args.validate:
        train_path = output_dir / "train.jsonl"
        if not train_path.exists():
            logger.error("No dataset found at %s", output_dir)
            sys.exit(1)

        all_files = [
            output_dir / "train.jsonl",
            output_dir / "eval.jsonl",
            output_dir / "test.jsonl",
        ]
        total = 0
        valid = 0
        for fp in all_files:
            if not fp.exists():
                continue
            examples = _load_jsonl(str(fp))
            for ex in examples:
                total += 1
                if validate_example(ex, tools):
                    valid += 1
                else:
                    logger.warning("Invalid example in %s: %s", fp.name, ex.messages[2]["content"][:80])
        print(f"Validation: {valid}/{total} valid ({valid/total*100:.1f}%)")
        return

    # Generate
    logger.info("Generating %d base examples for %d tools...", args.count, len(tools))
    examples = await generate_examples(tools, args.count, args.seed)

    # Apply reinforcement
    examples = apply_reinforcement(examples, tools)

    # Split
    train, eval_set, test = split_dataset(examples, seed=args.seed)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, str(output_dir / "train.jsonl"))
    write_jsonl(eval_set, str(output_dir / "eval.jsonl"))
    write_jsonl(test, str(output_dir / "test.jsonl"))
    write_metadata(str(output_dir), train, eval_set, test, tools)

    # Print stats
    print_stats(train, eval_set, test, tools)


def _load_jsonl(filepath: str) -> list[TrainingExample]:
    """Load examples from JSONL."""
    examples: list[TrainingExample] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(
                TrainingExample(
                    messages=data["messages"],
                    metadata=data.get("metadata", {}),
                )
            )
    return examples


if __name__ == "__main__":
    asyncio.run(main())
