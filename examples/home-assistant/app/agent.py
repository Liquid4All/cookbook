import ast
import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from app.tools.schemas import TOOL_SCHEMAS
from app.tools.handlers import TOOL_HANDLERS

load_dotenv()

local_client  = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

BACKENDS = {
    "local":  {"client": local_client,  "model": "local"},
    "openai": {"client": openai_client, "model": "gpt-4o-mini"},
}

SYSTEM_PROMPT = (
    "You are a home assistant AI. Use tools to control the home; respond in text when no tool is needed. "
    "Output function calls as JSON.\n"
    "Lights (on/off): bedroom, bathroom, office, hallway, kitchen, living_room.\n"
    "Doors (lock/unlock): front, back, garage, side.\n"
    "Thermostat: temperature 60-80°F, modes: heat, cool, auto.\n"
    "Scenes: movie_night, bedtime, morning, away, party.\n"
    "Call intent_unclear (never plain text) when the request is: "
    "ambiguous (could be satisfied by multiple different home control actions, e.g. 'make it nicer in here' could mean thermostat, lights, or a scene), "
    "off_topic (unrelated to home control), "
    "incomplete (no target device or room specified even after reading conversation history, e.g. 'turn it on' as the opening message), "
    "or unsupported_device (refers to a device or feature not available, e.g. brightness, TV, music)."
)


def _extract_lfm2_tool_calls(content: str) -> list[dict] | None:
    """Parse tool calls from LFM2 text-format content.

    Handles both JSON format:
      <|tool_call_start|>[{"name": "toggle_lights", "arguments": {...}}]<|tool_call_end|>
    and function-call notation:
      <|tool_call_start|>[toggle_lights(room="kitchen", state="on")]<|tool_call_end|>

    Returns a list of {"name": str, "arguments": dict} or None.
    """
    match = re.search(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", content, re.DOTALL)
    if not match:
        return None
    inner = match.group(1).strip()

    # Try JSON array/object format first.
    try:
        parsed = json.loads(inner)
        calls = parsed if isinstance(parsed, list) else [parsed]
        result = []
        for c in calls:
            args = c.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            result.append({"name": c["name"], "arguments": args})
        return result
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fall back to Python function-call notation: [func(a=1, b="x"), ...]
    try:
        tree = ast.parse(inner, mode="eval")
        body = tree.body
        nodes = body.elts if isinstance(body, ast.List) else ([body] if isinstance(body, ast.Call) else [])
        result = []
        for node in nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
                result.append({"name": node.func.id, "arguments": args})
        if result:
            return result
    except Exception:
        pass

    return None



def get_model_name(backend: str, port: int = 8080) -> str:
    if backend == "local":
        try:
            client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="unused")
            models = client.models.list()
            return models.data[0].id.split("_", 2)[-1] if models.data else "unknown"
        except Exception:
            return "unknown"
    return BACKENDS[backend]["model"]


def run_agent(
    user_message: str,
    history: list[dict] | None = None,
    backend: str = "local",
    on_tool_call=None,
    messages_out: list | None = None,
    temperature: float = 0.0,
    raw_tool_call_parsing: bool = False,
    port: int = 8080,
) -> str:
    """Runs the agent loop and returns the final text response."""

    if backend == "local":
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="unused")
        model = "local"
    else:
        backend_cfg = BACKENDS[backend]
        client = backend_cfg["client"]
        model  = backend_cfg["model"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": user_message},
    ]

    seen_calls: set[str] = set()  # Guard against repeated identical tool calls
    max_iter = 5
    final_response = None
    for _ in range(max_iter):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=512,
        )
        message = response.choices[0].message

        # Resolve tool calls: structured (OpenAI format) or LFM2 text format.
        structured_calls = message.tool_calls
        lfm2_calls = None
        if not structured_calls and raw_tool_call_parsing:
            content = message.content or ""
            lfm2_calls = _extract_lfm2_tool_calls(content)

        if not structured_calls and not lfm2_calls:
            messages.append({"role": "assistant", "content": message.content})
            final_response = message.content
            break

        if structured_calls:
            # Check for duplicate calls before appending to messages, so the
            # messages list stays in a valid state for the forced-text fallback.
            def _call_key(tc):
                try:
                    return f"{tc.function.name}:{json.dumps(json.loads(tc.function.arguments), sort_keys=True)}"
                except json.JSONDecodeError:
                    return f"{tc.function.name}:__malformed__"

            duplicate = any(_call_key(tc) in seen_calls for tc in structured_calls)
            if duplicate:
                break

            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in structured_calls
                ],
            })

            for tool_call in structured_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
                seen_calls.add(call_key)

                handler = TOOL_HANDLERS.get(name)
                try:
                    result = handler(**args) if handler else {"error": f"Unknown tool: {name}"}
                except Exception as exc:
                    result = {"error": str(exc)}

                if on_tool_call:
                    on_tool_call(name, args, result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

        else:
            # LFM2 text-format tool calls: parse from content and execute handlers.
            assert lfm2_calls is not None
            duplicate = any(
                f"{c['name']}:{json.dumps(c['arguments'], sort_keys=True)}" in seen_calls
                for c in lfm2_calls
            )
            if duplicate:
                break

            fake_tool_calls = [
                {"id": f"lfm2_{i}", "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                for i, c in enumerate(lfm2_calls)
            ]
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": fake_tool_calls,
            })

            for i, call_dict in enumerate(lfm2_calls):
                name = call_dict["name"]
                args = call_dict["arguments"]

                call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
                seen_calls.add(call_key)

                handler = TOOL_HANDLERS.get(name)
                try:
                    result = handler(**args) if handler else {"error": f"Unknown tool: {name}"}
                except Exception as exc:
                    result = {"error": str(exc)}

                if on_tool_call:
                    on_tool_call(name, args, result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": f"lfm2_{i}",
                    "content": json.dumps(result),
                })

    if final_response is None:
        # Forced text-only call: model summarises what it just did.
        # Reached when the model loops on duplicate tool calls or hits max_iter.
        final = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="none",
            temperature=temperature,
            max_tokens=256,
        )
        final_response = final.choices[0].message.content or "Done."
        messages.append({"role": "assistant", "content": final_response})

    if messages_out is not None:
        messages_out.extend(messages)

    return final_response
