import json
import os
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
    "Scenes: movie_night, bedtime, morning, away, party."
)


def run_agent(
    user_message: str,
    history: list[dict] | None = None,
    backend: str = "local",
    on_tool_call=None,
) -> str:
    """Runs the agent loop and returns the final text response."""

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
    for _ in range(max_iter):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=512,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content

        # Check for duplicate calls before appending to messages, so the
        # messages list stays in a valid state for the forced-text fallback.
        duplicate = any(
            f"{tc.function.name}:{json.dumps(json.loads(tc.function.arguments), sort_keys=True)}"
            in seen_calls
            for tc in message.tool_calls
        )
        if duplicate:
            break

        messages.append(message)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
            seen_calls.add(call_key)

            handler = TOOL_HANDLERS.get(name)
            result = handler(**args) if handler else {"error": f"Unknown tool: {name}"}

            if on_tool_call:
                on_tool_call(name, args, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    # Forced text-only call: model summarises what it just did.
    # Reached when the model loops on duplicate tool calls or hits max_iter.
    final = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="none",
        temperature=0.1,
        max_tokens=256,
    )
    return final.choices[0].message.content or "Done."
