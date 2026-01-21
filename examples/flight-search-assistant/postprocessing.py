import ast
import json
import re
from typing import Any, Dict, Iterator, List, Tuple

from llama_cpp import CreateChatCompletionStreamResponse


def get_content_and_tool_calls_from_chunks(
    response: Iterator[CreateChatCompletionStreamResponse], print_chunks: bool
) -> Tuple[str, List]:
    """
    Extracts the `content` and `tool_calls` field data from the incoming response iterator.
    If print_chunks, the chunks are printed on console.

    Args:
        response
        print_chunks

    Returns:
        A tuple (content: str, tool_calls: list)
    """
    # Accumulate the full response while printing chunks
    full_content = ""
    tool_calls = []

    for chunk in response:
        delta = chunk["choices"][0]["delta"]

        # Handle text content
        if "content" in delta and delta["content"]:
            if print_chunks:
                print(delta["content"], end="", flush=True)
            full_content += delta["content"]

        # Handle tool calls (if any)
        if "tool_calls" in delta:
            tool_calls.extend(delta["tool_calls"])

    if print_chunks:
        print()  # Newline after streaming completes

    return full_content, tool_calls


def parse_tool_calls_from_content(content: str) -> list[dict]:
    """
    Parse tool calls from content like:
    <think>...</think>
    [search_flights(departure='New York', destination='Paris', date='2026-01-19')]

    Returns list of dicts: [{"name": "func_name", "arguments": {...}}]
    """
    if not content:
        return []

    # Remove <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Match [function_name(args)]
    pattern = r"\[(\w+)\((.*?)\)\]"
    matches = re.findall(pattern, cleaned)

    tool_calls = []
    for func_name, args_str in matches:
        arguments = parse_arguments(args_str)
        tool_calls.append({"name": func_name, "arguments": arguments})

    return tool_calls


def parse_arguments(args_str: str) -> dict:
    """Parse 'key1=value1, key2=value2' into a dict."""
    if not args_str.strip():
        return {}

    arguments = {}
    # Match: key = 'string' | "string" | number | True/False/None
    pattern = r"(\w+)\s*=\s*('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|-?\d+(?:\.\d+)?|True|False|None|\[.*?\])"

    for key, value in re.findall(pattern, args_str):
        try:
            arguments[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            arguments[key] = value  # Keep as string if parsing fails

    return arguments


def execute_tool_calls(
    tool_calls: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    tool_registry: Dict[str, Any],
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Execute parsed tool calls and append results to messages.

    Args:
        tool_calls: List of tool call dicts with 'name' and 'arguments' keys
        messages: Conversation messages list to append results to
        tool_registry: Dict mapping function names to callable functions
        verbose: Whether to print execution details

    Returns:
        The updated messages list with tool results appended
    """
    import uuid

    new_messages = []

    for tool_call in tool_calls:
        function_name = tool_call.get("name")
        function_args = tool_call.get("arguments", {})
        tool_call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}"

        if verbose:
            print(f"  Function: {function_name}")
            print(f"  Arguments: {json.dumps(function_args, indent=4)}")

        # Look up and execute the function
        function_to_call = tool_registry.get(function_name)
        if not function_to_call:
            if verbose:
                print(f"⚠️  Unknown function: {function_name}")
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": json.dumps(
                        {"error": f"Unknown function: {function_name}"}
                    ),
                }
            )
            continue

        try:
            function_response = function_to_call(**function_args)

            if verbose:
                if (
                    isinstance(function_response, dict)
                    and "flights" in function_response
                ):
                    print(
                        f"  ✓ Found {function_response.get('total_flights_found', 0)} flights\n"
                    )
                else:
                    print(f"  ✓ Function executed successfully\n")

            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
            )

        except Exception as e:
            if verbose:
                print(f"  ❌ Error executing {function_name}: {e}\n")
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": json.dumps({"error": str(e)}),
                }
            )

    return new_messages
