#!/usr/bin/env python3
"""
Flight Search Assistant with CLI
Supports both OpenAI models and local models via llama-server
"""

import re
from argparse import ArgumentParser
from typing import Any, Iterator, List, cast

from llama_cpp import (
    ChatCompletionRequestMessage,
    CreateChatCompletionStreamResponse,
    Llama,
)

from model import load_model
from postprocessing import (
    execute_tool_calls,
    get_content_and_tool_calls_from_chunks,
    parse_tool_calls_from_content,
)
from tools import tool_registry, tool_schemas

HF_REPO_ID = "LiquidAI/LFM2.5-1.2B-Thinking-GGUF"
HF_FILE_NAME = "LFM2.5-1.2B-Thinking-Q8_0.gguf"
SYSTEM_PROMPT = """You are an AI assistant that can help users search and book flights using a set of tools.
When a user asks a question, determine if a tool should be called to help answer.
If a tool is needed, respond with a tool call using the following format:
Each tool function call should use json-like syntax, e.g., {"name": "speak", "arguments": {"name": "Hello"}}.
If no tool is needed, answer the user directly.
Always use the most relevant tool(s) for the user request.
If a tool returns an error, explain the error to the user.
Be concise and helpful. force json schema."""


def pprint(x):
    import json
    print(json.dumps(x, indent=2))


def print_model_response(response: Iterator[CreateChatCompletionStreamResponse]):
    for chunk in response:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta and delta["content"]:
            print(delta["content"], end="", flush=True)


def run_conversation(
    user_message: str,
    verbose: bool = True,
    max_iterations: int = 1,
    strip_thinking: bool = False,
):
    """
    Run a complete conversation with tool calling.

    Args:
        user_message: User's input message
        verbose: Whether to show detailed output
        max_iterations: Maximum number of iterations (model call -> tool execution -> model call)
        strip_thinking: Whether to strip <think>...</think> blocks from assistant messages
                        before storing in history. Defaults to False. Set to True to help
                        avoid context window overflow with thinking models.

    Returns:
        List of message dictionaries representing the conversation history
    """
    model: Llama = load_model(
        hf_repo_id="LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
        hf_model_file="LFM2.5-1.2B-Thinking-Q8_0.gguf",
    )

    messages: List[ChatCompletionRequestMessage] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"User: {user_message}")
        print(f"{'=' * 80}\n")

    for iteration in range(max_iterations):
        try:
            # Reset KV cache before each call to avoid context overflow
            if iteration > 0:
                model.reset()

            # Send chat completion request with streaming
            response = model.create_chat_completion(
                messages,
                tools=tool_schemas,
                max_tokens=2048,
                temperature=0.05,
                top_k=50,
                top_p=0.1,
                repeat_penalty=1.05,
                stream=True,
            )

            # Extract content and tool_calls from the response
            response = cast(Iterator[CreateChatCompletionStreamResponse], response)
            content, tool_calls = get_content_and_tool_calls_from_chunks(
                response, print_chunks=verbose
            )

            # Optionally strip <think>...</think> blocks to save context window space
            content_for_history = content
            if strip_thinking:
                content_for_history = re.sub(
                    r"<think>.*?</think>\s*", "", content, flags=re.DOTALL
                ).strip()

            new_message: dict[str, Any] = {
                "role": "assistant",
                "content": content_for_history,
            }
            if tool_calls:
                new_message["tool_calls"] = tool_calls
            messages.append(new_message)
            if verbose:
                pprint(new_message)

            # Parse any tool calls from the response content and execute them
            tool_calls = parse_tool_calls_from_content(content)
            if tool_calls:
                messages += execute_tool_calls(
                    tool_calls, messages, tool_registry, verbose=verbose
                )
            else:
                # No tool calls - this is the final response
                final_content = content_for_history
                if final_content:
                    print(f"\nAssistant: {final_content}")
                break

        except Exception as e:
            import traceback

            print(f"❌ Error calling model: {type(e).__name__}: {e}")
            traceback.print_exc()
            return messages


def main():
    """
    Main entry point for the CLI application.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--query", type=str, help="User prompt/query for flight search", required=True
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )

    parser.add_argument(
        "--strip-thinking",
        action="store_true",
        help="Strip <think> blocks from message history to save context window space",
    )

    args = parser.parse_args()

    run_conversation(
        args.query,
        strip_thinking=args.strip_thinking,
        max_iterations=5,
    )


if __name__ == "__main__":
    main()
