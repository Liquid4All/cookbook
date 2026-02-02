import argparse
import subprocess
import time
import urllib.request
import urllib.error
from datetime import datetime

from agno.agent import Agent
from rich.console import Console
from rich.panel import Panel
from agno.models.llama_cpp import LlamaCpp
from agno.tools import tool

MODELS = [
    "LiquidAI/LFM2-1.2B-Tool-GGUF:Q8_0",
    "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q8_0",
    "LiquidAI/LFM2.5-1.2B-Thinking-GGUF:Q8_0",
]

DEFAULT_MODEL = "LiquidAI/LFM2-1.2B-Tool-GGUF:Q8_0"


def start_llama_server(model_id: str, verbose: bool = False) -> subprocess.Popen:
    """Start llama-server as a subprocess.

    Args:
        model_id: HuggingFace model ID to load
        verbose: If True, show server output; otherwise suppress it

    Returns:
        The subprocess handle
    """
    cmd = ["llama-server", "-hf", model_id, "--jinja"]
    if verbose:
        process = subprocess.Popen(cmd)
    else:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return process


def wait_for_server(url: str, timeout: int = 30) -> None:
    """Wait for the server to become healthy.

    Args:
        url: Health endpoint URL to poll
        timeout: Maximum seconds to wait

    Raises:
        TimeoutError: If server doesn't respond within timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Server at {url} did not become healthy within {timeout} seconds")


def stop_server(process: subprocess.Popen) -> None:
    """Stop the llama-server subprocess gracefully.

    Args:
        process: The subprocess handle to terminate
    """
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

@tool
def get_current_datetime() -> dict:
    """Get the current date and time.

    Returns:
        dict: A dictionary containing:
            - date: Current date in YYYY-MM-DD format
            - time: Current time in HH:MM:SS format
            - weekday: Full name of the current day (e.g., Monday)
    """
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
    }

TOOLS = [
    get_current_datetime,
]

# system_prompt="""You are an AI assistant with access to a set of tools.
# When a user asks a question, determine if a tool should be called to help answer.
# If a tool is needed, respond with a tool call using json schema.
# If no tool is needed, answer the user directly.
# Always use the most relevant tool(s) for the user request.
# If a tool returns an error, explain the error to the user.
# Be concise and helpful."""

system_prompts = {
    'none': None,
    
    'naive': """You are an AI assistant with access to a set of tools.
When a user asks a question, determine if a tool should be called to help answer.""",

    'use-json-schema': """You are an AI assistant with access to a set of tools.
When a user asks a question, determine if a tool should be called to help answer.
If a tool is needed, respond with a tool call using json schema.
If no tool is needed, answer the user directly.
Always use the most relevant tool(s) for the user request.
If a tool returns an error, explain the error to the user.
Be concise and helpful. Format in json schema""",

}

def main():
    parser = argparse.ArgumentParser(description="Run an AI agent with tool support")
    parser.add_argument(
        "--system-prompt",
        choices=list(system_prompts.keys()),
        default="none",
        help=f"System prompt to use. Available: {list(system_prompts.keys())}",
    )
    
    parser.add_argument(
        "--model",
        choices=MODELS,
        default=DEFAULT_MODEL,
        help=f"Model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show llama-server output",
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        default="What's the time right now?",
        help="The prompt to send to the agent",
    )
    args = parser.parse_args()

    console = Console()
    console.print(Panel(
        str(system_prompts[args.system_prompt]),
        title="System Prompt",
        border_style="blue",
    ))
    console.print(Panel(
        args.user_prompt,
        title="User Prompt",
        border_style="green",
    ))

    print(f"Starting llama-server with model: {args.model}")
    server_process = start_llama_server(args.model, verbose=args.verbose)
    try:
        print("Waiting for server to be ready...")
        wait_for_server("http://127.0.0.1:8080/health")
        print("Server ready!")

        agent = Agent(
            model=LlamaCpp(
                base_url="http://127.0.0.1:8080/v1",
                temperature=0,
            ),
            tools=TOOLS,
            system_message=system_prompts[args.system_prompt],
        )

        agent.print_response(args.user_prompt)
    finally:
        print("Stopping server...")
        stop_server(server_process)


if __name__ == "__main__":
    main()