import readline  # noqa: F401 — enables arrow-key history in input()
import subprocess
import sys

import click

from .agent import Agent
from .config import Config, load_config
from .llm import get_llm_client
from .server import start_local_server

BANNER = """
╔══════════════════════════════════════╗
║   Meeting Intelligence Agent         ║
║  Type your request and press Enter.  ║
║  Ctrl+C or 'exit' to quit.           ║
╚══════════════════════════════════════╝
"""


@click.command()
@click.option("--model", default=None, help="Model name or HuggingFace/GGUF path")
@click.option("-p", "--prompt", default=None, help="Run a single prompt non-interactively and exit")
def main(model: str | None, prompt: str | None) -> None:
    config = load_config()

    if model:
        config.local_model = model

    server_proc: subprocess.Popen | None = None
    if model:
        server_proc = start_local_server(config)

    try:
        llm = get_llm_client(config)
        agent = Agent(llm=llm, config=config)

        # Non-interactive mode: run one prompt and exit
        if prompt:
            try:
                agent.run_turn(prompt)
            except Exception as e:
                print(f"[error] {e}")
                sys.exit(1)
            return

        print(BANNER)
        print(f"  Model : {config.local_model} @ {config.local_base_url}\n")

        while True:
            try:
                user_input = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                sys.exit(0)

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye.")
                break

            try:
                agent.run_turn(user_input)
            except KeyboardInterrupt:
                print("\n[interrupted]")
                continue
            except Exception as e:
                print(f"[error] {e}")

    finally:
        if server_proc is not None:
            server_proc.terminate()
            server_proc.wait()
