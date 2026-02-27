import readline  # noqa: F401 — enables arrow-key history in input()
import sys

import click

from .agent import Agent
from .config import Config, load_config
from .llm import get_llm_client
from .tools import set_working_directory

BANNER = """
╔══════════════════════════════════════╗
║      Local Coding Assistant          ║
║  Type your request and press Enter.  ║
║  Ctrl+C or 'exit' to quit.           ║
╚══════════════════════════════════════╝
"""


def _model_name(config: Config) -> str:
    if config.backend == "anthropic":
        return config.anthropic_model
    return f"{config.llama_model} @ {config.llama_base_url}"


@click.command()
@click.option("--backend", default=None, help="LLM backend: anthropic or llama")
@click.option("--model", default=None, help="Model name override")
@click.option("--working-dir", default=None, help="Working directory for bash commands")
def main(backend: str | None, model: str | None, working_dir: str | None) -> None:
    config = load_config()

    # CLI flags override env vars
    if backend:
        config.backend = backend  # type: ignore[assignment]
    if model:
        if config.backend == "anthropic":
            config.anthropic_model = model
        else:
            config.llama_model = model
    if working_dir:
        config.working_directory = working_dir

    set_working_directory(config.working_directory)

    llm = get_llm_client(config)
    agent = Agent(llm=llm, config=config)

    print(BANNER)
    print(f"  Backend : {config.backend}")
    print(f"  Model   : {_model_name(config)}")
    print(f"  Work dir: {config.working_directory}\n")

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
