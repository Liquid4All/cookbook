import readline  # noqa: F401 — enables arrow-key history in input()
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse

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
    return f"{config.local_model} @ {config.local_base_url}"


def _start_local_server(config: Config) -> subprocess.Popen:
    """Start llama-server for config.local_model. Returns the process."""
    port = urlparse(config.local_base_url).port or 8080
    model = config.local_model

    cmd = [
        "llama-server",
        "--port", str(port),
        "--ctx-size", str(config.local_ctx_size),
        "--n-gpu-layers", str(config.local_n_gpu_layers),
        "--flash-attn", "on",
    ]

    # Local file path vs HuggingFace repo path
    if model.startswith("/") or model.startswith("./"):
        cmd += ["--model", model]
    else:
        import os
        cmd += ["-hf", model]
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            cmd += ["-hft", hf_token]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Poll /health until ready (up to 3 minutes for large model downloads/loads)
    health_url = f"http://localhost:{port}/health"
    print(f"  Starting llama-server for {model} ...", flush=True)
    for _ in range(180):
        try:
            with urllib.request.urlopen(health_url, timeout=1) as resp:
                if resp.status == 200:
                    print("  llama-server ready.\n", flush=True)
                    return proc
        except Exception:
            pass
        time.sleep(1)

    proc.kill()
    raise RuntimeError("llama-server did not become ready within 3 minutes")


@click.command()
@click.option("--backend", default=None, help="LLM backend: anthropic or local")
@click.option("--model", default=None, help="Model name (Anthropic model ID, or HF path/GGUF for local)")
@click.option("--working-dir", default=None, help="Working directory for bash commands")
@click.option("-p", "--prompt", default=None, help="Run a single prompt non-interactively and exit")
def main(backend: str | None, model: str | None, working_dir: str | None, prompt: str | None) -> None:
    config = load_config()

    # CLI flags override env vars
    if backend:
        config.backend = backend  # type: ignore[assignment]
    if model:
        if config.backend == "anthropic":
            config.anthropic_model = model
        else:
            config.local_model = model
    if working_dir:
        config.working_directory = working_dir

    set_working_directory(config.working_directory)

    server_proc: subprocess.Popen | None = None
    if config.backend == "local" and model:
        server_proc = _start_local_server(config)

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

    finally:
        if server_proc is not None:
            server_proc.terminate()
            server_proc.wait()
