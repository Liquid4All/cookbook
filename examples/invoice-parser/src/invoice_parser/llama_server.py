"""Manage a llama-server subprocess for local LLM inference."""

import subprocess
import time
import urllib.error
import urllib.request

from loguru import logger

DEFAULT_PORT = 8080


def start_llama_server(
    model_id: str,
    port: int = DEFAULT_PORT,
    verbose: bool = False,
) -> subprocess.Popen:
    """Start llama-server as a subprocess.

    Args:
        model_id: HuggingFace model ID to load
        port: Port to run the server on
        verbose: If True, show server output; otherwise suppress it

    Returns:
        The subprocess handle
    """
    cmd = ["llama-server", "-hf", model_id, "--jinja", "--port", str(port)]
    if verbose:
        process = subprocess.Popen(cmd)
    else:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return process


def wait_for_server(port: int = DEFAULT_PORT, timeout: int = 120) -> None:
    """Poll the /health endpoint until the server is ready.

    Args:
        port: Port the server is running on
        timeout: Maximum seconds to wait

    Raises:
        TimeoutError: If server doesn't respond within timeout
    """
    url = f"http://127.0.0.1:{port}/health"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    logger.info("llama-server is ready")
                    return
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become healthy within {timeout}s")


def stop_server(process: subprocess.Popen) -> None:
    """Gracefully stop the llama-server subprocess.

    Args:
        process: The subprocess handle to terminate
    """
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    logger.info("llama-server stopped")
