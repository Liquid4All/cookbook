import os
import subprocess
import time
import urllib.request
from urllib.parse import urlparse

from .config import Config


def start_local_server(config: Config) -> subprocess.Popen:
    """Start llama-server for config.local_model. Returns the process."""
    port = urlparse(config.local_base_url).port or 8080
    model = config.local_model

    cmd = [
        "llama-server",
        "--port", str(port),
        "--ctx-size", str(config.local_ctx_size),
        "--n-gpu-layers", str(config.local_n_gpu_layers),
        "--flash-attn", "on",
        "--jinja",
        # Recommended generation parameters for LFM2-24B-A2B
        "--temp", "0.1",
        "--top-k", "50",
        "--repeat-penalty", "1.05",
    ]

    if model.startswith("/") or model.startswith("./"):
        cmd += ["--model", model]
    else:
        cmd += ["-hf", model]
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            cmd += ["-hft", hf_token]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
