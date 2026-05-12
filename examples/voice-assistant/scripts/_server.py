"""Boot and tear down `llama-liquid-audio-server` for an LFM2.5-Audio GGUF.

Shared by `eval.py` (Step 4) and `demo.py` (Step 5). Hides the platform-runner
download, GGUF resolution, subprocess start, and health polling behind two
functions: `boot_model_server` and `stop_model_server`.
"""

from __future__ import annotations

import platform
import stat
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

PLATFORM_MAP: dict[tuple[str, str], str] = {
    ("darwin", "arm64"): "macos-arm64",
    ("linux", "x86_64"): "ubuntu-x64",
    ("linux", "aarch64"): "ubuntu-arm64",
}


@dataclass
class ServerHandle:
    """Bundle of subprocess + resolved GGUF paths a caller may want."""

    process: subprocess.Popen[bytes]
    port: int
    binary: Path
    model: Path
    mmproj: Path
    vocoder: Path
    tokenizer: Path


def detect_platform() -> str:
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    key = (sys_name, machine)
    if key not in PLATFORM_MAP:
        raise RuntimeError(
            f"Unsupported platform: {sys_name}/{machine}. "
            f"Supported: {sorted(PLATFORM_MAP.values())}"
        )
    return PLATFORM_MAP[key]


def _extract_runner_zip(snapshot_dir: Path, plat: str) -> Path:
    """Unzip the platform runner if not already extracted; return path to the binary."""
    runner_dir = snapshot_dir / "runners" / plat
    binary_candidates = list(runner_dir.rglob("llama-liquid-audio-server"))
    if binary_candidates:
        binary = binary_candidates[0]
    else:
        runner_zip = snapshot_dir / "runners" / f"llama-liquid-audio-{plat}.zip"
        if not runner_zip.exists():
            raise FileNotFoundError(f"Runner zip not found: {runner_zip}")
        runner_dir.mkdir(parents=True, exist_ok=True)
        print(f"  unzipping {runner_zip.name}", flush=True)
        with zipfile.ZipFile(runner_zip) as zf:
            zf.extractall(runner_dir)
        binary_candidates = list(runner_dir.rglob("llama-liquid-audio-server"))
        if not binary_candidates:
            raise FileNotFoundError(
                f"llama-liquid-audio-server binary not found inside {runner_zip}"
            )
        binary = binary_candidates[0]

    for f in binary.parent.iterdir():
        if f.is_file():
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return binary


def download_artifacts(model_repo: str, quant: str) -> tuple[Path, Path, Path, Path, Path]:
    """Snapshot-download the GGUF repo, unzip the runner, locate the four GGUFs.

    Pulls only the files needed for this run (the four GGUFs at the chosen
    quant plus the runner zip for the current platform) to keep first-run
    footprint near 3 GB instead of 15 GB.

    Returns: (server_binary, model_gguf, mmproj_gguf, vocoder_gguf, tokenizer_gguf).
    """
    plat = detect_platform()
    model_stem = model_repo.split("/")[-1].removesuffix("-GGUF")
    allow_patterns = [
        f"{model_stem}-{quant}.gguf",
        f"mmproj-{model_stem}-{quant}.gguf",
        f"vocoder-{model_stem}-{quant}.gguf",
        f"tokenizer-{model_stem}-{quant}.gguf",
        f"runners/llama-liquid-audio-{plat}.zip",
    ]
    print(f"Downloading {model_repo} (quant={quant}, platform={plat}) ...", flush=True)
    snapshot_dir = Path(snapshot_download(repo_id=model_repo, allow_patterns=allow_patterns))
    binary = _extract_runner_zip(snapshot_dir, plat)

    model_path = snapshot_dir / f"{model_stem}-{quant}.gguf"
    mmproj_path = snapshot_dir / f"mmproj-{model_stem}-{quant}.gguf"
    vocoder_path = snapshot_dir / f"vocoder-{model_stem}-{quant}.gguf"
    tokenizer_path = snapshot_dir / f"tokenizer-{model_stem}-{quant}.gguf"

    for label, p in [
        ("model", model_path),
        ("mmproj", mmproj_path),
        ("vocoder", vocoder_path),
        ("tokenizer", tokenizer_path),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} gguf not found: {p}")

    return binary, model_path, mmproj_path, vocoder_path, tokenizer_path


def _start_subprocess(
    binary: Path,
    model: Path,
    mmproj: Path,
    vocoder: Path,
    tokenizer: Path,
    port: int,
    verbose: bool,
) -> subprocess.Popen[bytes]:
    cmd = [
        str(binary),
        "-m", str(model),
        "-mm", str(mmproj),
        "-mv", str(vocoder),
        "--tts-speaker-file", str(tokenizer),
        "--port", str(port),
    ]
    print(f"Starting server on :{port}", flush=True)
    kwargs: dict[str, Any] = {}
    if not verbose:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return subprocess.Popen(cmd, **kwargs)


def _wait_for_health(port: int, timeout: int = 180) -> None:
    """Wait until the server answers HTTP on `port`.

    `llama-liquid-audio-server` does not expose a dedicated `/health` route;
    any HTTP response (even a 404) means the bind is live and the model is
    loaded enough to dispatch. URLError still indicates the listener has not
    come up yet.
    """
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                _ = resp.status
                return
        except urllib.error.HTTPError:
            return
        except urllib.error.URLError:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy within {timeout}s")


def boot_model_server(
    model_repo: str,
    quant: str,
    port: int,
    verbose: bool = False,
) -> ServerHandle:
    """Download artifacts, start the server subprocess, wait until healthy.

    Caller is responsible for stopping the returned handle (typically in a
    try/finally).
    """
    binary, model, mmproj, vocoder, tokenizer = download_artifacts(model_repo, quant)
    proc = _start_subprocess(binary, model, mmproj, vocoder, tokenizer, port, verbose)
    _wait_for_health(port)
    print(f"  server ready on :{port}\n", flush=True)
    return ServerHandle(
        process=proc,
        port=port,
        binary=binary,
        model=model,
        mmproj=mmproj,
        vocoder=vocoder,
        tokenizer=tokenizer,
    )


def stop_model_server(handle: ServerHandle) -> None:
    p = handle.process
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
