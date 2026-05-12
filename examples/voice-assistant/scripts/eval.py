"""Evaluate an LFM2.5-Audio-1.5B GGUF on OHF-Voice function-calling.

Reads a YAML config that specifies model_repo, quant, system_prompt, and a few
inference knobs. Downloads the GGUFs + platform-specific llama-liquid-audio-server
binary, starts the server, evaluates a stratified subset of the published test
split, and writes a markdown report + a JSON dump of per-sample results.

Configs that ship in this repo:
- configs/baseline.yaml: upstream LiquidAI/LFM2.5-Audio-1.5B-GGUF, system_prompt
  "Perform ASR.", establishes the floor (0/0/0 on all three metrics).
- configs/finetuned-q8.yaml: fine-tuned fork, Q8_0 quant, system_prompt
  "Perform ASR." (matches the training-time chat shape). The default used
  in the README snippets.
- configs/finetuned-f16.yaml, configs/finetuned-q4.yaml: same fine-tune at
  other quants for sweep comparisons.

Usage:
    uv run python scripts/eval.py --config configs/baseline.yaml
    uv run python scripts/eval.py --config configs/finetuned-q8.yaml
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import platform
import stat
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from openai import OpenAI

load_dotenv()

DATASET_REPO = "Paulescu/OHF-Voice-audio-20260504"
TEST_SPLIT = "test"

# (system, machine) -> platform key used in runners/llama-liquid-audio-{key}.zip
PLATFORM_MAP: dict[tuple[str, str], str] = {
    ("darwin", "arm64"): "macos-arm64",
    ("linux", "x86_64"): "ubuntu-x64",
    ("linux", "aarch64"): "ubuntu-arm64",
}


@dataclass
class EvalConfig:
    name: str
    model_repo: str
    quant: str
    system_prompt: str | None
    samples_per_function: int = 10
    max_new_tokens: int = 64
    temperature: float = 0.0
    port: int = 8080

    @classmethod
    def from_yaml(cls, path: Path) -> EvalConfig:
        with path.open() as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls(**data)


@dataclass
class TestSample:
    sample_id: str
    function_name: str
    ground_truth: str
    audio_bytes: bytes


@dataclass
class SampleResult:
    sample_id: str
    function_name: str
    ground_truth: str
    prediction: str
    parseable: bool
    function_match: bool
    args_match: bool


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
    """Unzip the platform runner if not already extracted; return path to the server binary."""
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

    # Make sure binary and sibling shared libs are executable.
    for f in binary.parent.iterdir():
        if f.is_file():
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return binary


def download_artifacts(model_repo: str, quant: str) -> tuple[Path, Path, Path, Path, Path]:
    """Snapshot-download the GGUF repo, unzip the runner, and locate the four GGUFs.

    Pulls only the files needed for this run (the four GGUFs at the chosen quant
    plus the runner zip for the current platform), not every quant and runner in
    the repo, to keep the first-run footprint near 3 GB instead of 15 GB.

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


def start_server(
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


def wait_for_server(port: int, timeout: int = 180) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy within {timeout}s")


def stop_server(p: subprocess.Popen[bytes]) -> None:
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def get_audio_bytes(audio_field: Any) -> bytes:
    """Extract WAV bytes from the audio field of a dataset row.

    The HF datasets library returns raw bytes when the column is typed as a plain
    binary, but returns a decoded {'array': ndarray, 'sampling_rate': int} dict
    (or {'bytes': ..., 'path': ...}) when the column is typed as `Audio`. Handle
    all three shapes so the eval script works regardless of how the dataset is
    declared in HF.
    """
    if isinstance(audio_field, (bytes, bytearray)):
        return bytes(audio_field)
    if isinstance(audio_field, dict):
        if audio_field.get("bytes") is not None:
            return bytes(audio_field["bytes"])
        if "array" in audio_field and "sampling_rate" in audio_field:
            buf = io.BytesIO()
            sf.write(buf, audio_field["array"], audio_field["sampling_rate"], format="WAV")
            return buf.getvalue()
    raise TypeError(f"Unrecognised audio field: {type(audio_field).__name__}")


def load_stratified_test_samples(samples_per_function: int) -> list[TestSample]:
    print(f"Loading {DATASET_REPO} (split={TEST_SPLIT}) ...", flush=True)
    ds = load_dataset(DATASET_REPO, split=TEST_SPLIT)
    print(f"  {len(ds):,} test samples available", flush=True)

    by_function: dict[str, list[TestSample]] = defaultdict(list)
    for row in ds:
        fc = row["text_chat"][1]["content"][0]["text"]
        fn = fc.split("|")[0]
        if len(by_function[fn]) >= samples_per_function:
            continue
        audio = get_audio_bytes(row["audio_chat"][0]["content"][0]["audio"])
        by_function[fn].append(
            TestSample(
                sample_id=str(row["id"]),
                function_name=fn,
                ground_truth=fc,
                audio_bytes=audio,
            )
        )

    samples: list[TestSample] = []
    for fn in sorted(by_function):
        samples.extend(by_function[fn])
    print(
        f"  stratified subset: {len(samples)} samples across "
        f"{len(by_function)} functions (up to {samples_per_function}/function)",
        flush=True,
    )
    return samples


def audio_to_b64(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("ascii")


def predict(
    client: OpenAI,
    model_name: str,
    audio_bytes: bytes,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
) -> str:
    """Request a completion from llama-liquid-audio-server and return the joined
    text. The server requires `stream=True` (non-streaming returns HTTP 400), so
    we collect the streamed delta chunks and concatenate.
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_to_b64(audio_bytes), "format": "wav"},
                }
            ],
        }
    )
    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    chunks: list[str] = []
    for event in stream:
        if event.choices and event.choices[0].delta and event.choices[0].delta.content:
            chunks.append(event.choices[0].delta.content)
    return "".join(chunks).strip()


def parse_call(text: str) -> tuple[str, dict[str, str]] | None:
    """Parse `FuncName|$arg=val|...` into (function_name, args_dict).

    Returns None if the string isn't a well-formed function call. A bare
    function name with no args (e.g. `HassRespond`) is valid and returns
    (name, {}). The function name must be a valid identifier (letters, digits,
    underscores) so that error strings or other non-conforming text don't
    accidentally count as parseable.
    """
    text = text.strip()
    if not text:
        return None
    parts = text.split("|")
    fn = parts[0].strip()
    if not fn or not fn.replace("_", "").isalnum():
        return None
    args: dict[str, str] = {}
    for p in parts[1:]:
        if "=" not in p:
            return None
        k, v = p.split("=", 1)
        k = k.strip()
        if not k.startswith("$") or not k[1:].replace("_", "").isalnum():
            return None
        args[k] = v.strip()
    return fn, args


def score(prediction: str, ground_truth: str) -> tuple[bool, bool, bool]:
    """Three layered metrics: (parseable, function_name_match, args_match)."""
    gt = parse_call(ground_truth)
    if gt is None:
        raise ValueError(f"unparseable ground truth: {ground_truth!r}")
    pred = parse_call(prediction)
    if pred is None:
        return False, False, False
    fn_match = pred[0] == gt[0]
    args_match = fn_match and pred[1] == gt[1]
    return True, fn_match, args_match


def render_report(config: EvalConfig, results: list[SampleResult]) -> str:
    n = len(results)
    parseable_n = sum(1 for r in results if r.parseable)
    fn_match_n = sum(1 for r in results if r.function_match)
    args_match_n = sum(1 for r in results if r.args_match)

    def pct(x: int) -> str:
        return f"{100 * x / n:.1f}%" if n else "n/a"

    by_fn: dict[str, list[SampleResult]] = defaultdict(list)
    for r in results:
        by_fn[r.function_name].append(r)

    lines: list[str] = []
    lines.append(f"# Eval report: {config.name}")
    lines.append("")
    lines.append(f"- model: `{config.model_repo}` ({config.quant})")
    lines.append(f"- dataset: `{DATASET_REPO}` (test split)")
    lines.append(f"- samples: {n}")
    lines.append(f"- system_prompt: {'set' if config.system_prompt else 'none'}")
    lines.append("")
    lines.append("## Layered metrics")
    lines.append("")
    lines.append("| metric | count | pct |")
    lines.append("|---|---|---|")
    lines.append(f"| Format compliance (parseable) | {parseable_n}/{n} | {pct(parseable_n)} |")
    lines.append(f"| Function-name accuracy | {fn_match_n}/{n} | {pct(fn_match_n)} |")
    lines.append(f"| Argument accuracy | {args_match_n}/{n} | {pct(args_match_n)} |")
    lines.append("")
    lines.append("## Per-function breakdown")
    lines.append("")
    lines.append("| function | n | fmt | name | args |")
    lines.append("|---|---|---|---|---|")
    for fn in sorted(by_fn):
        items = by_fn[fn]
        m = len(items)
        p_n = sum(1 for r in items if r.parseable)
        f_n = sum(1 for r in items if r.function_match)
        a_n = sum(1 for r in items if r.args_match)
        lines.append(f"| {fn} | {m} | {p_n}/{m} | {f_n}/{m} | {a_n}/{m} |")
    return "\n".join(lines) + "\n"


def write_results(out_dir: Path, config: EvalConfig, results: list[SampleResult]) -> None:
    report = render_report(config, results)
    (out_dir / "report.md").write_text(report)

    payload = {
        "config": {
            "name": config.name,
            "model_repo": config.model_repo,
            "quant": config.quant,
            "system_prompt": config.system_prompt,
            "samples_per_function": config.samples_per_function,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
        },
        "summary": {
            "total": len(results),
            "parseable": sum(1 for r in results if r.parseable),
            "function_match": sum(1 for r in results if r.function_match),
            "args_match": sum(1 for r in results if r.args_match),
        },
        "samples": [
            {
                "id": r.sample_id,
                "function": r.function_name,
                "ground_truth": r.ground_truth,
                "prediction": r.prediction,
                "parseable": r.parseable,
                "function_match": r.function_match,
                "args_match": r.args_match,
            }
            for r in results
        ],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    print()
    print(report)
    print(f"Report: {out_dir}/report.md")
    print(f"Results: {out_dir}/results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML eval config.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evals"),
        help="Where to write eval reports (default: evals/).",
    )
    parser.add_argument(
        "--verbose-server",
        action="store_true",
        help="Show server stdout/stderr (default: silent).",
    )
    args = parser.parse_args()

    cfg = EvalConfig.from_yaml(args.config)
    binary, model, mmproj, vocoder, tokenizer = download_artifacts(cfg.model_repo, cfg.quant)
    samples = load_stratified_test_samples(cfg.samples_per_function)

    server: subprocess.Popen[bytes] | None = None
    try:
        server = start_server(
            binary, model, mmproj, vocoder, tokenizer, cfg.port, args.verbose_server
        )
        wait_for_server(cfg.port)
        print(f"  server ready on :{cfg.port}\n", flush=True)

        client = OpenAI(base_url=f"http://127.0.0.1:{cfg.port}/v1", api_key="dummy")
        model_name = f"{cfg.model_repo}:{cfg.quant}"

        results: list[SampleResult] = []
        for i, sample in enumerate(samples, 1):
            try:
                pred = predict(
                    client,
                    model_name,
                    sample.audio_bytes,
                    cfg.system_prompt,
                    cfg.max_new_tokens,
                    cfg.temperature,
                )
            except Exception as exc:
                pred = f"<ERROR: {exc}>"

            parseable, fn_match, args_match = score(pred, sample.ground_truth)
            results.append(
                SampleResult(
                    sample_id=sample.sample_id,
                    function_name=sample.function_name,
                    ground_truth=sample.ground_truth,
                    prediction=pred,
                    parseable=parseable,
                    function_match=fn_match,
                    args_match=args_match,
                )
            )
            status = "args" if args_match else ("name" if fn_match else ("fmt" if parseable else "no"))
            print(f"[{i}/{len(samples)}] {status:>4} | gt: {sample.ground_truth[:60]}", flush=True)
            print(f"               pred: {pred[:60]}", flush=True)
    finally:
        if server is not None:
            stop_server(server)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"{cfg.name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_results(out_dir, cfg, results)


if __name__ == "__main__":
    main()
