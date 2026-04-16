"""Evaluation backends and metrics for wildfire risk prediction."""

import base64
import json
import subprocess
import time
import urllib.error
import urllib.request
from time import perf_counter
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from wildfire_prevention.annotator import MODEL as ANTHROPIC_MODEL
from wildfire_prevention.annotator import SYSTEM_PROMPT, annotate

EVAL_FIELDS: list[str] = [
    "risk_level",
    "dry_vegetation_present",
    "urban_interface",
    "steep_terrain",
    "water_body_present",
    "image_quality_limited",
]

_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "risk_level":             {"type": "string", "enum": ["low", "medium", "high"]},
        "dry_vegetation_present": {"type": "boolean"},
        "urban_interface":        {"type": "boolean"},
        "steep_terrain":          {"type": "boolean"},
        "water_body_present":     {"type": "boolean"},
        "image_quality_limited":  {"type": "boolean"},
    },
    "required": EVAL_FIELDS,
}

USER_TEXT = (
    "Image 1 is the RGB composite. Image 2 is the SWIR composite. "
    "Return the wildfire risk JSON for this tile."
)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class PredictFn(Protocol):
    def __call__(self, rgb_bytes: bytes, swir_bytes: bytes) -> dict[str, object]: ...


def anthropic_backend() -> PredictFn:
    """Return a predict function that calls claude-opus-4-6 via the Anthropic SDK."""
    return annotate


def llama_backend(model: str, port: int = 8080) -> PredictFn:
    """Return a predict function that calls a local llama-server via the OpenAI API."""
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed")

    def predict(rgb_bytes: bytes, swir_bytes: bytes) -> dict[str, object]:
        def _data_url(image_bytes: bytes) -> str:
            return "data:image/png;base64," + base64.standard_b64encode(image_bytes).decode()

        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "WildfireRisk", "schema": _RESPONSE_SCHEMA},
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _data_url(rgb_bytes)}},
                        {"type": "image_url", "image_url": {"url": _data_url(swir_bytes)}},
                        {"type": "text", "text": USER_TEXT},
                    ],
                },
            ],
        )
        content = response.choices[0].message.content or ""
        return json.loads(content)  # type: ignore[no-any-return]

    return predict


# ---------------------------------------------------------------------------
# llama-server lifecycle (mirrors invoice-parser pattern)
# ---------------------------------------------------------------------------

def start_llama_server(
    model: str,
    quant: str | None = None,
    port: int = 8080,
    verbose: bool = False,
) -> subprocess.Popen[bytes]:
    hf_repo = f"{model}:{quant}" if quant else model
    cmd = ["llama-server", "-hf", hf_repo, "--jinja", "--port", str(port)]
    kwargs: dict[str, object] = {}
    if not verbose:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return subprocess.Popen(cmd, **kwargs)  # type: ignore[call-overload]


def wait_for_server(port: int = 8080, timeout: int = 120) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become healthy within {timeout}s")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


# ---------------------------------------------------------------------------
# Per-sample result
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    id: str
    valid_json: bool
    fields_present: bool
    field_matches: dict[str, bool]  # field -> match against ground truth
    latency_s: float = 0.0

    @property
    def all_fields_match(self) -> bool:
        return all(self.field_matches.values())


def evaluate_sample(
    location_id: str,
    rgb_bytes: bytes,
    swir_bytes: bytes,
    ground_truth: dict[str, object],
    predict: PredictFn,
) -> SampleResult:
    t0 = perf_counter()
    try:
        prediction = predict(rgb_bytes, swir_bytes)
        valid_json = True
    except Exception:
        return SampleResult(
            id=location_id,
            valid_json=False,
            fields_present=False,
            field_matches={f: False for f in EVAL_FIELDS},
            latency_s=perf_counter() - t0,
        )

    latency_s = perf_counter() - t0
    fields_present = all(f in prediction for f in EVAL_FIELDS)
    field_matches = {
        f: prediction.get(f) == ground_truth.get(f)
        for f in EVAL_FIELDS
    }
    return SampleResult(
        id=location_id,
        valid_json=valid_json,
        fields_present=fields_present,
        field_matches=field_matches,
        latency_s=latency_s,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalSummary:
    results: list[SampleResult]

    def valid_json_accuracy(self) -> float:
        return sum(r.valid_json for r in self.results) / len(self.results) if self.results else 0.0

    def fields_present_accuracy(self) -> float:
        return sum(r.fields_present for r in self.results) / len(self.results) if self.results else 0.0

    def field_accuracy(self, field: str) -> float:
        matches = [r.field_matches[field] for r in self.results if r.fields_present]
        return sum(matches) / len(matches) if matches else 0.0

    def overall_accuracy(self) -> float:
        all_matches = [
            match
            for r in self.results
            if r.fields_present
            for match in r.field_matches.values()
        ]
        return sum(all_matches) / len(all_matches) if all_matches else 0.0

    def avg_latency_s(self) -> float:
        return sum(r.latency_s for r in self.results) / len(self.results) if self.results else 0.0


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _tick(value: bool) -> str:
    return "✓" if value else "✗"


def render_report(
    summary: EvalSummary,
    dataset: str,
    backend: str,
    model: str,
    eval_run_id: str,
) -> str:
    lines: list[str] = []

    lines.append(f"# Wildfire Risk Eval — {eval_run_id}")
    lines.append("")
    lines.append(f"**Dataset:** {dataset}  ")
    lines.append(f"**Backend:** {backend}  ")
    lines.append(f"**Model:** {model}")
    lines.append("")

    # Accuracy summary first
    lines.append("## Accuracy summary")
    lines.append("")
    lines.append("| field | accuracy |")
    lines.append("|---|---|")
    lines.append(f"| valid_json | {summary.valid_json_accuracy():.2f} |")
    lines.append(f"| fields_present | {summary.fields_present_accuracy():.2f} |")
    for field in EVAL_FIELDS:
        acc = summary.field_accuracy(field)
        lines.append(f"| {field} | {acc:.2f} |")
    lines.append(f"| **overall** | **{summary.overall_accuracy():.2f}** |")
    lines.append(f"| **avg latency (s)** | **{summary.avg_latency_s():.2f}** |")
    lines.append("")

    # Per-sample table
    lines.append("## Per-sample results")
    lines.append("")
    header = (
        "| id | latency (s) | valid_json | fields_present"
        " | risk_level | dry_veg | urban | terrain | water | quality |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in summary.results:
        fm = r.field_matches
        lines.append(
            f"| {r.id}"
            f" | {r.latency_s:.2f}"
            f" | {_tick(r.valid_json)}"
            f" | {_tick(r.fields_present)}"
            f" | {_tick(fm.get('risk_level', False))}"
            f" | {_tick(fm.get('dry_vegetation_present', False))}"
            f" | {_tick(fm.get('urban_interface', False))}"
            f" | {_tick(fm.get('steep_terrain', False))}"
            f" | {_tick(fm.get('water_body_present', False))}"
            f" | {_tick(fm.get('image_quality_limited', False))}"
            " |"
        )
    lines.append("")

    return "\n".join(lines)


def model_name(backend: str, llama_model: str, quant: str = "") -> str:
    if backend == "anthropic":
        return ANTHROPIC_MODEL
    return f"{llama_model}:{quant}" if quant else llama_model
