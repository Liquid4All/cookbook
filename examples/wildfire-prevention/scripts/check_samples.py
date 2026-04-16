"""Validate generated samples for a given run and print a summary.

Usage:
    uv run scripts/check_samples.py 20260416_143052
    uv run scripts/check_samples.py  # uses the most recent run
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildfire_prevention.locations import LOCATIONS

DATA_DIR = Path(__file__).parent.parent / "data"
RISK_LEVELS = ["low", "medium", "high"]


def resolve_run_dir(run_arg: str | None) -> Path:
    if run_arg:
        run_dir = DATA_DIR / run_arg
        if not run_dir.is_dir():
            print(f"Run not found: {run_dir}")
            sys.exit(1)
        return run_dir
    # Only consider timestamped run dirs (YYYYMMDD_HHMMSS).
    runs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and len(p.name) == 15 and p.name[8] == "_")
    if not runs:
        print(f"No timestamped runs found in {DATA_DIR}")
        sys.exit(1)
    return runs[-1]


def main() -> None:
    run_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir = resolve_run_dir(run_arg)
    print(f"Checking run: {run_dir.name}\n")

    rows: list[tuple[str, str, str, str]] = []
    errors: list[str] = []
    model_counts: dict[str, int] = {r: 0 for r in RISK_LEVELS}

    for loc in LOCATIONS:
        sample_dir = run_dir / loc.id
        annotation_path = sample_dir / "annotation.json"
        rgb_path = sample_dir / "rgb.png"
        swir_path = sample_dir / "swir.png"

        missing = [p.name for p in (rgb_path, swir_path, annotation_path) if not p.exists()]
        if missing:
            errors.append(f"{loc.id}: missing {', '.join(missing)}")
            rows.append((loc.id, loc.expected_risk, "—", "—"))
            continue

        try:
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{loc.id}: annotation.json is invalid JSON — {exc}")
            rows.append((loc.id, loc.expected_risk, "parse error", "✗"))
            continue

        model_risk = data.get("risk_level", "?")
        match = "✓" if model_risk == loc.expected_risk else "✗"
        if model_risk in model_counts:
            model_counts[model_risk] += 1
        rows.append((loc.id, loc.expected_risk, model_risk, match))

    col_w = max(len(r[0]) for r in rows) + 2
    header = f"{'id':<{col_w}} {'expected':<10} {'model':<10} {'match'}"
    print(header)
    print("-" * len(header))
    for loc_id, expected, model, match in rows:
        print(f"{loc_id:<{col_w}} {expected:<10} {model:<10} {match}")

    print()
    print("Samples per risk level (model output):")
    for level in RISK_LEVELS:
        print(f"  {level:<10} {model_counts[level]}")

    if errors:
        print()
        print("Errors:")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
