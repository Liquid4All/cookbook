"""Validate generated samples for a given run and print a summary per split.

Usage:
    uv run scripts/check_samples.py 20260416_143052
    uv run scripts/check_samples.py  # uses the most recent run
"""

import json
import sys
from pathlib import Path

from wildfire_prevention.locations import LOCATIONS_BY_ID

DATA_DIR = Path(__file__).parent.parent / "data"
RISK_LEVELS = ["low", "medium", "high"]


def resolve_run_dir(run_arg: str | None) -> Path:
    if run_arg:
        run_dir = DATA_DIR / run_arg
        if not run_dir.is_dir():
            print(f"Run not found: {run_dir}")
            sys.exit(1)
        return run_dir
    runs = sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_dir() and len(p.name) == 15 and p.name[8] == "_"
    )
    if not runs:
        print(f"No timestamped runs found in {DATA_DIR}")
        sys.exit(1)
    return runs[-1]


def check_split(split_dir: Path) -> list[str]:
    """Check all tiles in a split directory. Returns a list of error strings."""
    loc_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if not loc_dirs:
        print(f"  (no locations found)")
        return []

    rows: list[tuple[str, str, str, str]] = []
    errors: list[str] = []
    model_counts: dict[str, int] = {r: 0 for r in RISK_LEVELS}
    total = 0

    for loc_dir in loc_dirs:
        loc_id = loc_dir.name
        expected = LOCATIONS_BY_ID[loc_id].expected_risk if loc_id in LOCATIONS_BY_ID else "?"
        tile_dirs = sorted(p for p in loc_dir.iterdir() if p.is_dir())

        for tile_dir in tile_dirs:
            total += 1
            tile_key = f"{loc_id}/{tile_dir.name}"
            rgb_path = tile_dir / "rgb.png"
            swir_path = tile_dir / "swir.png"
            annotation_path = tile_dir / "annotation.json"

            missing = [p.name for p in (rgb_path, swir_path, annotation_path) if not p.exists()]
            if missing:
                errors.append(f"{tile_key}: missing {', '.join(missing)}")
                rows.append((tile_key, expected, "—", "—"))
                continue

            try:
                data = json.loads(annotation_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"{tile_key}: annotation.json is invalid JSON — {exc}")
                rows.append((tile_key, expected, "parse error", "✗"))
                continue

            model_risk = data.get("risk_level", "?")
            match = "✓" if model_risk == expected else "✗"
            if isinstance(model_risk, str) and model_risk in model_counts:
                model_counts[model_risk] += 1
            rows.append((tile_key, expected, str(model_risk), match))

    missing_count = sum(1 for _, _, m, _ in rows if m == "—")
    print(f"  {total} tiles  ({missing_count} missing)\n")

    col_w = max((len(r[0]) for r in rows), default=10) + 2
    header = f"  {'tile':<{col_w}} {'expected':<10} {'model':<10} match"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tile_key, expected, model, match in rows:
        print(f"  {tile_key:<{col_w}} {expected:<10} {model:<10} {match}")

    print()
    print("  Samples per risk level (model output):")
    for level in RISK_LEVELS:
        print(f"    {level:<10} {model_counts[level]}")

    return errors


def main() -> None:
    run_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir = resolve_run_dir(run_arg)
    print(f"Checking run: {run_dir.name}\n")

    all_errors: list[str] = []

    for split in ("train", "test"):
        split_dir = run_dir / split
        if not split_dir.is_dir():
            continue
        print(f"[{split}]")
        errors = check_split(split_dir)
        all_errors.extend(errors)
        print()

    if all_errors:
        print("Errors:")
        for err in all_errors:
            print(f"  {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
