"""Evaluation comparison UI for wildfire risk models.

Run from the project root:
    uv run streamlit run app/eval_compare.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import streamlit as st

EVALS_DIR = Path(__file__).parent.parent / "evals"
DATA_DIR = Path(__file__).parent.parent / "data"
RISK_LEVELS = ["low", "medium", "high"]
EVAL_FIELDS = [
    "risk_level",
    "dry_vegetation_present",
    "urban_interface",
    "steep_terrain",
    "water_body_present",
    "image_quality_limited",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_eval_run(run_id: str) -> tuple[dict[str, object], list[dict[str, object]]] | None:
    """Return (meta, results) for a run, or None if no results.json exists."""
    run_dir = EVALS_DIR / run_id
    results_path = run_dir / "results.json"
    meta_path = run_dir / "meta.json"
    if not results_path.exists():
        return None
    results: list[dict[str, object]] = json.loads(results_path.read_text(encoding="utf-8"))
    meta: dict[str, object] = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.exists()
        else {"eval_run_id": run_id, "model": "unknown", "dataset": "unknown"}
    )
    return meta, results


def list_runs() -> list[str]:
    if not EVALS_DIR.exists():
        return []
    return sorted(
        (d.name for d in EVALS_DIR.iterdir() if d.is_dir()),
        reverse=True,
    )


def run_label(run_id: str, meta: dict[str, object]) -> str:
    model = str(meta.get("model", "?"))
    dataset = str(meta.get("dataset", "?"))
    short_model = model.split("/")[-1] if "/" in model else model
    return f"{run_id} | {short_model} | {dataset}"


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_summary(results: list[dict[str, object]]) -> dict[str, float]:
    n = len(results)
    if n == 0:
        return {}
    valid = sum(1 for r in results if r["valid_json"]) / n
    present = sum(1 for r in results if r["fields_present"]) / n
    field_accs: dict[str, float] = {}
    for field in EVAL_FIELDS:
        matches = [
            r["field_matches"][field]  # type: ignore[index]
            for r in results
            if r["fields_present"] and field in r["field_matches"]  # type: ignore[operator]
        ]
        field_accs[field] = sum(matches) / len(matches) if matches else 0.0
    all_matches = [
        v
        for r in results
        if r["fields_present"]
        for v in r["field_matches"].values()  # type: ignore[union-attr]
    ]
    overall = sum(all_matches) / len(all_matches) if all_matches else 0.0
    avg_lat = sum(float(r["latency_s"]) for r in results) / n  # type: ignore[arg-type]
    return {"valid_json": valid, "fields_present": present, **field_accs, "overall": overall, "avg_latency_s": avg_lat}


def confusion_matrix(results: list[dict[str, object]]) -> pd.DataFrame:
    """3x3 confusion matrix for risk_level (rows=true, cols=predicted)."""
    counts: dict[tuple[str, str], int] = {}
    for r in results:
        if not r.get("fields_present") or not r.get("ground_truth") or not r.get("prediction"):
            continue
        true_val = str(r["ground_truth"]["risk_level"])  # type: ignore[index]
        pred_val = str(r["prediction"]["risk_level"])  # type: ignore[index]
        counts[(true_val, pred_val)] = counts.get((true_val, pred_val), 0) + 1

    data = {pred: [counts.get((true, pred), 0) for true in RISK_LEVELS] for pred in RISK_LEVELS}
    df = pd.DataFrame(data, index=RISK_LEVELS)
    df.index.name = "true \\ predicted"
    return df


def risk_distribution(results: list[dict[str, object]]) -> pd.DataFrame:
    """Side-by-side count of true vs predicted risk levels."""
    true_counts: dict[str, int] = {lv: 0 for lv in RISK_LEVELS}
    pred_counts: dict[str, int] = {lv: 0 for lv in RISK_LEVELS}
    for r in results:
        if r.get("ground_truth"):
            lv = str(r["ground_truth"]["risk_level"])  # type: ignore[index]
            true_counts[lv] = true_counts.get(lv, 0) + 1
        if r.get("prediction") and r.get("fields_present"):
            lv = str(r["prediction"]["risk_level"])  # type: ignore[index]
            pred_counts[lv] = pred_counts.get(lv, 0) + 1
    return pd.DataFrame({"true": true_counts, "predicted": pred_counts}, index=RISK_LEVELS)


def wrong_risk_level_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for r in results:
        if not r.get("fields_present") or not r.get("ground_truth") or not r.get("prediction"):
            continue
        true_val = r["ground_truth"]["risk_level"]  # type: ignore[index]
        pred_val = r["prediction"]["risk_level"]  # type: ignore[index]
        if true_val != pred_val:
            rows.append({"id": r["id"], "true": true_val, "predicted": pred_val})
    return rows


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def find_images(sample_id: str) -> tuple[Path | None, Path | None]:
    """Search data/* directories for rgb.png and swir.png for a sample_id.

    sample_id format: '{location}/{tile}' e.g. 'angeles_nf_ca/s00_t10'
    Images can be in data/{run}/{split}/{location}/{tile}/ or data/{run}/images/.
    """
    if not DATA_DIR.exists():
        return None, None
    loc, tile = sample_id.split("/", 1) if "/" in sample_id else (sample_id, "")

    for run_dir in sorted(DATA_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        for split in ("test", "train"):
            tile_dir = run_dir / split / loc / tile
            rgb = tile_dir / "rgb.png"
            swir = tile_dir / "swir.png"
            if rgb.exists() and swir.exists():
                return rgb, swir
    return None, None


def img_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.standard_b64encode(data).decode()
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def render_summary_tab(
    selected_run_ids: list[str],
    loaded: dict[str, tuple[dict[str, object], list[dict[str, object]]]],
) -> None:
    if not selected_run_ids:
        st.info("Select one or more eval runs in the sidebar.")
        return

    rows = []
    for run_id in selected_run_ids:
        meta, results = loaded[run_id]
        summary = compute_summary(results)
        row: dict[str, object] = {
            "run": run_id,
            "model": str(meta.get("model", "?")),
            "dataset": str(meta.get("dataset", "?")),
            "n": len(results),
        }
        row.update(summary)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("run")
    pct_cols = ["valid_json", "fields_present"] + EVAL_FIELDS + ["overall"]
    df[pct_cols] = (df[pct_cols] * 100).round(1)
    df = df.rename(columns={c: f"{c} %" for c in pct_cols})

    st.subheader("Accuracy summary across runs")
    st.dataframe(df, use_container_width=True)


def render_risk_analysis_tab(
    selected_run_ids: list[str],
    loaded: dict[str, tuple[dict[str, object], list[dict[str, object]]]],
) -> None:
    if not selected_run_ids:
        st.info("Select one or more eval runs in the sidebar.")
        return

    for run_id in selected_run_ids:
        meta, results = loaded[run_id]
        model = str(meta.get("model", "?")).split("/")[-1]
        st.markdown(f"### {run_id} | {model}")

        col_dist, col_cm = st.columns(2)

        with col_dist:
            st.markdown("**Risk level distribution: true vs predicted**")
            dist_df = risk_distribution(results)
            st.bar_chart(dist_df)
            st.caption(
                "If 'predicted' is concentrated on one level while 'true' is spread out, "
                "the model is class-collapsing."
            )

        with col_cm:
            st.markdown("**Confusion matrix (risk_level)**")
            cm_df = confusion_matrix(results)
            total = cm_df.values.sum() or 1

            def _shade(val: int) -> str:  # type: ignore[override]
                intensity = int(val / total * 255)
                return f"background-color: rgba(100, 149, 237, {intensity / 255:.2f})"

            styled = cm_df.style.map(_shade)  # type: ignore[call-overload]
            st.dataframe(styled, use_container_width=True)
            st.caption("Rows: true label. Columns: predicted label.")

        wrong = wrong_risk_level_rows(results)
        if wrong:
            st.markdown(f"**Samples with wrong risk_level ({len(wrong)} / {len(results)})**")
            wrong_df = pd.DataFrame(wrong)
            st.dataframe(wrong_df, use_container_width=True)
        else:
            st.success("All risk_level predictions correct.")

        st.divider()


def render_sample_explorer_tab(
    selected_run_ids: list[str],
    loaded: dict[str, tuple[dict[str, object], list[dict[str, object]]]],
) -> None:
    if not selected_run_ids:
        st.info("Select one or more eval runs in the sidebar.")
        return

    # Build a flat dataframe with one row per (run, sample)
    rows = []
    for run_id in selected_run_ids:
        meta, results = loaded[run_id]
        model = str(meta.get("model", "?")).split("/")[-1]
        for r in results:
            row: dict[str, object] = {
                "run": run_id,
                "model": model,
                "id": r["id"],
                "valid_json": r["valid_json"],
                "fields_present": r["fields_present"],
                "latency_s": round(float(r["latency_s"]), 2),  # type: ignore[arg-type]
            }
            fm: dict[str, bool] = r.get("field_matches") or {}  # type: ignore[assignment]
            for field in EVAL_FIELDS:
                row[field] = "✓" if fm.get(field) else "✗"
            gt: dict[str, object] = r.get("ground_truth") or {}  # type: ignore[assignment]
            pred: dict[str, object] = r.get("prediction") or {}  # type: ignore[assignment]
            row["gt_risk"] = gt.get("risk_level", "?")
            row["pred_risk"] = pred.get("risk_level", "?")
            rows.append(row)

    df = pd.DataFrame(rows)

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_location = st.text_input("Filter by location", "")
    with col_f2:
        filter_risk = st.selectbox("Filter by true risk level", ["all"] + RISK_LEVELS)
    with col_f3:
        filter_wrong_only = st.checkbox("Show wrong risk_level only", value=False)

    filtered = df.copy()
    if filter_location:
        filtered = filtered[filtered["id"].str.contains(filter_location, case=False)]
    if filter_risk != "all":
        filtered = filtered[filtered["gt_risk"] == filter_risk]
    if filter_wrong_only:
        filtered = filtered[filtered["gt_risk"] != filtered["pred_risk"]]

    st.dataframe(filtered, use_container_width=True, height=400)

    # Detail view
    st.subheader("Sample detail")
    sample_ids = sorted(filtered["id"].unique().tolist())
    if not sample_ids:
        st.info("No samples match the current filter.")
        return

    selected_sample = st.selectbox("Select a sample to inspect", sample_ids)
    if not selected_sample:
        return

    rgb_path, swir_path = find_images(selected_sample)

    img_col, pred_col = st.columns([2, 1])

    with img_col:
        if rgb_path and swir_path:
            ic1, ic2 = st.columns(2)
            with ic1:
                st.image(str(rgb_path), caption="RGB", use_container_width=True)
            with ic2:
                st.image(str(swir_path), caption="SWIR", use_container_width=True)
        else:
            st.warning("Images not found. Run generate_samples.py to create local data.")

    with pred_col:
        for run_id in selected_run_ids:
            _, results = loaded[run_id]
            sample_results = [r for r in results if r["id"] == selected_sample]
            if not sample_results:
                continue
            r = sample_results[0]
            meta_run = loaded[run_id][0]
            model = str(meta_run.get("model", "?")).split("/")[-1]
            st.markdown(f"**{run_id} | {model}**")
            gt = r.get("ground_truth") or {}
            pred = r.get("prediction") or {}
            comparison_rows = []
            for field in EVAL_FIELDS:
                gt_val = gt.get(field, "?")
                pred_val = pred.get(field, "?")
                match = "✓" if gt_val == pred_val else "✗"
                comparison_rows.append({"field": field, "ground truth": str(gt_val), "predicted": str(pred_val), "match": match})
            st.dataframe(pd.DataFrame(comparison_rows).set_index("field"), use_container_width=True)
            st.caption(f"latency: {float(r['latency_s']):.2f}s")  # type: ignore[arg-type]
            st.divider()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Eval Comparison", layout="wide")
    st.title("Wildfire model evaluation comparison")

    all_runs = list_runs()
    if not all_runs:
        st.error(f"No eval runs found in {EVALS_DIR}. Run evaluate.py first.")
        return

    # Separate runs with results.json from report-only runs
    runs_with_results = []
    runs_report_only = []
    loaded: dict[str, tuple[dict[str, object], list[dict[str, object]]]] = {}
    for run_id in all_runs:
        data = load_eval_run(run_id)
        if data is not None:
            runs_with_results.append(run_id)
            loaded[run_id] = data
        else:
            runs_report_only.append(run_id)

    with st.sidebar:
        st.header("Select eval runs")
        if not runs_with_results:
            st.warning("No runs with results.json found. Re-run evaluate.py to generate structured data.")
        else:
            options = {run_label(rid, loaded[rid][0]): rid for rid in runs_with_results}
            selected_labels = st.multiselect(
                "Runs (newest first)",
                list(options.keys()),
                default=list(options.keys())[:2] if len(options) >= 2 else list(options.keys()),
            )
            selected_run_ids = [options[lbl] for lbl in selected_labels]

        if runs_report_only:
            with st.expander(f"Report-only runs ({len(runs_report_only)}, no results.json)"):
                for rid in runs_report_only:
                    st.write(rid)

    if not runs_with_results:
        return

    tab_summary, tab_risk, tab_samples = st.tabs(
        ["Summary", "Risk level analysis", "Sample explorer"]
    )

    with tab_summary:
        render_summary_tab(selected_run_ids, loaded)

    with tab_risk:
        render_risk_analysis_tab(selected_run_ids, loaded)

    with tab_samples:
        render_sample_explorer_tab(selected_run_ids, loaded)


if __name__ == "__main__":
    main()
