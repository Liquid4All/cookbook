"""Streamlit frontend for the wildfire prevention system.

Run from the project root:
    uv run streamlit run app/app.py
"""

import base64
import time
from pathlib import Path

import pydeck as pdk
import streamlit as st

from wildfire_prevention.db import fetch_all, init_db

DB_PATH = Path(__file__).parent.parent / "wildfire.db"
DB_IMAGES_DIR = Path(__file__).parent.parent / "db_images"

_RISK_COLOR: dict[str, list[int]] = {
    "low":    [0, 180, 0, 200],
    "medium": [255, 165, 0, 200],
    "high":   [220, 0, 0, 200],
}
_DEFAULT_COLOR = [128, 128, 128, 200]


def _to_data_url(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = base64.standard_b64encode(p.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _tile_bounds(lon: float, lat: float, size_km: float) -> list[list[float]]:
    d = size_km / 111.0
    return [[lon - d, lat - d], [lon + d, lat + d]]


def _filter_rows(
    rows: list[dict[str, object]],
    risk_filter: list[str],
    model_filter: list[str],
    date_range: tuple[str, str] | tuple[object, ...],
) -> list[dict[str, object]]:
    start = str(date_range[0])
    end = str(date_range[1]) if len(date_range) > 1 else str(date_range[0])
    out = []
    for r in rows:
        if risk_filter and r.get("risk_level") not in risk_filter:
            continue
        if model_filter and r.get("model") not in model_filter:
            continue
        created = str(r.get("created_at", ""))[:10]
        if created < start or created > end:
            continue
        out.append(r)
    return out


def main() -> None:
    st.set_page_config(page_title="Wildfire Prevention", layout="wide")
    st.title("Wildfire Risk Map")

    conn = init_db(DB_PATH)
    all_rows = fetch_all(conn)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Filters")

        risk_opts = ["low", "medium", "high"]
        risk_filter = st.multiselect("Risk level", risk_opts, default=risk_opts)

        all_models = sorted({str(r["model"]) for r in all_rows if r.get("model")})
        model_filter = st.multiselect("Model", all_models, default=all_models)

        from datetime import date
        dates = sorted(str(r["created_at"])[:10] for r in all_rows if r.get("created_at"))
        min_date = date.fromisoformat(dates[0]) if dates else date(2020, 1, 1)
        max_date = date.fromisoformat(dates[-1]) if dates else date.today()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
        )

        st.divider()
        st.subheader("Auto-refresh")
        auto_refresh = st.toggle("Enabled", value=False)
        refresh_interval = st.number_input(
            "Interval (seconds)", min_value=5, max_value=300, value=30, step=5
        )
        if st.button("Refresh now"):
            st.rerun()

    # --- Main: map ---
    rows = _filter_rows(all_rows, risk_filter, model_filter, date_range)

    if not rows:
        st.info(
            "No predictions in the DB yet. "
            "Start `predict.py` to begin collecting live predictions, "
            "or run `backfill.py` to seed with historical data."
        )
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        return

    bitmap_data = []
    scatter_data = []

    for r in rows:
        lon = float(r["lon"])  # type: ignore[arg-type]
        lat = float(r["lat"])  # type: ignore[arg-type]
        size_km = float(r.get("size_km") or 5.0)  # type: ignore[arg-type]
        risk = str(r.get("risk_level") or "")
        color = _RISK_COLOR.get(risk, _DEFAULT_COLOR)

        data_url = _to_data_url(str(r.get("rgb_path") or ""))
        if data_url:
            bitmap_data.append({
                "image": data_url,
                "bounds": _tile_bounds(lon, lat, size_km),
                "row_id": r["id"],
            })

        scatter_data.append({
            "lon": lon,
            "lat": lat,
            "color": color,
            "risk": risk,
            "row_id": r["id"],
            "timestamp": str(r.get("timestamp", "")),
            "model": str(r.get("model", "")),
        })

    bitmap_layer = pdk.Layer(
        "BitmapLayer",
        data=bitmap_data,
        get_image="image",
        bounds="bounds",
        opacity=0.8,
        pickable=True,
    )
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=scatter_data,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=3000,
        pickable=True,
    )

    view_state = pdk.ViewState(
        longitude=float(scatter_data[0]["lon"]),
        latitude=float(scatter_data[0]["lat"]),
        zoom=4,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[bitmap_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip={"text": "Risk: {risk}\n{timestamp}\n{model}"},
        map_style="mapbox://styles/mapbox/satellite-streets-v12",
    )

    selected = st.pydeck_chart(deck, use_container_width=True)

    # --- Detail panel: show most recent row by default ---
    st.subheader("Tile detail")
    detail_row = rows[0]

    col1, col2 = st.columns(2)
    with col1:
        rgb_url = _to_data_url(str(detail_row.get("rgb_path") or ""))
        if rgb_url:
            st.image(rgb_url, caption="RGB", use_container_width=True)
        else:
            st.write("No RGB image.")
    with col2:
        swir_url = _to_data_url(str(detail_row.get("swir_path") or ""))
        if swir_url:
            st.image(swir_url, caption="SWIR", use_container_width=True)
        else:
            st.write("No SWIR image.")

    display_fields = [
        "id", "lon", "lat", "timestamp", "risk_level",
        "dry_vegetation_present", "urban_interface", "steep_terrain",
        "water_body_present", "image_quality_limited", "model", "source", "created_at",
    ]
    st.json({k: detail_row.get(k) for k in display_fields})

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
