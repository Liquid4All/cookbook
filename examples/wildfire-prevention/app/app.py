"""Streamlit frontend for the wildfire prevention system.

Run from the project root:
    uv run streamlit run app/app.py
"""

import base64
import math
import time
from pathlib import Path

import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from wildfire_prevention.db import fetch_all, init_db
from wildfire_prevention.regions import REGIONS

DB_PATH = Path(__file__).parent.parent / "wildfire.db"
DB_IMAGES_DIR = Path(__file__).parent.parent / "db_images"

_RISK_COLOR: dict[str, list[int]] = {
    "low":    [0, 180, 0, 200],
    "medium": [255, 165, 0, 200],
    "high":   [220, 0, 0, 200],
}
_RISK_SCORE: dict[str, float] = {"low": 1.0, "medium": 2.0, "high": 3.0}
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


def _region_view_state(region_id: str) -> pdk.ViewState:
    region = REGIONS[region_id]
    center_lon = (region.lon_min + region.lon_max) / 2.0
    center_lat = (region.lat_min + region.lat_max) / 2.0
    span = max(region.lon_max - region.lon_min, region.lat_max - region.lat_min)
    zoom = max(6, min(12, round(8 - math.log2(max(span, 0.01) / 0.1))))
    return pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=zoom, pitch=0)


def _filter_rows(
    rows: list[dict[str, object]],
    risk_filter: list[str],
    model_filter: list[str],
    date_range: tuple[str, str] | tuple[object, ...],
    region_filter: str | None,
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
        if region_filter and r.get("region_id") != region_filter:
            continue
        out.append(r)
    return out


def _time_series_charts(rows: list[dict[str, object]]) -> None:
    """Render per-tile lines and region-average chart for the given rows."""
    # Group by tile key (rounded coordinates) and timestamp date.
    tile_series: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        risk = str(r.get("risk_level") or "")
        score = _RISK_SCORE.get(risk)
        if score is None:
            continue
        tile_key = f"{float(r['lon']):.3f},{float(r['lat']):.3f}"  # type: ignore[arg-type]
        date = str(r.get("timestamp", ""))[:10]
        if not date:
            continue
        tile_series.setdefault(tile_key, {}).setdefault(date, []).append(score)

    if not tile_series:
        return

    # Compute per-tile mean score per date.
    tile_means: dict[str, dict[str, float]] = {
        key: {date: sum(scores) / len(scores) for date, scores in by_date.items()}
        for key, by_date in tile_series.items()
    }

    all_dates = sorted({d for by_date in tile_means.values() for d in by_date})

    # Region average and band across all tiles per date.
    region_avg: list[float] = []
    region_min: list[float] = []
    region_max: list[float] = []
    for date in all_dates:
        scores_at_date = [tile_means[k][date] for k in tile_means if date in tile_means[k]]
        if scores_at_date:
            region_avg.append(sum(scores_at_date) / len(scores_at_date))
            region_min.append(min(scores_at_date))
            region_max.append(max(scores_at_date))

    st.subheader("Risk over time")
    col_left, col_right = st.columns(2)

    # Left: per-tile lines (top 10 highest-risk tiles when grid is large).
    with col_left:
        st.caption("Per-tile risk score")
        sorted_tiles = sorted(
            tile_means.items(),
            key=lambda kv: sum(kv[1].values()) / len(kv[1]),
            reverse=True,
        )
        display_tiles = sorted_tiles[:10] if len(sorted_tiles) > 20 else sorted_tiles

        fig_tiles = go.Figure()
        for tile_key, by_date in display_tiles:
            ys = [by_date.get(d) for d in all_dates]
            mean_score = sum(v for v in ys if v is not None) / max(1, sum(1 for v in ys if v is not None))
            if mean_score >= 2.5:
                color = "red"
            elif mean_score >= 1.5:
                color = "orange"
            else:
                color = "green"
            fig_tiles.add_trace(go.Scatter(
                x=all_dates,
                y=ys,
                mode="lines+markers",
                name=tile_key,
                line={"color": color, "width": 1},
                marker={"size": 4},
                connectgaps=True,
            ))
        fig_tiles.update_layout(
            yaxis={"tickvals": [1, 2, 3], "ticktext": ["low", "medium", "high"], "range": [0.5, 3.5]},
            xaxis_title="Date",
            showlegend=False,
            margin={"t": 10, "b": 40, "l": 60, "r": 10},
            height=300,
        )
        st.plotly_chart(fig_tiles, use_container_width=True)

    # Right: region average with min/max shaded band.
    with col_right:
        st.caption("Region average (shaded band: min/max)")
        fig_avg = go.Figure()
        fig_avg.add_trace(go.Scatter(
            x=all_dates + all_dates[::-1],
            y=region_max + region_min[::-1],
            fill="toself",
            fillcolor="rgba(255,165,0,0.15)",
            line={"color": "rgba(255,255,255,0)"},
            hoverinfo="skip",
            showlegend=False,
            name="band",
        ))
        fig_avg.add_trace(go.Scatter(
            x=all_dates,
            y=region_avg,
            mode="lines+markers",
            name="region avg",
            line={"color": "darkorange", "width": 2},
            marker={"size": 5},
        ))
        fig_avg.update_layout(
            yaxis={"tickvals": [1, 2, 3], "ticktext": ["low", "medium", "high"], "range": [0.5, 3.5]},
            xaxis_title="Date",
            showlegend=False,
            margin={"t": 10, "b": 40, "l": 60, "r": 10},
            height=300,
        )
        st.plotly_chart(fig_avg, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Wildfire Prevention", layout="wide")
    st.title("Wildfire Risk Map")

    conn = init_db(DB_PATH)
    all_rows = fetch_all(conn)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Filters")

        region_options = ["All"] + list(REGIONS.keys())
        selected_region = st.selectbox("Region", region_options, index=0)
        region_filter = None if selected_region == "All" else selected_region

        if region_filter:
            st.caption(REGIONS[region_filter].description)

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
    rows = _filter_rows(all_rows, risk_filter, model_filter, date_range, region_filter)

    if not rows:
        st.info(
            "No predictions in the DB yet. "
            "Start `predict.py` to begin collecting live predictions, "
            "or run `backfill.py --region <name>` to seed with historical data."
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

    if region_filter:
        view_state = _region_view_state(region_filter)
    else:
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

    st.pydeck_chart(deck, use_container_width=True)

    # --- Time-series charts (shown when a region is selected) ---
    if region_filter:
        _time_series_charts(rows)

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
        "water_body_present", "image_quality_limited", "model", "source",
        "region_id", "created_at",
    ]
    st.json({k: detail_row.get(k) for k in display_fields})

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
