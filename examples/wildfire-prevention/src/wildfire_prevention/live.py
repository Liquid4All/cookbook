"""Fetch the current satellite position and images from SimSat."""

import requests

from wildfire_prevention.simsat import SIMSAT_BASE_URL


def get_current_position(base_url: str = SIMSAT_BASE_URL) -> tuple[float, float]:
    """Return (lon, lat) of the satellite's current simulated position."""
    response = requests.get(f"{base_url}/data/current/position", timeout=30)
    response.raise_for_status()
    data: dict[str, object] = response.json()
    lon_lat_alt: list[float] = data["lon-lat-alt"]  # type: ignore[assignment]
    return float(lon_lat_alt[0]), float(lon_lat_alt[1])


def _fetch_current_image(
    bands: list[str],
    size_km: float,
    base_url: str = SIMSAT_BASE_URL,
) -> bytes:
    """Call the SimSat live image endpoint (no timestamp required)."""
    params: list[tuple[str, object]] = [
        ("size_km", size_km),
        ("return_type", "png"),
    ] + [("spectral_bands", b) for b in bands]
    response = requests.get(
        f"{base_url}/data/current/image/sentinel", params=params, timeout=60
    )
    response.raise_for_status()
    return response.content


def fetch_live_images(
    size_km: float = 5.0,
    base_url: str = SIMSAT_BASE_URL,
) -> tuple[float, float, bytes, bytes]:
    """Return (lon, lat, rgb_bytes, swir_bytes) at the current satellite position."""
    lon, lat = get_current_position(base_url)
    rgb_bytes = _fetch_current_image(["red", "green", "blue"], size_km, base_url)
    swir_bytes = _fetch_current_image(["swir16", "nir08", "red"], size_km, base_url)
    return lon, lat, rgb_bytes, swir_bytes
