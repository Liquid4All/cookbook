import requests

SIMSAT_BASE_URL = "http://localhost:9005"


def fetch_image(
    lon: float,
    lat: float,
    timestamp: str,
    bands: list[str],
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> bytes:
    """Return raw PNG bytes for the requested Sentinel-2 band composite.

    Raises:
        requests.HTTPError: if SimSat returns a non-2xx status (e.g. 404 when
            no Sentinel pass exists for the requested location/time).
    """
    params: list[tuple[str, object]] = [
        ("lon", lon),
        ("lat", lat),
        ("timestamp", timestamp),
        ("size_km", size_km),
        ("return_type", "png"),
    ] + [("spectral_bands", b) for b in bands]
    response = requests.get(f"{base_url}/data/image/sentinel", params=params, timeout=60)
    response.raise_for_status()
    return response.content


def fetch_rgb(
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> bytes:
    """Fetch a natural-color RGB composite (B4-B3-B2)."""
    return fetch_image(lon, lat, timestamp, ["red", "green", "blue"], size_km, base_url)


def fetch_swir(
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float = 10.0,
    base_url: str = SIMSAT_BASE_URL,
) -> bytes:
    """Fetch a SWIR composite (B12-B8-B4) optimised for vegetation stress and dryness."""
    return fetch_image(lon, lat, timestamp, ["swir16", "nir08", "red"], size_km, base_url)
