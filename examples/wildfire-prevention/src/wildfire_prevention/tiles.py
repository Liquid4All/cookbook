"""Deterministic spatial and temporal tile generation for dataset construction.

All functions are pure and parameter-driven: the same inputs always produce the
same outputs, which is the basis for reproducible dataset generation.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class TileCoord:
    """A single tile center produced by the spatial grid."""

    index: int  # 0-based, row-major within the grid
    lon: float
    lat: float


def spatial_grid(
    center_lon: float,
    center_lat: float,
    n_tiles: int,
    size_km: float,
) -> list[TileCoord]:
    """Return n_tiles tile centers arranged in a centered square grid.

    Tiles are spaced size_km apart with no overlap. The grid is always centered
    on (center_lon, center_lat). For non-perfect-square n_tiles the first N cells
    of the smallest enclosing square are taken in row-major order.

    index=0 is always the top-left cell of the grid. For n_tiles=1 it is the
    center itself.

    Args:
        center_lon: Longitude of the location center in decimal degrees.
        center_lat: Latitude of the location center in decimal degrees.
        n_tiles: Number of tiles to generate.
        size_km: Edge length of each tile in km, also used as the grid spacing.

    Returns:
        List of TileCoord with length n_tiles, ordered row-major top-left to
        bottom-right.
    """
    if n_tiles < 1:
        raise ValueError(f"n_tiles must be >= 1, got {n_tiles}")

    grid_size = math.ceil(math.sqrt(n_tiles))

    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(center_lat))

    coords: list[TileCoord] = []
    for i in range(n_tiles):
        row = i // grid_size
        col = i % grid_size

        # Center the grid symmetrically around (center_lon, center_lat).
        row_offset = row - (grid_size - 1) / 2.0
        col_offset = col - (grid_size - 1) / 2.0

        delta_lat = row_offset * size_km / km_per_deg_lat
        delta_lon = col_offset * size_km / km_per_deg_lon

        coords.append(
            TileCoord(
                index=i,
                lon=round(center_lon + delta_lon, 6),
                lat=round(center_lat + delta_lat, 6),
            )
        )

    return coords


def temporal_timestamps(
    start_date: datetime,
    end_date: datetime,
    n: int,
) -> list[str]:
    """Return n evenly spaced ISO 8601 timestamps within [start_date, end_date].

    Uses bin-center placement: timestamps fall at the midpoints of n equal
    sub-intervals, so they are always strictly inside the window and evenly
    spaced regardless of n. This avoids edge cases at the train/test cutoff.

    For n=1 the single timestamp is the midpoint of the full window.
    For n=2 the timestamps are at 25% and 75% of the window.

    Args:
        start_date: Start of the time window (inclusive).
        end_date: End of the time window (inclusive).
        n: Number of timestamps to generate.

    Returns:
        List of ISO 8601 strings, length n, in ascending order.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if end_date <= start_date:
        raise ValueError("end_date must be after start_date")

    duration = end_date - start_date
    bin_width = duration / n

    timestamps: list[str] = []
    for i in range(n):
        ts = start_date + bin_width * i + bin_width / 2
        timestamps.append(_format_iso(ts))

    return timestamps


def train_test_cutoff(
    start_date: datetime,
    end_date: datetime,
    test_ratio: float,
) -> datetime:
    """Return the cutoff timestamp that splits the window into train and test.

    Samples with timestamp < cutoff go to train.
    Samples with timestamp >= cutoff go to test.

    The cutoff divides the window so the test fraction equals test_ratio of the
    total duration. Combined with bin-center temporal timestamps, this guarantees
    that no sample pair within one bin width spans the boundary.

    Args:
        start_date: Start of the time window.
        end_date: End of the time window.
        test_ratio: Fraction of the window reserved for test, in (0, 1).

    Returns:
        The cutoff datetime.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")

    duration = end_date - start_date
    return start_date + duration * (1.0 - test_ratio)


def _format_iso(dt: datetime) -> str:
    """Return a compact ISO 8601 string with second precision."""
    truncated = dt - timedelta(microseconds=dt.microsecond)
    return truncated.isoformat()
