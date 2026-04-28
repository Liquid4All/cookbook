"""Named geographic regions for wildfire risk monitoring.

Each region defines a bounding box over a fire-prone Mediterranean or Spanish
natural area. Tile grids generated from these regions are the unit of work for
both the live watch loop (predict.py) and the historical backfill (backfill.py).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    description: str


REGIONS: dict[str, Region] = {
    "collserola": Region(
        id="collserola",
        name="Parc de Collserola, Barcelona",
        lon_min=2.02,
        lat_min=41.40,
        lon_max=2.22,
        lat_max=41.50,
        description="Metropolitan forest park on Barcelona's doorstep; burns most summers",
    ),
    "garraf": Region(
        id="garraf",
        name="Parc del Garraf, Barcelona",
        lon_min=1.70,
        lat_min=41.18,
        lon_max=1.98,
        lat_max=41.38,
        description="Coastal limestone scrubland south of Barcelona; high fire frequency",
    ),
    "montseny": Region(
        id="montseny",
        name="Parc Natural del Montseny",
        lon_min=2.25,
        lat_min=41.70,
        lon_max=2.55,
        lat_max=41.87,
        description="UNESCO Biosphere Reserve, 50 km north-east of Barcelona",
    ),
    "donana": Region(
        id="donana",
        name="Parque Nacional de Doñana, Huelva",
        lon_min=-6.60,
        lat_min=36.90,
        lon_max=-6.20,
        lat_max=37.20,
        description="Spain's most famous national park; major fire in 2017",
    ),
    "sierra_nevada": Region(
        id="sierra_nevada",
        name="Parque Nacional Sierra Nevada, Granada",
        lon_min=-3.60,
        lat_min=36.90,
        lon_max=-2.90,
        lat_max=37.25,
        description="Mediterranean mountain ecosystem; high fire risk in summer",
    ),
}


def generate_tile_grid(region: Region, size_km: float) -> list[tuple[float, float]]:
    """Return (lon, lat) tile centers covering the region bounding box.

    Latitude step is uniform (size_km / 111 km per degree).
    Longitude step is corrected for latitude so tiles are approximately square.
    """
    lat_step = size_km / 111.0
    lat_mid = (region.lat_min + region.lat_max) / 2.0
    lon_step = size_km / (111.0 * math.cos(math.radians(lat_mid)))

    tiles: list[tuple[float, float]] = []
    lat = region.lat_min + lat_step / 2.0
    while lat <= region.lat_max:
        lon = region.lon_min + lon_step / 2.0
        while lon <= region.lon_max:
            tiles.append((round(lon, 5), round(lat, 5)))
            lon += lon_step
        lat += lat_step
    return tiles


def tile_contains(
    lon: float,
    lat: float,
    tile_lon: float,
    tile_lat: float,
    size_km: float,
) -> bool:
    """Return True if (lon, lat) falls within the bounding box of a tile.

    The tile is centred at (tile_lon, tile_lat) with edge length size_km.
    """
    lat_half = (size_km / 111.0) / 2.0
    lon_half = (size_km / (111.0 * math.cos(math.radians(tile_lat)))) / 2.0
    return (
        tile_lon - lon_half <= lon <= tile_lon + lon_half
        and tile_lat - lat_half <= lat <= tile_lat + lat_half
    )


def find_tile(
    lon: float,
    lat: float,
    tiles: list[tuple[float, float]],
    size_km: float,
) -> tuple[float, float] | None:
    """Return the tile center that contains (lon, lat), or None if outside all tiles."""
    for tile_lon, tile_lat in tiles:
        if tile_contains(lon, lat, tile_lon, tile_lat, size_km):
            return tile_lon, tile_lat
    return None
