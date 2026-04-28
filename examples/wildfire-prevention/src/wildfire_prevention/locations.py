from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    id: str
    lon: float
    lat: float
    expected_risk: str  # "low" | "medium" | "high" | "critical" — for validation only, never passed to the model


LOCATIONS: list[Location] = [
    Location("angeles_nf_ca",        -118.1,  34.3,  "high"),
    Location("santa_barbara_ca",     -119.7,  34.5,  "high"),
    Location("napa_valley_ca",       -122.3,  38.5,  "high"),
    Location("sierra_nevada_ca",     -120.5,  37.5,  "medium"),
    Location("alentejo_portugal",      -7.9,  38.5,  "high"),
    Location("attica_greece",          23.7,  38.1,  "high"),
    Location("cerrado_brazil",        -47.9, -15.8,  "high"),
    Location("patagonia_argentina",   -69.0, -40.5,  "medium"),
    Location("black_forest_germany",    8.1,  48.2,  "low"),
    Location("scottish_highlands",     -4.5,  57.0,  "low"),
    Location("borneo_rainforest",     114.0,   1.5,  "low"),
    Location("tanzania_savanna",       35.0,  -5.0,  "low"),
    Location("outback_nsw_australia", 145.0, -32.0,  "high"),
    Location("victorian_alpine_au",   147.0, -37.0,  "high"),
    Location("kalahari_botswana",      24.0, -22.0,  "medium"),
    Location("zagros_iran",            47.0,  33.5,  "medium"),
    Location("negev_israel",           34.8,  30.8,  "medium"),
    Location("alpine_switzerland",      8.2,  46.8,  "low"),
    Location("amazon_brazil",         -60.0,  -3.0,  "low"),
    Location("congo_basin_drc",        23.0,  -2.0,  "low"),
    # Known active fire events — expected to trigger "critical"
    Location("lahaina_maui_hi",      -156.68,  20.88, "high"),  # 2023 Maui fire
    Location("mati_attica_gr",         23.97,  38.05, "high"),  # 2018 Attica fire
]

LOCATIONS_BY_ID: dict[str, Location] = {loc.id: loc for loc in LOCATIONS}
