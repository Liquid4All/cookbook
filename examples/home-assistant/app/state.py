import random

home_state = {
    "lights": {
        "living_room": {"state": "off"},
        "bedroom":     {"state": "off"},
        "kitchen":     {"state": "off"},
        "bathroom":    {"state": "off"},
        "office":      {"state": "off"},
        "hallway":     {"state": "off"},
    },
    "thermostat": {"temperature": 68, "mode": "auto"},
    "doors": {
        "front":  "locked",
        "back":   "locked",
        "garage": "locked",
        "side":   "locked",
    },
    "active_scene": None,
}


def randomize_state() -> None:
    """Randomize lights and doors in home_state for varied training data."""
    for room in home_state["lights"]:
        home_state["lights"][room]["state"] = random.choice(["on", "off"])
    for door in home_state["doors"]:
        home_state["doors"][door] = random.choice(["locked", "unlocked"])
