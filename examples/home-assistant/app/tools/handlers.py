from app.state import home_state


def toggle_lights(room: str, state: str) -> dict:
    home_state["lights"][room]["state"] = state
    return {"success": True, "room": room, "state": state}


def set_thermostat(temperature: int, mode: str) -> dict:
    home_state["thermostat"]["temperature"] = temperature
    home_state["thermostat"]["mode"] = mode
    return {"success": True, "temperature": temperature, "mode": mode}


def lock_door(door: str, state: str) -> dict:
    home_state["doors"][door] = "locked" if state == "lock" else "unlocked"
    return {"success": True, "door": door, "state": home_state["doors"][door]}


def get_device_status(device_type: str, room: str = None) -> dict:
    if device_type == "lights":
        if room:
            return {"device_type": "lights", "room": room, "status": home_state["lights"].get(room)}
        return {"device_type": "lights", "status": home_state["lights"]}
    elif device_type == "thermostat":
        return {"device_type": "thermostat", "status": home_state["thermostat"]}
    elif device_type == "door":
        if room:
            return {"device_type": "door", "door": room, "status": home_state["doors"].get(room)}
        return {"device_type": "door", "status": home_state["doors"]}
    else:  # "all"
        return {
            "lights":       home_state["lights"],
            "thermostat":   home_state["thermostat"],
            "doors":        home_state["doors"],
            "active_scene": home_state["active_scene"],
        }


def set_scene(scene: str) -> dict:
    lights = home_state["lights"]
    doors  = home_state["doors"]
    therm  = home_state["thermostat"]

    if scene == "movie_night":
        lights["living_room"]["state"] = "on"
        therm["temperature"] = 72
        therm["mode"] = "auto"
    elif scene == "bedtime":
        for room in lights:
            lights[room]["state"] = "off"
        for door in doors:
            doors[door] = "locked"
        therm["temperature"] = 68
        therm["mode"] = "auto"
    elif scene == "morning":
        lights["kitchen"]["state"] = "on"
        lights["hallway"]["state"] = "on"
        therm["temperature"] = 72
        therm["mode"] = "auto"
    elif scene == "away":
        for room in lights:
            lights[room]["state"] = "off"
        for door in doors:
            doors[door] = "locked"
        therm["temperature"] = 65
        therm["mode"] = "auto"
    elif scene == "party":
        lights["living_room"]["state"] = "on"
        lights["kitchen"]["state"] = "on"
        therm["temperature"] = 70
        therm["mode"] = "auto"

    home_state["active_scene"] = scene
    return {"success": True, "scene": scene}


def intent_unclear(reason: str = "unknown") -> dict:
    return {"success": False, "reason": reason}


TOOL_HANDLERS = {
    "toggle_lights":    toggle_lights,
    "set_thermostat":   set_thermostat,
    "lock_door":        lock_door,
    "get_device_status": get_device_status,
    "set_scene":        set_scene,
    "intent_unclear":   intent_unclear,
}
