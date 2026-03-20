from dataclasses import dataclass
from typing import Callable


@dataclass
class Task:
    id: int
    name: str
    difficulty: str
    prompt: str
    verifier: Callable
    history: list[dict] | None = None
    capability: str = ""   # e.g. "toggle_lights.kitchen.on"
    phrasing: str = ""     # direct | colloquial | indirect | question


ALL_ROOMS = {"living_room", "bedroom", "kitchen", "bathroom", "office", "hallway"}


def _find_all_calls(tool_calls: list[dict], tool_name: str) -> list[dict]:
    return [call for call in tool_calls if call["name"] == tool_name]


def _find_last_call(tool_calls: list[dict], tool_name: str) -> dict | None:
    calls = _find_all_calls(tool_calls, tool_name)
    return calls[-1] if calls else None


def _result(task_id, name, difficulty, passed, call, duration):
    from benchmark.run import TaskResult
    return TaskResult(
        task_id=task_id,
        name=name,
        difficulty=difficulty,
        passed=passed,
        tool_called=call["name"] if call else None,
        args_correct=passed,
        duration_s=duration,
    )


# ---------------------------------------------------------------------------
# Verifier factory functions
# ---------------------------------------------------------------------------

def _make_light_verifier(task_id, name, difficulty, room, expected_state):
    def v(tool_calls, duration, state):
        call = _find_last_call(tool_calls, "toggle_lights")
        passed = state["lights"][room]["state"] == expected_state
        return _result(task_id, name, difficulty, passed, call, duration)
    return v


def _make_door_verifier(task_id, name, difficulty, door, expected_state):
    def v(tool_calls, duration, state):
        call = _find_last_call(tool_calls, "lock_door")
        passed = state["doors"][door] == expected_state
        return _result(task_id, name, difficulty, passed, call, duration)
    return v


def _make_thermostat_verifier(task_id, name, difficulty, temp, mode):
    def v(tool_calls, duration, state):
        call = _find_last_call(tool_calls, "set_thermostat")
        passed = (state["thermostat"]["temperature"] == temp
                  and state["thermostat"]["mode"] == mode)
        return _result(task_id, name, difficulty, passed, call, duration)
    return v


def _make_thermostat_mode_verifier(task_id, name, difficulty, expected_mode):
    def v(tool_calls, duration, state):
        call = _find_last_call(tool_calls, "set_thermostat")
        passed = call is not None and state["thermostat"]["mode"] == expected_mode
        return _result(task_id, name, difficulty, passed, call, duration)
    return v


def _make_scene_verifier(task_id, name, difficulty, scene_name):
    def v(tool_calls, duration, state):
        call = _find_last_call(tool_calls, "set_scene")
        passed = state["active_scene"] == scene_name
        return _result(task_id, name, difficulty, passed, call, duration)
    return v


def _make_status_verifier(task_id, name, difficulty, device_type, room=None):
    def v(tool_calls, duration, state):
        calls = _find_all_calls(tool_calls, "get_device_status")
        if room is not None:
            call = next((c for c in calls
                         if c["args"].get("device_type") in {device_type, "all"}
                         and c["args"].get("room") == room), None)
        else:
            call = next((c for c in calls
                         if c["args"].get("device_type") in {device_type, "all"}), None)
        passed = call is not None
        return _result(task_id, name, difficulty, passed, call, duration)
    return v


def _make_reject_verifier(task_id, name, difficulty, reason):
    def v(tool_calls, duration, state):
        call = _find_last_call(tool_calls, "intent_unclear")
        passed = call is not None and call["args"].get("reason") == reason
        return _result(task_id, name, difficulty, passed, call, duration)
    return v


# ---------------------------------------------------------------------------
# Existing task verifiers (IDs 1-19)
# ---------------------------------------------------------------------------

def _verify_task_1(tool_calls, duration, state):
    call = _find_last_call(tool_calls, "toggle_lights")
    passed = state["lights"]["kitchen"]["state"] == "on"
    return _result(1, "Turn on kitchen lights", "easy", passed, call, duration)


def _verify_task_2(tool_calls, duration, state):
    call = _find_last_call(tool_calls, "lock_door")
    passed = state["doors"]["front"] == "locked"
    return _result(2, "Lock the front door", "easy", passed, call, duration)


def _verify_task_3(tool_calls, duration, state):
    call = _find_last_call(tool_calls, "set_thermostat")
    passed = (
        state["thermostat"]["temperature"] == 72
        and state["thermostat"]["mode"] == "heat"
    )
    return _result(3, "Heat house to 72 degrees", "easy", passed, call, duration)


def _verify_task_4(tool_calls, duration, state):
    calls = _find_all_calls(tool_calls, "get_device_status")
    device_types = {c["args"].get("device_type") for c in calls}
    # Accept a single "all" call or individual calls covering all three types
    passed = (
        "all" in device_types
        or {"lights", "thermostat", "door"}.issubset(device_types)
    )
    call = _find_last_call(tool_calls, "get_device_status")
    return _result(4, "Get status of all devices", "easy", passed, call, duration)


def _verify_task_5(tool_calls, duration, state):
    call = _find_last_call(tool_calls, "set_scene")
    passed = state["active_scene"] == "movie_night"
    return _result(5, "Activate movie night scene", "medium", passed, call, duration)


def _verify_task_6(tool_calls, duration, state):
    call = _find_last_call(tool_calls, "lock_door")
    passed = state["doors"]["garage"] == "unlocked"
    return _result(6, "Unlock the garage door", "medium", passed, call, duration)


def _verify_task_7(tool_calls, duration, state):
    call = _find_last_call(tool_calls, "get_device_status")
    passed = (
        call is not None
        and call["args"].get("device_type") == "lights"
        and call["args"].get("room") == "bedroom"
    )
    return _result(7, "Check bedroom light status", "medium", passed, call, duration)


def _verify_task_8(tool_calls, duration, state):
    call = _find_last_call(tool_calls, "set_thermostat")
    passed = (
        state["thermostat"]["temperature"] == 74
        and state["thermostat"]["mode"] == "cool"
    )
    return _result(8, "Cool house to 74 degrees", "medium", passed, call, duration)


def _verify_task_9(tool_calls, duration, state):
    # Requires inferring "heading out for the day" -> scene="away" AND handler applied full away state
    call = _find_last_call(tool_calls, "set_scene")
    passed = (
        state["active_scene"] == "away"
        and all(state["lights"][r]["state"] == "off" for r in ALL_ROOMS)
        and all(v == "locked" for v in state["doors"].values())
        and state["thermostat"]["temperature"] == 65
    )
    return _result(9, "Away scene via indirect phrasing", "hard", passed, call, duration)


def _verify_task_10(tool_calls, duration, state):
    # Requires calling two tools in one turn
    back_door_calls    = [c for c in _find_all_calls(tool_calls, "lock_door")     if c["args"].get("door") == "back"]
    office_light_calls = [c for c in _find_all_calls(tool_calls, "toggle_lights") if c["args"].get("room") == "office"]
    call = (back_door_calls[-1] if back_door_calls else None) or (office_light_calls[-1] if office_light_calls else None)
    passed = (
        state["doors"]["back"] == "locked"
        and state["lights"]["office"]["state"] == "off"
    )
    return _result(10, "Lock back door + off office lights (multi-tool)", "hard", passed, call, duration)


def _verify_task_11(tool_calls, duration, state):
    calls = _find_all_calls(tool_calls, "toggle_lights")
    call = calls[0] if calls else None
    passed = all(state["lights"][r]["state"] == "on" for r in ALL_ROOMS)
    return _result(11, "Turn on all lights (multi-tool)", "hard", passed, call, duration)


def _verify_task_12(tool_calls, duration, state):
    # "switch it off" after turning on the bedroom light - pronoun reference
    call = _find_last_call(tool_calls, "toggle_lights")
    passed = state["lights"]["bedroom"]["state"] == "off"
    return _result(12, "Turn off bedroom light (pronoun reference)", "hard", passed, call, duration)


def _verify_task_13(tool_calls, duration, state):
    # "keep the hallway one off" after turning on all lights - correction
    calls = _find_all_calls(tool_calls, "toggle_lights")
    call = calls[0] if calls else None
    non_hallway = ALL_ROOMS - {"hallway"}
    passed = (
        all(state["lights"][r]["state"] == "on" for r in non_hallway)
        and state["lights"]["hallway"]["state"] == "off"
    )
    return _result(13, "Correct bulk action (hallway off)", "hard", passed, call, duration)


def _verify_task_14(tool_calls, duration, state, initial_state=None):
    # "bump it up by 2 degrees" after setting thermostat to 68 - relative adjustment
    call = _find_last_call(tool_calls, "set_thermostat")
    initial_temp = (initial_state["thermostat"]["temperature"]
                    if initial_state is not None else 68)
    passed = state["thermostat"]["temperature"] == initial_temp + 2
    return _result(14, "Relative thermostat increase (+2 degrees)", "hard", passed, call, duration)


def _verify_task_15(tool_calls, duration, state):
    # "unlock the first one" after locking front then garage - back-reference
    front_calls = [c for c in _find_all_calls(tool_calls, "lock_door") if c["args"].get("door") == "front"]
    call = front_calls[-1] if front_calls else None
    passed = state["doors"]["front"] == "unlocked"
    return _result(15, "Unlock first door (3-turn back-reference)", "hard", passed, call, duration)


def _verify_task_16(tool_calls, duration, state):
    # "Dim the lights to 50%" - brightness control is unsupported
    call = _find_last_call(tool_calls, "intent_unclear")
    passed = call is not None and call["args"].get("reason") == "unsupported_device"
    return _result(16, "Reject: unsupported device (dim lights)", "easy", passed, call, duration)


def _verify_task_17(tool_calls, duration, state):
    # "Order a pizza" - off-topic request
    call = _find_last_call(tool_calls, "intent_unclear")
    passed = call is not None and call["args"].get("reason") == "off_topic"
    return _result(17, "Reject: off-topic request", "easy", passed, call, duration)


def _verify_task_18(tool_calls, duration, state):
    # "Turn it on" with no prior context - incomplete request
    call = _find_last_call(tool_calls, "intent_unclear")
    passed = call is not None and call["args"].get("reason") == "incomplete"
    return _result(18, "Reject: incomplete request", "easy", passed, call, duration)


def _verify_task_19(tool_calls, duration, state):
    # "Make it comfortable" - ambiguous request
    call = _find_last_call(tool_calls, "intent_unclear")
    passed = call is not None and call["args"].get("reason") == "ambiguous"
    return _result(19, "Reject: ambiguous request", "easy", passed, call, duration)


# ---------------------------------------------------------------------------
# Adversarial verifiers for IDs 51, 52, 68
# ---------------------------------------------------------------------------

def _verify_task_51(tool_calls, duration, state):
    call = _find_last_call(tool_calls, "toggle_lights")
    passed = (call is not None and call["args"].get("state") == "on"
              and any(state["lights"][r]["state"] == "on" for r in ALL_ROOMS))
    return _result(51, "Lights on: implicit dark-room (adversarial)", "hard", passed, call, duration)


def _verify_task_52(tool_calls, duration, state):
    call = _find_last_call(tool_calls, "toggle_lights")
    passed = state["lights"]["bathroom"]["state"] == "off"
    return _result(52, "Bathroom off: implicit departure (adversarial)", "hard", passed, call, duration)


def _verify_task_68(tool_calls, duration, state):
    lock_calls = _find_all_calls(tool_calls, "lock_door")
    scene_call = _find_last_call(tool_calls, "set_scene")
    passed = (len(lock_calls) > 0
              or (scene_call is not None
                  and scene_call["args"].get("scene") in {"away", "bedtime"}))
    call = lock_calls[-1] if lock_calls else scene_call
    return _result(68, "Lock up: implicit security (adversarial)", "hard", passed, call, duration)


# ---------------------------------------------------------------------------
# Task list
# ---------------------------------------------------------------------------

TASKS = [
    # --- Existing tasks (IDs 1-19, unchanged) ---
    Task(1,  "Turn on kitchen lights",                      "easy",   "Turn on the kitchen lights",                                        _verify_task_1,  capability="toggle_lights.kitchen.on",    phrasing="direct"),
    Task(2,  "Lock the front door",                         "easy",   "Lock the front door",                                               _verify_task_2,  capability="lock_door.front.lock",        phrasing="direct"),
    Task(3,  "Heat house to 72 degrees",                    "easy",   "Heat the house to 72 degrees",                                      _verify_task_3,  capability="set_thermostat.heat",         phrasing="direct"),
    Task(4,  "Get status of all devices",                   "easy",   "What is the current status of all devices?",                        _verify_task_4,  capability="get_device_status.all",       phrasing="question"),
    Task(5,  "Activate movie night scene",                  "medium", "Activate movie night mode",                                         _verify_task_5,  capability="set_scene.movie_night",       phrasing="direct"),
    Task(6,  "Unlock the garage door",                      "medium", "Unlock the garage door",                                            _verify_task_6,  capability="lock_door.garage.unlock",     phrasing="direct"),
    Task(7,  "Check bedroom light status",                  "medium", "Are the bedroom lights on?",                                        _verify_task_7,  capability="get_device_status.lights",    phrasing="question"),
    Task(8,  "Cool house to 74 degrees",                    "medium", "Cool the house down to 74 degrees",                                 _verify_task_8,  capability="set_thermostat.cool",         phrasing="direct"),
    Task(9,  "Away scene via indirect phrasing",            "hard",   "I'm heading out for the day, set the house accordingly",            _verify_task_9,  capability="set_scene.away",              phrasing="indirect"),
    Task(10, "Lock back door + off office lights",          "hard",   "Lock the back door and turn off the office lights",                 _verify_task_10, capability="multi_tool",                  phrasing="direct"),
    Task(11, "Turn on all lights (multi-tool)",             "hard",   "switch on all the lights",                                          _verify_task_11, capability="multi_tool",                  phrasing="colloquial"),
    Task(12, "Turn off bedroom light (pronoun reference)",  "hard",   "switch it off",                                                     _verify_task_12,
         history=[
             {"role": "user",      "content": "switch on the bedroom light"},
             {"role": "assistant", "content": "The bedroom light has been turned on."},
         ], capability="toggle_lights.bedroom.off", phrasing="indirect"),
    Task(13, "Correct bulk action (hallway off)",           "hard",   "actually keep the hallway one off",                                 _verify_task_13,
         history=[
             {"role": "user",      "content": "turn on all the lights"},
             {"role": "assistant", "content": "All lights have been turned on."},
         ], capability="multi_tool", phrasing="indirect"),
    Task(14, "Relative thermostat increase (+2 degrees)",   "hard",   "bump it up by 2 degrees",                                           _verify_task_14,
         history=[
             {"role": "user",      "content": "set the thermostat to 68 degrees"},
             {"role": "assistant", "content": "The thermostat has been set to 68°F."},
         ], capability="set_thermostat.heat", phrasing="indirect"),
    Task(15, "Unlock first door (3-turn back-reference)",   "hard",   "unlock the first one",                                              _verify_task_15,
         history=[
             {"role": "user",      "content": "lock the front door"},
             {"role": "assistant", "content": "The front door has been locked."},
             {"role": "user",      "content": "and the garage too"},
             {"role": "assistant", "content": "The garage door has been locked."},
         ], capability="lock_door.front.unlock", phrasing="indirect"),
    Task(16, "Reject: unsupported device (dim lights)",    "easy",   "Dim the living room lights to 50%",                                 _verify_task_16, capability="intent_unclear.unsupported_device", phrasing="direct"),
    Task(17, "Reject: off-topic request",                  "easy",   "Order a pizza for delivery",                                        _verify_task_17, capability="intent_unclear.off_topic",         phrasing="direct"),
    Task(18, "Reject: incomplete request",                 "easy",   "Turn it on",                                                        _verify_task_18, capability="intent_unclear.incomplete",        phrasing="direct"),
    Task(19, "Reject: ambiguous request",                  "easy",   "Make it more comfortable in here",                                  _verify_task_19, capability="intent_unclear.ambiguous",         phrasing="indirect"),

    # --- toggle_lights: on, new rooms (IDs 20-34) ---
    Task(20, "Living room lights on (direct)",   "easy",   "Turn on the living room lights",         _make_light_verifier(20, "Living room lights on (direct)",   "easy",   "living_room", "on"), capability="toggle_lights.living_room.on",  phrasing="direct"),
    Task(21, "Living room lights on (colloq)",   "easy",   "Living room lights on please",           _make_light_verifier(21, "Living room lights on (colloq)",   "easy",   "living_room", "on"), capability="toggle_lights.living_room.on",  phrasing="colloquial"),
    Task(22, "Living room lights on (question)", "easy",   "Can you get the living room lights on?", _make_light_verifier(22, "Living room lights on (question)", "easy",   "living_room", "on"), capability="toggle_lights.living_room.on",  phrasing="question"),
    Task(23, "Bedroom lights on (direct)",       "easy",   "Turn on the bedroom lights",             _make_light_verifier(23, "Bedroom lights on (direct)",       "easy",   "bedroom",     "on"), capability="toggle_lights.bedroom.on",      phrasing="direct"),
    Task(24, "Bedroom lights on (colloquial)",   "easy",   "Bedroom lights on",                      _make_light_verifier(24, "Bedroom lights on (colloquial)",   "easy",   "bedroom",     "on"), capability="toggle_lights.bedroom.on",      phrasing="colloquial"),
    Task(25, "Bedroom lights on (question)",     "easy",   "Can you switch on the bedroom lights?",  _make_light_verifier(25, "Bedroom lights on (question)",     "easy",   "bedroom",     "on"), capability="toggle_lights.bedroom.on",      phrasing="question"),
    Task(26, "Bathroom light on (direct)",       "easy",   "Turn on the bathroom light",             _make_light_verifier(26, "Bathroom light on (direct)",       "easy",   "bathroom",    "on"), capability="toggle_lights.bathroom.on",     phrasing="direct"),
    Task(27, "Bathroom light on (colloquial)",   "easy",   "Bathroom light on please",               _make_light_verifier(27, "Bathroom light on (colloquial)",   "easy",   "bathroom",    "on"), capability="toggle_lights.bathroom.on",     phrasing="colloquial"),
    Task(28, "Bathroom light on (indirect)",     "medium", "It's too dark in the bathroom",          _make_light_verifier(28, "Bathroom light on (indirect)",     "medium", "bathroom",    "on"), capability="toggle_lights.bathroom.on",     phrasing="indirect"),
    Task(29, "Hallway light on (direct)",        "easy",   "Turn on the hallway light",              _make_light_verifier(29, "Hallway light on (direct)",        "easy",   "hallway",     "on"), capability="toggle_lights.hallway.on",      phrasing="direct"),
    Task(30, "Hallway lights on (colloquial)",   "easy",   "Hallway lights on",                      _make_light_verifier(30, "Hallway lights on (colloquial)",   "easy",   "hallway",     "on"), capability="toggle_lights.hallway.on",      phrasing="colloquial"),
    Task(31, "Hallway light on (question)",      "easy",   "Could you switch the hallway light on?", _make_light_verifier(31, "Hallway light on (question)",      "easy",   "hallway",     "on"), capability="toggle_lights.hallway.on",      phrasing="question"),
    Task(32, "Office lights on (direct)",        "easy",   "Turn on the office lights",              _make_light_verifier(32, "Office lights on (direct)",        "easy",   "office",      "on"), capability="toggle_lights.office.on",       phrasing="direct"),
    Task(33, "Office lights on (colloquial)",    "easy",   "Office lights on please",                _make_light_verifier(33, "Office lights on (colloquial)",    "easy",   "office",      "on"), capability="toggle_lights.office.on",       phrasing="colloquial"),
    Task(34, "Office lights on (indirect)",      "medium", "I need to see in the office",            _make_light_verifier(34, "Office lights on (indirect)",      "medium", "office",      "on"), capability="toggle_lights.office.on",       phrasing="indirect"),

    # --- toggle_lights: off, all rooms (IDs 35-50) ---
    Task(35, "Kitchen lights off (direct)",       "easy",   "Turn off the kitchen lights",                    _make_light_verifier(35, "Kitchen lights off (direct)",       "easy",   "kitchen",     "off"), capability="toggle_lights.kitchen.off",     phrasing="direct"),
    Task(36, "Kitchen lights off (colloquial)",   "easy",   "Kitchen lights off",                             _make_light_verifier(36, "Kitchen lights off (colloquial)",   "easy",   "kitchen",     "off"), capability="toggle_lights.kitchen.off",     phrasing="colloquial"),
    Task(37, "Kitchen lights off (question)",     "easy",   "Can you switch off the kitchen lights?",         _make_light_verifier(37, "Kitchen lights off (question)",     "easy",   "kitchen",     "off"), capability="toggle_lights.kitchen.off",     phrasing="question"),
    Task(38, "Living room lights off (direct)",   "easy",   "Turn off the living room lights",                _make_light_verifier(38, "Living room lights off (direct)",   "easy",   "living_room", "off"), capability="toggle_lights.living_room.off", phrasing="direct"),
    Task(39, "Living room lights off (colloq)",   "easy",   "Living room lights off please",                  _make_light_verifier(39, "Living room lights off (colloq)",   "easy",   "living_room", "off"), capability="toggle_lights.living_room.off", phrasing="colloquial"),
    Task(40, "Living room lights off (indirect)", "medium", "No one is in the living room anymore",           _make_light_verifier(40, "Living room lights off (indirect)", "medium", "living_room", "off"), capability="toggle_lights.living_room.off", phrasing="indirect"),
    Task(41, "Bedroom lights off (direct)",       "easy",   "Turn off the bedroom lights",                    _make_light_verifier(41, "Bedroom lights off (direct)",       "easy",   "bedroom",     "off"), capability="toggle_lights.bedroom.off",     phrasing="direct"),
    Task(42, "Bedroom lights off (colloquial)",   "easy",   "Bedroom lights off",                             _make_light_verifier(42, "Bedroom lights off (colloquial)",   "easy",   "bedroom",     "off"), capability="toggle_lights.bedroom.off",     phrasing="colloquial"),
    Task(43, "Bedroom lights off (question)",     "easy",   "Are the bedroom lights still on? Turn them off", _make_light_verifier(43, "Bedroom lights off (question)",     "easy",   "bedroom",     "off"), capability="toggle_lights.bedroom.off",     phrasing="question"),
    Task(44, "Bathroom light off (direct)",       "easy",   "Turn off the bathroom light",                    _make_light_verifier(44, "Bathroom light off (direct)",       "easy",   "bathroom",    "off"), capability="toggle_lights.bathroom.off",    phrasing="direct"),
    Task(45, "Bathroom light off (colloquial)",   "easy",   "Bathroom light off please",                      _make_light_verifier(45, "Bathroom light off (colloquial)",   "easy",   "bathroom",    "off"), capability="toggle_lights.bathroom.off",    phrasing="colloquial"),
    Task(46, "Hallway light off (direct)",        "easy",   "Turn off the hallway light",                     _make_light_verifier(46, "Hallway light off (direct)",        "easy",   "hallway",     "off"), capability="toggle_lights.hallway.off",     phrasing="direct"),
    Task(47, "Hallway light off (indirect)",      "medium", "The hallway light is wasting electricity",       _make_light_verifier(47, "Hallway light off (indirect)",      "medium", "hallway",     "off"), capability="toggle_lights.hallway.off",     phrasing="indirect"),
    Task(48, "Office lights off (direct)",        "easy",   "Turn off the office lights",                     _make_light_verifier(48, "Office lights off (direct)",        "easy",   "office",      "off"), capability="toggle_lights.office.off",      phrasing="direct"),
    Task(49, "Office lights off (colloquial)",    "easy",   "Office lights off",                              _make_light_verifier(49, "Office lights off (colloquial)",    "easy",   "office",      "off"), capability="toggle_lights.office.off",      phrasing="colloquial"),
    Task(50, "Office lights off (question)",      "easy",   "Can you kill the office lights?",                _make_light_verifier(50, "Office lights off (question)",      "easy",   "office",      "off"), capability="toggle_lights.office.off",      phrasing="question"),

    # --- toggle_lights: adversarial (IDs 51-52) ---
    Task(51, "Lights on: implicit dark-room",    "hard", "It's a bit dark in here",   _verify_task_51, capability="toggle_lights.any.on",       phrasing="indirect"),
    Task(52, "Bathroom off: implicit departure", "hard", "I just left the bathroom",  _verify_task_52, capability="toggle_lights.bathroom.off", phrasing="indirect"),

    # --- lock_door: new combos (IDs 53-68) ---
    Task(53, "Unlock front door (direct)",     "easy",   "Unlock the front door",             _make_door_verifier(53, "Unlock front door (direct)",     "easy",   "front",  "unlocked"), capability="lock_door.front.unlock",  phrasing="direct"),
    Task(54, "Unlock front door (colloquial)", "easy",   "Front door unlock please",          _make_door_verifier(54, "Unlock front door (colloquial)", "easy",   "front",  "unlocked"), capability="lock_door.front.unlock",  phrasing="colloquial"),
    Task(55, "Unlock front door (question)",   "easy",   "Can you unlock the front door?",    _make_door_verifier(55, "Unlock front door (question)",   "easy",   "front",  "unlocked"), capability="lock_door.front.unlock",  phrasing="question"),
    Task(56, "Unlock back door (direct)",      "easy",   "Unlock the back door",              _make_door_verifier(56, "Unlock back door (direct)",      "easy",   "back",   "unlocked"), capability="lock_door.back.unlock",   phrasing="direct"),
    Task(57, "Unlock back door (colloquial)",  "easy",   "Back door open",                    _make_door_verifier(57, "Unlock back door (colloquial)",  "easy",   "back",   "unlocked"), capability="lock_door.back.unlock",   phrasing="colloquial"),
    Task(58, "Unlock back door (question)",    "easy",   "Could you unlock the back door?",   _make_door_verifier(58, "Unlock back door (question)",    "easy",   "back",   "unlocked"), capability="lock_door.back.unlock",   phrasing="question"),
    Task(59, "Lock garage door (direct)",      "easy",   "Lock the garage door",              _make_door_verifier(59, "Lock garage door (direct)",      "easy",   "garage", "locked"),   capability="lock_door.garage.lock",  phrasing="direct"),
    Task(60, "Lock garage door (colloquial)",  "easy",   "Garage locked please",              _make_door_verifier(60, "Lock garage door (colloquial)",  "easy",   "garage", "locked"),   capability="lock_door.garage.lock",  phrasing="colloquial"),
    Task(61, "Lock garage door (indirect)",    "medium", "Make sure the garage is secure",    _make_door_verifier(61, "Lock garage door (indirect)",    "medium", "garage", "locked"),   capability="lock_door.garage.lock",  phrasing="indirect"),
    Task(62, "Lock side door (direct)",        "easy",   "Lock the side door",                _make_door_verifier(62, "Lock side door (direct)",        "easy",   "side",   "locked"),   capability="lock_door.side.lock",    phrasing="direct"),
    Task(63, "Lock side door (colloquial)",    "easy",   "Side door, lock it",                _make_door_verifier(63, "Lock side door (colloquial)",    "easy",   "side",   "locked"),   capability="lock_door.side.lock",    phrasing="colloquial"),
    Task(64, "Lock side door (question)",      "easy",   "Can you lock the side door?",       _make_door_verifier(64, "Lock side door (question)",      "easy",   "side",   "locked"),   capability="lock_door.side.lock",    phrasing="question"),
    Task(65, "Unlock side door (direct)",      "easy",   "Unlock the side door",              _make_door_verifier(65, "Unlock side door (direct)",      "easy",   "side",   "unlocked"), capability="lock_door.side.unlock",  phrasing="direct"),
    Task(66, "Unlock side door (colloquial)",  "easy",   "Side door unlocked please",         _make_door_verifier(66, "Unlock side door (colloquial)",  "easy",   "side",   "unlocked"), capability="lock_door.side.unlock",  phrasing="colloquial"),
    Task(67, "Unlock side door (question)",    "easy",   "Could you open the side door?",     _make_door_verifier(67, "Unlock side door (question)",    "easy",   "side",   "unlocked"), capability="lock_door.side.unlock",  phrasing="question"),
    Task(68, "Lock up: implicit security",     "hard",   "Lock up before you go",             _verify_task_68, capability="lock_door.any.lock", phrasing="indirect"),

    # --- set_thermostat: auto mode + new temps (IDs 69-77) ---
    Task(69, "Thermostat 70 auto (direct)",       "easy", "Set the thermostat to 70 degrees in auto mode", _make_thermostat_verifier(69, "Thermostat 70 auto (direct)",       "easy", 70, "auto"), capability="set_thermostat.auto", phrasing="direct"),
    Task(70, "Thermostat 70 auto (colloquial)",   "easy", "70 degrees auto please",                        _make_thermostat_verifier(70, "Thermostat 70 auto (colloquial)",   "easy", 70, "auto"), capability="set_thermostat.auto", phrasing="colloquial"),
    Task(71, "Thermostat 70 auto (question)",     "easy", "Can you set the temperature to 70 on auto?",    _make_thermostat_verifier(71, "Thermostat 70 auto (question)",     "easy", 70, "auto"), capability="set_thermostat.auto", phrasing="question"),
    Task(72, "Heat to 65 degrees (direct)",       "easy", "Heat the house to 65 degrees",                  _make_thermostat_verifier(72, "Heat to 65 degrees (direct)",       "easy", 65, "heat"), capability="set_thermostat.heat", phrasing="direct"),
    Task(73, "Heat to 65 degrees (colloquial)",   "easy", "65 heat mode",                                  _make_thermostat_verifier(73, "Heat to 65 degrees (colloquial)",   "easy", 65, "heat"), capability="set_thermostat.heat", phrasing="colloquial"),
    Task(74, "Cool to 78 degrees (direct)",       "easy", "Cool the house down to 78 degrees",             _make_thermostat_verifier(74, "Cool to 78 degrees (direct)",       "easy", 78, "cool"), capability="set_thermostat.cool", phrasing="direct"),
    Task(75, "Cool to 78 degrees (colloquial)",   "easy", "Set AC to 78",                                  _make_thermostat_verifier(75, "Cool to 78 degrees (colloquial)",   "easy", 78, "cool"), capability="set_thermostat.cool", phrasing="colloquial"),
    Task(76, "Cool mode: implicit hot",           "hard", "It's getting really hot in here",               _make_thermostat_mode_verifier(76, "Cool mode: implicit hot",           "hard", "cool"), capability="set_thermostat.cool", phrasing="indirect"),
    Task(77, "Heat mode: implicit cold",          "hard", "It's freezing, can you warm it up a bit?",      _make_thermostat_mode_verifier(77, "Heat mode: implicit cold",          "hard", "heat"), capability="set_thermostat.heat", phrasing="indirect"),

    # --- get_device_status: missing coverage (IDs 78-86) ---
    Task(78, "Thermostat status (direct)",      "easy",   "What is the thermostat set to?",          _make_status_verifier(78, "Thermostat status (direct)",      "easy",   "thermostat"),            capability="get_device_status.thermostat", phrasing="direct"),
    Task(79, "Thermostat status (question)",    "easy",   "What's the current temperature setting?", _make_status_verifier(79, "Thermostat status (question)",    "easy",   "thermostat"),            capability="get_device_status.thermostat", phrasing="question"),
    Task(80, "Thermostat status (colloquial)",  "easy",   "Thermostat status",                       _make_status_verifier(80, "Thermostat status (colloquial)",  "easy",   "thermostat"),            capability="get_device_status.thermostat", phrasing="colloquial"),
    Task(81, "Door status (direct)",            "easy",   "Are all the doors locked?",               _make_status_verifier(81, "Door status (direct)",            "easy",   "door"),                  capability="get_device_status.door",       phrasing="direct"),
    Task(82, "Door status (question)",          "easy",   "What's the status of the doors?",         _make_status_verifier(82, "Door status (question)",          "easy",   "door"),                  capability="get_device_status.door",       phrasing="question"),
    Task(83, "Door status (colloquial)",        "easy",   "Door status please",                      _make_status_verifier(83, "Door status (colloquial)",        "easy",   "door"),                  capability="get_device_status.door",       phrasing="colloquial"),
    Task(84, "Living room lights status",       "medium", "Are the living room lights on?",          _make_status_verifier(84, "Living room lights status",       "medium", "lights", "living_room"), capability="get_device_status.lights",     phrasing="direct"),
    Task(85, "Kitchen light status (question)", "medium", "Is the kitchen light on?",                _make_status_verifier(85, "Kitchen light status (question)", "medium", "lights", "kitchen"),     capability="get_device_status.lights",     phrasing="question"),
    Task(86, "Bathroom light status (direct)",  "medium", "Check the bathroom light status",         _make_status_verifier(86, "Bathroom light status (direct)",  "medium", "lights", "bathroom"),    capability="get_device_status.lights",     phrasing="direct"),

    # --- set_scene: 3 missing scenes (IDs 87-95) ---
    Task(87, "Bedtime scene (direct)",     "easy", "Activate bedtime mode",                          _make_scene_verifier(87, "Bedtime scene (direct)",     "easy", "bedtime"), capability="set_scene.bedtime",  phrasing="direct"),
    Task(88, "Bedtime scene (colloquial)", "easy", "Bedtime scene please",                           _make_scene_verifier(88, "Bedtime scene (colloquial)", "easy", "bedtime"), capability="set_scene.bedtime",  phrasing="colloquial"),
    Task(89, "Bedtime scene (indirect)",  "hard", "I'm going to sleep now, set things up",           _make_scene_verifier(89, "Bedtime scene (indirect)",  "hard", "bedtime"), capability="set_scene.bedtime",  phrasing="indirect"),
    Task(90, "Morning scene (direct)",    "easy", "Activate the morning scene",                      _make_scene_verifier(90, "Morning scene (direct)",    "easy", "morning"), capability="set_scene.morning",  phrasing="direct"),
    Task(91, "Morning scene (colloquial)","easy", "Morning mode please",                             _make_scene_verifier(91, "Morning scene (colloquial)","easy", "morning"), capability="set_scene.morning",  phrasing="colloquial"),
    Task(92, "Morning scene (indirect)",  "hard", "Good morning, get the house ready for the day",   _make_scene_verifier(92, "Morning scene (indirect)",  "hard", "morning"), capability="set_scene.morning",  phrasing="indirect"),
    Task(93, "Party scene (direct)",      "easy", "Activate party mode",                             _make_scene_verifier(93, "Party scene (direct)",      "easy", "party"),   capability="set_scene.party",    phrasing="direct"),
    Task(94, "Party scene (colloquial)",  "easy", "Party scene please",                              _make_scene_verifier(94, "Party scene (colloquial)",  "easy", "party"),   capability="set_scene.party",    phrasing="colloquial"),
    Task(95, "Party scene (indirect)",   "hard", "We're having people over, set the mood",           _make_scene_verifier(95, "Party scene (indirect)",   "hard", "party"),    capability="set_scene.party",    phrasing="indirect"),

    # --- intent_unclear: additional + boundary (IDs 96-101) ---
    Task(96,  "Reject: unsupported room (patio)", "easy", "Turn on the patio lights",          _make_reject_verifier(96,  "Reject: unsupported room (patio)", "easy", "unsupported_device"), capability="intent_unclear.unsupported_device", phrasing="direct"),
    Task(97,  "Reject: out-of-range temperature", "easy", "Set the temperature to 95 degrees", _make_reject_verifier(97,  "Reject: out-of-range temperature", "easy", "unsupported_device"), capability="intent_unclear.unsupported_device", phrasing="direct"),
    Task(98,  "Reject: off-topic (car)",          "easy", "Lock the car",                      _make_reject_verifier(98,  "Reject: off-topic (car)",          "easy", "off_topic"),          capability="intent_unclear.off_topic",          phrasing="direct"),
    Task(99,  "Reject: off-topic (weather)",      "easy", "What's the weather today?",         _make_reject_verifier(99,  "Reject: off-topic (weather)",      "easy", "off_topic"),          capability="intent_unclear.off_topic",          phrasing="question"),
    Task(100, "Reject: ambiguous (make nicer)",   "easy", "Make it nicer in here",             _make_reject_verifier(100, "Reject: ambiguous (make nicer)",   "easy", "ambiguous"),          capability="intent_unclear.ambiguous",          phrasing="indirect"),
    Task(101, "Reject: incomplete (the thing)",   "easy", "Turn the thing off",                _make_reject_verifier(101, "Reject: incomplete (the thing)",   "easy", "incomplete"),         capability="intent_unclear.incomplete",         phrasing="direct"),
]
