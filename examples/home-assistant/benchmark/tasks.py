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


ALL_ROOMS = {"living_room", "bedroom", "kitchen", "bathroom", "office", "hallway"}


def _find_all_calls(tool_calls: list[dict], tool_name: str) -> list[dict]:
    return [call for call in tool_calls if call["name"] == tool_name]


def _find_last_call(tool_calls: list[dict], tool_name: str) -> dict | None:
    calls = _find_all_calls(tool_calls, tool_name)
    return calls[-1] if calls else None


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
    call = _find_last_call(tool_calls, "get_device_status")
    passed = (
        call is not None
        and call["args"].get("device_type") == "all"
    )
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
    back_door_calls   = [c for c in _find_all_calls(tool_calls, "lock_door")    if c["args"].get("door") == "back"]
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


def _verify_task_14(tool_calls, duration, state):
    # "bump it up by 2 degrees" after setting thermostat to 68 - relative adjustment
    call = _find_last_call(tool_calls, "set_thermostat")
    passed = state["thermostat"]["temperature"] == 70
    return _result(14, "Relative thermostat increase (+2 degrees)", "hard", passed, call, duration)


def _verify_task_15(tool_calls, duration, state):
    # "unlock the first one" after locking front then garage - back-reference
    front_calls = [c for c in _find_all_calls(tool_calls, "lock_door") if c["args"].get("door") == "front"]
    call = front_calls[-1] if front_calls else None
    passed = state["doors"]["front"] == "unlocked"
    return _result(15, "Unlock first door (3-turn back-reference)", "hard", passed, call, duration)


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


TASKS = [
    Task(1,  "Turn on kitchen lights",                      "easy",   "Turn on the kitchen lights",                                        _verify_task_1),
    Task(2,  "Lock the front door",                         "easy",   "Lock the front door",                                               _verify_task_2),
    Task(3,  "Heat house to 72 degrees",                    "easy",   "Heat the house to 72 degrees",                                      _verify_task_3),
    Task(4,  "Get status of all devices",                   "easy",   "What is the current status of all devices?",                        _verify_task_4),
    Task(5,  "Activate movie night scene",                  "medium", "Activate movie night mode",                                         _verify_task_5),
    Task(6,  "Unlock the garage door",                      "medium", "Unlock the garage door",                                            _verify_task_6),
    Task(7,  "Check bedroom light status",                  "medium", "Are the bedroom lights on?",                                        _verify_task_7),
    Task(8,  "Cool house to 74 degrees",                    "medium", "Cool the house down to 74 degrees",                                 _verify_task_8),
    Task(9,  "Away scene via indirect phrasing",            "hard",   "I'm heading out for the day, set the house accordingly",            _verify_task_9),
    Task(10, "Lock back door + off office lights",          "hard",   "Lock the back door and turn off the office lights",                 _verify_task_10),
    Task(11, "Turn on all lights (multi-tool)",             "hard",   "switch on all the lights",                                          _verify_task_11),
    Task(12, "Turn off bedroom light (pronoun reference)",  "hard",   "switch it off",                                                     _verify_task_12,
         history=[
             {"role": "user",      "content": "switch on the bedroom light"},
             {"role": "assistant", "content": "The bedroom light has been turned on."},
         ]),
    Task(13, "Correct bulk action (hallway off)",           "hard",   "actually keep the hallway one off",                                 _verify_task_13,
         history=[
             {"role": "user",      "content": "turn on all the lights"},
             {"role": "assistant", "content": "All lights have been turned on."},
         ]),
    Task(14, "Relative thermostat increase (+2 degrees)",   "hard",   "bump it up by 2 degrees",                                           _verify_task_14,
         history=[
             {"role": "user",      "content": "set the thermostat to 68 degrees"},
             {"role": "assistant", "content": "The thermostat has been set to 68°F."},
         ]),
    Task(15, "Unlock first door (3-turn back-reference)",   "hard",   "unlock the first one",                                              _verify_task_15,
         history=[
             {"role": "user",      "content": "lock the front door"},
             {"role": "assistant", "content": "The front door has been locked."},
             {"role": "user",      "content": "and the garage too"},
             {"role": "assistant", "content": "The garage door has been locked."},
         ]),
]
