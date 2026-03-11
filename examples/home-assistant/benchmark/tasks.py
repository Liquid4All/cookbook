from dataclasses import dataclass
from typing import Callable


@dataclass
class Task:
    id: int
    name: str
    difficulty: str
    prompt: str
    verifier: Callable


ALL_ROOMS = {"living_room", "bedroom", "kitchen", "bathroom", "office", "hallway"}


def _find_call(tool_calls: list[dict], tool_name: str) -> dict | None:
    for call in tool_calls:
        if call["name"] == tool_name:
            return call
    return None


def _find_all_calls(tool_calls: list[dict], tool_name: str) -> list[dict]:
    return [call for call in tool_calls if call["name"] == tool_name]


def _verify_task_1(tool_calls, duration):
    call = _find_call(tool_calls, "toggle_lights")
    passed = (
        call is not None
        and call["args"].get("room") == "kitchen"
        and call["args"].get("state") == "on"
    )
    return _result(1, "Turn on kitchen lights", "easy", passed, call, duration)


def _verify_task_2(tool_calls, duration):
    call = _find_call(tool_calls, "lock_door")
    passed = (
        call is not None
        and call["args"].get("door") == "front"
        and call["args"].get("state") == "lock"
    )
    return _result(2, "Lock the front door", "easy", passed, call, duration)


def _verify_task_3(tool_calls, duration):
    call = _find_call(tool_calls, "set_thermostat")
    passed = (
        call is not None
        and call["args"].get("temperature") == 72
        and call["args"].get("mode") == "heat"
    )
    return _result(3, "Heat house to 72 degrees", "easy", passed, call, duration)


def _verify_task_4(tool_calls, duration):
    call = _find_call(tool_calls, "get_device_status")
    passed = (
        call is not None
        and call["args"].get("device_type") == "all"
    )
    return _result(4, "Get status of all devices", "easy", passed, call, duration)


def _verify_task_5(tool_calls, duration):
    call = _find_call(tool_calls, "set_scene")
    passed = (
        call is not None
        and call["args"].get("scene") == "movie_night"
    )
    return _result(5, "Activate movie night scene", "medium", passed, call, duration)


def _verify_task_6(tool_calls, duration):
    call = _find_call(tool_calls, "lock_door")
    passed = (
        call is not None
        and call["args"].get("door") == "garage"
        and call["args"].get("state") == "unlock"
    )
    return _result(6, "Unlock the garage door", "medium", passed, call, duration)


def _verify_task_7(tool_calls, duration):
    call = _find_call(tool_calls, "get_device_status")
    passed = (
        call is not None
        and call["args"].get("device_type") == "lights"
        and call["args"].get("room") == "bedroom"
    )
    return _result(7, "Check bedroom light status", "medium", passed, call, duration)


def _verify_task_8(tool_calls, duration):
    call = _find_call(tool_calls, "set_thermostat")
    passed = (
        call is not None
        and call["args"].get("temperature") == 74
        and call["args"].get("mode") == "cool"
    )
    return _result(8, "Cool house to 74 degrees", "medium", passed, call, duration)


def _verify_task_9(tool_calls, duration):
    # Requires inferring "heading out for the day" -> scene="away"
    call = _find_call(tool_calls, "set_scene")
    passed = (
        call is not None
        and call["args"].get("scene") == "away"
    )
    return _result(9, "Away scene via indirect phrasing", "hard", passed, call, duration)


def _verify_task_10(tool_calls, duration):
    # Requires calling two tools in one turn
    door_call   = _find_call(tool_calls, "lock_door")
    lights_call = _find_call(tool_calls, "toggle_lights")
    passed = (
        door_call is not None
        and door_call["args"].get("door") == "back"
        and door_call["args"].get("state") == "lock"
        and lights_call is not None
        and lights_call["args"].get("room") == "office"
        and lights_call["args"].get("state") == "off"
    )
    call = door_call or lights_call
    return _result(10, "Lock back door + off office lights (multi-tool)", "hard", passed, call, duration)


def _verify_task_11(tool_calls, duration):
    # Requires calling toggle_lights once per room, all with state="on"
    calls = _find_all_calls(tool_calls, "toggle_lights")
    rooms_on = {
        c["args"]["room"]
        for c in calls
        if c["args"].get("state") == "on" and "room" in c["args"]
    }
    passed = rooms_on == ALL_ROOMS
    call = calls[0] if calls else None
    return _result(11, "Turn on all lights (multi-tool)", "hard", passed, call, duration)


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
]
