"""
benchmark2/tasks.py
-------------------
100 tasks designed around a principled 3-dimensional taxonomy.

Dimension 1 - Capability (7 categories):
  lights       toggle_lights
  thermostat   set_thermostat
  doors        lock_door
  status       get_device_status
  scene        set_scene
  rejection    intent_unclear
  multi_tool   2+ tools in one turn

Dimension 2 - Phrasing (4 styles):
  imperative   direct command
  colloquial   casual / informal
  implicit     context-driven
  question     interrogative

Dimension 3 - Inference Depth (3 levels):
  literal      words map 1:1 to tool + args
  semantic     meaning is clear but requires translation
  boundary     model must recognise it cannot fulfil the request
"""

import copy
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: int
    name: str
    capability: str   # lights | thermostat | doors | status | scene | rejection | multi_tool
    phrasing: str     # imperative | colloquial | implicit | question
    depth: str        # literal | semantic | boundary
    prompt: str
    verifier: Callable
    initial_state: dict | None = None   # partial state override applied before each run
    history: list[dict] = field(default_factory=list)


ALL_ROOMS = {"living_room", "bedroom", "kitchen", "bathroom", "office", "hallway"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_all_calls(tool_calls: list[dict], tool_name: str) -> list[dict]:
    return [c for c in tool_calls if c["name"] == tool_name]


def _find_last_call(tool_calls: list[dict], tool_name: str) -> dict | None:
    calls = _find_all_calls(tool_calls, tool_name)
    return calls[-1] if calls else None


def _deep_get(d: dict, dotted_path: str):
    """Retrieve a nested value using a dotted key path, e.g. 'lights.kitchen.state'."""
    for key in dotted_path.split("."):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


# ---------------------------------------------------------------------------
# TaskResult (imported by run.py)
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: int
    name: str
    capability: str
    phrasing: str
    depth: str
    passed: bool
    tool_called: bool
    args_correct: bool
    duration_s: float


# ---------------------------------------------------------------------------
# Verifier factory helpers
# ---------------------------------------------------------------------------

def _state_result(task_id, name, capability, phrasing, depth, passed, tool_calls, duration):
    return TaskResult(
        task_id=task_id,
        name=name,
        capability=capability,
        phrasing=phrasing,
        depth=depth,
        passed=passed,
        tool_called=bool(tool_calls),
        args_correct=passed,
        duration_s=duration,
    )


def _tool_result(task_id, name, capability, phrasing, depth, passed, tool_called, args_correct, duration):
    return TaskResult(
        task_id=task_id,
        name=name,
        capability=capability,
        phrasing=phrasing,
        depth=depth,
        passed=passed,
        tool_called=tool_called,
        args_correct=args_correct,
        duration_s=duration,
    )


# ---------------------------------------------------------------------------
# Factory: lights
# ---------------------------------------------------------------------------

def _make_lights_task(tid, name, phrasing, depth, prompt, room, expected_state, initial_state=None):
    def verifier(tool_calls, duration, state):
        passed = state["lights"][room]["state"] == expected_state
        return _state_result(tid, name, "lights", phrasing, depth, passed, tool_calls, duration)
    return Task(
        id=tid, name=name, capability="lights", phrasing=phrasing, depth=depth,
        prompt=prompt, verifier=verifier, initial_state=initial_state,
    )


# ---------------------------------------------------------------------------
# Factory: thermostat
# ---------------------------------------------------------------------------

def _make_thermostat_task(tid, name, phrasing, depth, prompt, temperature, mode, initial_state=None):
    def verifier(tool_calls, duration, state):
        passed = (
            state["thermostat"]["temperature"] == temperature
            and state["thermostat"]["mode"] == mode
        )
        return _state_result(tid, name, "thermostat", phrasing, depth, passed, tool_calls, duration)
    return Task(
        id=tid, name=name, capability="thermostat", phrasing=phrasing, depth=depth,
        prompt=prompt, verifier=verifier, initial_state=initial_state,
    )


def _make_thermostat_mode_task(tid, name, phrasing, depth, prompt, expected_mode, initial_state=None):
    """Verifies only the mode (semantic tasks where the exact temp is flexible)."""
    def verifier(tool_calls, duration, state):
        passed = state["thermostat"]["mode"] == expected_mode
        return _state_result(tid, name, "thermostat", phrasing, depth, passed, tool_calls, duration)
    return Task(
        id=tid, name=name, capability="thermostat", phrasing=phrasing, depth=depth,
        prompt=prompt, verifier=verifier, initial_state=initial_state,
    )


# ---------------------------------------------------------------------------
# Factory: doors
# ---------------------------------------------------------------------------

def _make_door_task(tid, name, phrasing, depth, prompt, door, expected_state, initial_state=None):
    def verifier(tool_calls, duration, state):
        passed = state["doors"][door] == expected_state
        return _state_result(tid, name, "doors", phrasing, depth, passed, tool_calls, duration)
    return Task(
        id=tid, name=name, capability="doors", phrasing=phrasing, depth=depth,
        prompt=prompt, verifier=verifier, initial_state=initial_state,
    )


# ---------------------------------------------------------------------------
# Factory: status
# ---------------------------------------------------------------------------

def _make_status_task(tid, name, phrasing, depth, prompt, device_type, room=None):
    def verifier(tool_calls, duration, state):
        calls = _find_all_calls(tool_calls, "get_device_status")
        if room is not None:
            match = next(
                (c for c in calls
                 if c["args"].get("device_type") in {device_type, "all"}
                 and c["args"].get("room") == room),
                None,
            )
        else:
            match = next(
                (c for c in calls
                 if c["args"].get("device_type") in {device_type, "all"}),
                None,
            )
        tool_called = bool(calls)
        args_correct = match is not None
        passed = args_correct
        return _tool_result(tid, name, "status", phrasing, depth, passed, tool_called, args_correct, duration)
    return Task(
        id=tid, name=name, capability="status", phrasing=phrasing, depth=depth,
        prompt=prompt, verifier=verifier,
    )


# ---------------------------------------------------------------------------
# Factory: scene
# ---------------------------------------------------------------------------

def _make_scene_task(tid, name, phrasing, depth, prompt, scene_name, initial_state=None):
    def verifier(tool_calls, duration, state):
        passed = state["active_scene"] == scene_name
        return _state_result(tid, name, "scene", phrasing, depth, passed, tool_calls, duration)
    return Task(
        id=tid, name=name, capability="scene", phrasing=phrasing, depth=depth,
        prompt=prompt, verifier=verifier, initial_state=initial_state,
    )


# ---------------------------------------------------------------------------
# Factory: rejection
# ---------------------------------------------------------------------------

def _make_rejection_task(tid, name, phrasing, depth, prompt, reason):
    def verifier(tool_calls, duration, state):
        call = _find_last_call(tool_calls, "intent_unclear")
        tool_called = bool(tool_calls)
        args_correct = call is not None and call["args"].get("reason") == reason
        passed = args_correct
        return _tool_result(tid, name, "rejection", phrasing, depth, passed, tool_called, args_correct, duration)
    return Task(
        id=tid, name=name, capability="rejection", phrasing=phrasing, depth=depth,
        prompt=prompt, verifier=verifier,
    )


# ---------------------------------------------------------------------------
# Factory: multi_tool
# ---------------------------------------------------------------------------

def _make_multi_task(tid, name, phrasing, depth, prompt, expected: dict, initial_state=None):
    """
    expected: dict of dotted-path assertions checked against final state.
    Example: {"lights.kitchen.state": "on", "doors.front": "locked"}
    """
    def verifier(tool_calls, duration, state):
        passed = all(_deep_get(state, path) == val for path, val in expected.items())
        return _state_result(tid, name, "multi_tool", phrasing, depth, passed, tool_calls, duration)
    return Task(
        id=tid, name=name, capability="multi_tool", phrasing=phrasing, depth=depth,
        prompt=prompt, verifier=verifier, initial_state=initial_state,
    )


# ---------------------------------------------------------------------------
# Task definitions (100 total)
# ---------------------------------------------------------------------------

TASKS: list[Task] = [

    # =========================================================================
    # LIGHTS (24 tasks: IDs 1-24)
    # imperative(8), colloquial(6), implicit(6), question(4)
    # literal(14), semantic(6), boundary(4)
    # =========================================================================

    # --- imperative / literal (6) ---
    _make_lights_task(1,  "Kitchen lights on (imp/lit)",       "imperative", "literal",  "Turn on the kitchen lights",              "kitchen",     "on"),
    _make_lights_task(2,  "Bedroom lights off (imp/lit)",      "imperative", "literal",  "Turn off the bedroom lights",             "bedroom",     "off"),
    _make_lights_task(3,  "Hallway light on (imp/lit)",        "imperative", "literal",  "Switch on the hallway light",             "hallway",     "on"),
    _make_lights_task(4,  "Office lights off (imp/lit)",       "imperative", "literal",  "Turn off the office lights",              "office",      "off"),
    _make_lights_task(5,  "Bathroom light on (imp/lit)",       "imperative", "literal",  "Turn on the bathroom light",              "bathroom",    "on"),
    _make_lights_task(6,  "Living room lights off (imp/lit)",  "imperative", "literal",  "Turn off the living room lights",         "living_room", "off"),

    # --- imperative / semantic (2) ---
    _make_lights_task(7,  "Bedroom lights on for reading (imp/sem)",  "imperative", "semantic", "Get the bedroom lights ready for reading",  "bedroom",  "on"),
    _make_lights_task(8,  "Kitchen lights off after dinner (imp/sem)", "imperative", "semantic", "Shut the kitchen down for the night",       "kitchen",  "off",
                      initial_state={"lights": {"kitchen": {"state": "on"}}}),

    # --- colloquial / literal (4) ---
    _make_lights_task(9,  "Living room lights on (coll/lit)",  "colloquial", "literal",  "Living room lights on please",            "living_room", "on"),
    _make_lights_task(10, "Bathroom light off (coll/lit)",     "colloquial", "literal",  "Bathroom light off",                      "bathroom",    "off"),
    _make_lights_task(11, "Office lights on (coll/lit)",       "colloquial", "literal",  "Office lights, turn them on",             "office",      "on"),
    _make_lights_task(12, "Hallway light off (coll/lit)",      "colloquial", "literal",  "Kill the hallway light",                  "hallway",     "off"),

    # --- colloquial / semantic (2) ---
    _make_lights_task(13, "Bedroom lights cozy (coll/sem)",    "colloquial", "semantic", "Make the bedroom nice and cozy",          "bedroom",     "on"),
    _make_lights_task(14, "Kitchen lights waste (coll/sem)",   "colloquial", "semantic", "The kitchen light is wasting electricity", "kitchen",    "off",
                      initial_state={"lights": {"kitchen": {"state": "on"}}}),

    # --- implicit / literal (4) ---
    _make_lights_task(15, "Office lights on implicit (impl/lit)",    "implicit", "literal",  "I'm starting work in the office",         "office",      "on"),
    _make_lights_task(16, "Bedroom lights on implicit (impl/lit)",   "implicit", "literal",  "I'm heading to the bedroom to read",      "bedroom",     "on"),
    _make_lights_task(17, "Hallway light on implicit (impl/lit)",    "implicit", "literal",  "I can't see in the hallway",              "hallway",     "on"),
    _make_lights_task(18, "Bathroom light off implicit (impl/lit)",  "implicit", "literal",  "I just left the bathroom",                "bathroom",    "off",
                      initial_state={"lights": {"bathroom": {"state": "on"}}}),

    # --- implicit / semantic (2) ---
    _make_lights_task(19, "Living room lights dark (impl/sem)",  "implicit", "semantic", "It's a bit dark in here",              "living_room", "on"),
    _make_lights_task(20, "Office lights done for day (impl/sem)", "implicit", "semantic", "I'm done for the day in the office",  "office",      "off",
                      initial_state={"lights": {"office": {"state": "on"}}}),

    # --- question / literal (4) ---
    _make_lights_task(21, "Kitchen lights on (q/lit)",       "question", "literal",  "Can you turn on the kitchen lights?",         "kitchen",     "on"),
    _make_lights_task(22, "Bedroom lights off (q/lit)",      "question", "literal",  "Could you switch off the bedroom lights?",    "bedroom",     "off"),
    _make_lights_task(23, "Hallway light on (q/lit)",        "question", "literal",  "Would you mind turning on the hallway light?", "hallway",    "on"),
    _make_lights_task(24, "Bathroom light off (q/lit)",      "question", "literal",  "Can you turn off the bathroom light?",        "bathroom",    "off",
                      initial_state={"lights": {"bathroom": {"state": "on"}}}),


    # =========================================================================
    # THERMOSTAT (16 tasks: IDs 25-40)
    # imperative(5), colloquial(4), implicit(4), question(3)
    # literal(8), semantic(5), boundary(3)
    # =========================================================================

    # --- imperative / literal (4) ---
    _make_thermostat_task(25, "Heat to 72 (imp/lit)",     "imperative", "literal", "Heat the house to 72 degrees",           72, "heat"),
    _make_thermostat_task(26, "Cool to 76 (imp/lit)",     "imperative", "literal", "Cool the house down to 76 degrees",      76, "cool"),
    _make_thermostat_task(27, "Set 70 auto (imp/lit)",    "imperative", "literal", "Set the thermostat to 70 in auto mode",  70, "auto"),
    _make_thermostat_task(28, "Heat to 68 (imp/lit)",     "imperative", "literal", "Set heating to 68 degrees",              68, "heat"),

    # --- imperative / semantic (1) ---
    _make_thermostat_mode_task(29, "Thermostat warm imp (imp/sem)", "imperative", "semantic",
                               "Make it a bit warmer in here", "heat"),

    # --- colloquial / literal (2) ---
    _make_thermostat_task(30, "AC to 74 (coll/lit)",      "colloquial", "literal", "AC to 74",                               74, "cool"),
    _make_thermostat_task(31, "Heat 65 coll (coll/lit)",  "colloquial", "literal", "65 degrees heat mode",                   65, "heat"),

    # --- colloquial / semantic (2) ---
    _make_thermostat_mode_task(32, "Way too hot coll (coll/sem)", "colloquial", "semantic",
                               "It's way too hot in here", "cool"),
    _make_thermostat_mode_task(33, "Freezing coll (coll/sem)", "colloquial", "semantic",
                               "It's freezing, warm it up", "heat"),

    # --- implicit / literal (2) ---
    _make_thermostat_task(34, "Set 73 heat (impl/lit)",   "implicit", "literal", "I need the house at 73 degrees heat",    73, "heat"),
    _make_thermostat_task(35, "Set 77 cool (impl/lit)",   "implicit", "literal", "The house needs to be cooled to 77",     77, "cool"),

    # --- implicit / semantic (2) ---
    _make_thermostat_mode_task(36, "Summer implicit (impl/sem)", "implicit", "semantic",
                               "It feels like a sauna in here", "cool"),
    _make_thermostat_mode_task(37, "Winter implicit (impl/sem)", "implicit", "semantic",
                               "I can see my breath it's so cold", "heat"),

    # --- question / literal (3) ---
    _make_thermostat_task(38, "Set 71 heat q (q/lit)",    "question", "literal", "Can you set the heat to 71 degrees?",    71, "heat"),
    _make_thermostat_task(39, "Cool to 75 q (q/lit)",     "question", "literal", "Could you cool the house to 75?",        75, "cool"),
    _make_thermostat_task(40, "Set 69 auto q (q/lit)",    "question", "literal", "Can you put the thermostat on auto at 69?", 69, "auto"),


    # =========================================================================
    # DOORS (16 tasks: IDs 41-56)
    # imperative(5), colloquial(4), implicit(4), question(3)
    # literal(8), semantic(5), boundary(3)
    # =========================================================================

    # --- imperative / literal (4) ---
    _make_door_task(41, "Lock front door (imp/lit)",    "imperative", "literal", "Lock the front door",          "front",  "locked"),
    _make_door_task(42, "Unlock back door (imp/lit)",   "imperative", "literal", "Unlock the back door",         "back",   "unlocked"),
    _make_door_task(43, "Lock garage door (imp/lit)",   "imperative", "literal", "Lock the garage door",         "garage", "locked"),
    _make_door_task(44, "Unlock side door (imp/lit)",   "imperative", "literal", "Unlock the side door",         "side",   "unlocked"),

    # --- imperative / semantic (1) ---
    _make_door_task(45, "Secure garage (imp/sem)",      "imperative", "semantic", "Make sure the garage is secure",  "garage", "locked"),

    # --- colloquial / literal (2) ---
    _make_door_task(46, "Front door locked coll (coll/lit)", "colloquial", "literal", "Front door, lock it",          "front",  "locked"),
    _make_door_task(47, "Back door open coll (coll/lit)",    "colloquial", "literal", "Back door open please",        "back",   "unlocked"),

    # --- colloquial / semantic (2) ---
    _make_door_task(48, "Side door safe coll (coll/sem)",   "colloquial", "semantic", "Side door, keep it safe",     "side",   "locked"),
    _make_door_task(49, "Garage let in coll (coll/sem)",    "colloquial", "semantic", "Let me into the garage",      "garage", "unlocked"),

    # --- implicit / literal (2) ---
    _make_door_task(50, "Lock front implicit (impl/lit)",   "implicit", "literal", "I need the front door locked",  "front",  "locked"),
    _make_door_task(51, "Unlock back implicit (impl/lit)",  "implicit", "literal", "The delivery is at the back",   "back",   "unlocked"),

    # --- implicit / semantic (2) ---
    _make_door_task(52, "Side door leaving (impl/sem)",     "implicit", "semantic", "I'm leaving through the side", "side",   "unlocked",
                    initial_state={"doors": {"side": "locked"}}),
    _make_door_task(53, "Garage secure bedtime (impl/sem)", "implicit", "semantic", "I'm heading to bed, garage needs to be safe", "garage", "locked"),

    # --- question / literal (3) ---
    _make_door_task(54, "Lock front q (q/lit)",      "question", "literal", "Can you lock the front door?",         "front",  "locked"),
    _make_door_task(55, "Unlock garage q (q/lit)",   "question", "literal", "Could you unlock the garage door?",    "garage", "unlocked"),
    _make_door_task(56, "Lock side q (q/lit)",       "question", "literal", "Would you lock the side door please?", "side",   "locked"),


    # =========================================================================
    # STATUS (10 tasks: IDs 57-66)
    # imperative(3), question(5), colloquial(2)
    # literal(6), semantic(4)
    # =========================================================================

    # --- imperative / literal (3) ---
    _make_status_task(57, "Check all status (imp/lit)",        "imperative", "literal", "Check the status of all devices",             "all"),
    _make_status_task(58, "Check thermostat status (imp/lit)", "imperative", "literal", "Check the thermostat status",                 "thermostat"),
    _make_status_task(59, "Check door status (imp/lit)",       "imperative", "literal", "Check the status of all doors",               "door"),

    # --- question / literal (3) ---
    _make_status_task(60, "Bedroom lights on? (q/lit)",  "question", "literal", "Are the bedroom lights on?",             "lights",     "bedroom"),
    _make_status_task(61, "Front door locked? (q/lit)",  "question", "literal", "Is the front door locked?",              "door",       "front"),
    _make_status_task(62, "Thermostat set to? (q/lit)",  "question", "literal", "What is the thermostat set to?",         "thermostat"),

    # --- question / semantic (2) ---
    _make_status_task(63, "House secure? (q/sem)",       "question", "semantic", "Is the house secure right now?",         "door"),
    _make_status_task(64, "Lights check q (q/sem)",      "question", "semantic", "Which lights are currently running?",    "lights"),

    # --- colloquial / literal (2) ---
    _make_status_task(65, "Thermostat status coll (coll/lit)", "colloquial", "literal", "Thermostat status",             "thermostat"),
    _make_status_task(66, "Door status coll (coll/lit)",       "colloquial", "literal", "Door status please",            "door"),


    # =========================================================================
    # SCENE (10 tasks: IDs 67-76)
    # imperative(5), implicit(5)
    # literal(5), semantic(5)
    # =========================================================================

    # --- imperative / literal (5) ---
    _make_scene_task(67, "Movie night scene (imp/lit)",  "imperative", "literal", "Activate movie night mode",          "movie_night"),
    _make_scene_task(68, "Bedtime scene (imp/lit)",      "imperative", "literal", "Activate bedtime mode",              "bedtime"),
    _make_scene_task(69, "Morning scene (imp/lit)",      "imperative", "literal", "Activate the morning scene",         "morning"),
    _make_scene_task(70, "Away scene (imp/lit)",         "imperative", "literal", "Activate the away mode",             "away"),
    _make_scene_task(71, "Party scene (imp/lit)",        "imperative", "literal", "Activate party mode",                "party"),

    # --- implicit / semantic (5) ---
    _make_scene_task(72, "Movie night implicit (impl/sem)", "implicit", "semantic",
                     "We're about to watch a film, set the mood",          "movie_night"),
    _make_scene_task(73, "Bedtime implicit (impl/sem)",     "implicit", "semantic",
                     "I'm going to sleep now, get the house ready",        "bedtime"),
    _make_scene_task(74, "Morning implicit (impl/sem)",     "implicit", "semantic",
                     "Good morning, get the house ready for the day",      "morning"),
    _make_scene_task(75, "Away implicit (impl/sem)",        "implicit", "semantic",
                     "I'm heading out for the day, set the house accordingly", "away"),
    _make_scene_task(76, "Party implicit (impl/sem)",       "implicit", "semantic",
                     "We're having people over, set the mood",             "party"),


    # =========================================================================
    # REJECTION (12 tasks: IDs 77-88)
    # imperative(3), colloquial(4), question(3), implicit(2)
    # boundary(12)
    # =========================================================================

    # --- imperative / boundary (3) ---
    _make_rejection_task(77, "Reject: dim lights unsupported (imp/bnd)",   "imperative", "boundary",
                         "Dim the living room lights to 30%",               "unsupported_device"),
    _make_rejection_task(78, "Reject: patio lights unsupported (imp/bnd)", "imperative", "boundary",
                         "Turn on the patio lights",                        "unsupported_device"),
    _make_rejection_task(79, "Reject: 95 degree temp (imp/bnd)",           "imperative", "boundary",
                         "Set the temperature to 95 degrees",               "unsupported_device"),

    # --- colloquial / boundary (4) ---
    _make_rejection_task(80, "Reject: order pizza (coll/bnd)",    "colloquial", "boundary",
                         "Order a pizza for me",                            "off_topic"),
    _make_rejection_task(81, "Reject: car lock (coll/bnd)",       "colloquial", "boundary",
                         "Lock the car",                                    "off_topic"),
    _make_rejection_task(82, "Reject: turn it on (coll/bnd)",     "colloquial", "boundary",
                         "Turn it on",                                      "incomplete"),
    _make_rejection_task(83, "Reject: make nicer (coll/bnd)",     "colloquial", "boundary",
                         "Make it nicer in here",                           "ambiguous"),

    # --- question / boundary (3) ---
    _make_rejection_task(84, "Reject: weather query (q/bnd)",     "question",   "boundary",
                         "What's the weather like outside?",                "off_topic"),
    _make_rejection_task(85, "Reject: color change (q/bnd)",      "question",   "boundary",
                         "Can you change the light color to blue?",         "unsupported_device"),
    _make_rejection_task(86, "Reject: which thing? (q/bnd)",      "question",   "boundary",
                         "Can you turn the thing off?",                     "incomplete"),

    # --- implicit / boundary (2) ---
    _make_rejection_task(87, "Reject: ambiguous comfy (impl/bnd)", "implicit",  "boundary",
                         "Make it more comfortable in here",                "ambiguous"),
    _make_rejection_task(88, "Reject: music request (impl/bnd)",   "implicit",  "boundary",
                         "I want some background music",                    "off_topic"),


    # =========================================================================
    # MULTI_TOOL (12 tasks: IDs 89-100)
    # imperative(5), colloquial(4), implicit(3)
    # literal(6), semantic(6)
    # =========================================================================

    # --- imperative / literal (4) ---
    _make_multi_task(89, "Lock front + kitchen on (imp/lit)", "imperative", "literal",
                     "Lock the front door and turn on the kitchen lights",
                     {"doors.front": "locked", "lights.kitchen.state": "on"}),
    _make_multi_task(90, "Thermostat heat + office off (imp/lit)", "imperative", "literal",
                     "Set the heat to 72 and turn off the office lights",
                     {"thermostat.temperature": 72, "thermostat.mode": "heat", "lights.office.state": "off"}),
    _make_multi_task(91, "Unlock back + hallway on (imp/lit)", "imperative", "literal",
                     "Unlock the back door and turn on the hallway light",
                     {"doors.back": "unlocked", "lights.hallway.state": "on"}),
    _make_multi_task(92, "Lock garage + bedroom off (imp/lit)", "imperative", "literal",
                     "Lock the garage door and turn off the bedroom lights",
                     {"doors.garage": "locked", "lights.bedroom.state": "off"},
                     initial_state={"lights": {"bedroom": {"state": "on"}}}),

    # --- imperative / semantic (1) ---
    _make_multi_task(93, "Secure and prep for night (imp/sem)", "imperative", "semantic",
                     "Lock up and get the bedroom lights ready for sleep",
                     {"doors.front": "locked", "lights.bedroom.state": "on"}),

    # --- colloquial / literal (2) ---
    _make_multi_task(94, "Front locked + living on coll (coll/lit)", "colloquial", "literal",
                     "Front door locked and living room lights on please",
                     {"doors.front": "locked", "lights.living_room.state": "on"}),
    _make_multi_task(95, "Kitchen off + back locked coll (coll/lit)", "colloquial", "literal",
                     "Kitchen lights off and back door locked",
                     {"lights.kitchen.state": "off", "doors.back": "locked"},
                     initial_state={"lights": {"kitchen": {"state": "on"}}}),

    # --- colloquial / semantic (2) ---
    _make_multi_task(96, "Cozy evening setup (coll/sem)", "colloquial", "semantic",
                     "Let's get cozy, living room lights and lock up",
                     {"lights.living_room.state": "on", "doors.front": "locked"}),
    _make_multi_task(97, "Morning routine coll (coll/sem)", "colloquial", "semantic",
                     "Morning! Lights and unlock the front",
                     {"lights.living_room.state": "on", "doors.front": "unlocked"}),

    # --- implicit / literal (2) ---
    _make_multi_task(98, "Back door + hallway bedtime (impl/lit)", "implicit", "literal",
                     "I'm going to bed, lock the back door and turn off the hallway light",
                     {"doors.back": "locked", "lights.hallway.state": "off"},
                     initial_state={"lights": {"hallway": {"state": "on"}}}),
    _make_multi_task(99, "Office done for the day (impl/lit)", "implicit", "literal",
                     "I'm done in the office, turn off the lights and lock the side door",
                     {"lights.office.state": "off", "doors.side": "locked"},
                     initial_state={"lights": {"office": {"state": "on"}}}),

    # --- implicit / semantic (1) ---
    _make_multi_task(100, "Leaving the house (impl/sem)", "implicit", "semantic",
                     "I'm heading out, make sure the house is ready",
                     {"doors.front": "locked", "doors.back": "locked"}),
]


assert len(TASKS) == 100, f"Expected 100 tasks, got {len(TASKS)}"
