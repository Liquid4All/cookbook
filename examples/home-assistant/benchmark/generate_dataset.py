import sys
import copy
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from app.agent import run_agent, SYSTEM_PROMPT, openai_client
from app.state import home_state, randomize_state
from app.tools.schemas import TOOL_SCHEMAS
from benchmark.tasks import TASKS, _find_last_call, _find_all_calls

_DEFAULT_STATE = copy.deepcopy(home_state)

# Tasks where initial state should be randomized (no fixed history dependency)
_RANDOMIZED_TASKS = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    # New tasks (IDs 20-101): all are single-turn with no fixed history
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52,
    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 83, 84, 85, 86,
    87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101,
}

# ---------------------------------------------------------------------------
# Paraphrases (template-based for tasks 1-8; hand-crafted for complex tasks)
# ---------------------------------------------------------------------------

TASK_PARAPHRASES = {
    1: [
        "Kitchen lights on please",
        "Can you switch on the kitchen lights?",
        "Light up the kitchen",
        "Kitchen light on",
        "Please turn the kitchen lights on",
        "Get the kitchen lights on",
        "Could you put the kitchen lights on?",
        "Switch the kitchen lights on",
        "I'd like the kitchen lights on",
        "Kitchen lights need to be on",
    ],
    2: [
        "Lock up the front door",
        "Front door locked please",
        "Make sure the front door is locked",
        "Can you lock the front door?",
        "Please lock the front door",
        "Secure the front door",
        "Could you lock the front door for me?",
        "Lock front",
        "Make the front door secure",
        "Front door, please lock it",
    ],
    3: [
        "Warm the house up to 72 degrees",
        "72 degrees heating please",
        "Set the heat to 72",
        "Please heat the house to 72",
        "Can you set heating to 72 degrees?",
        "Turn on the heat at 72",
        "I want it at 72, heating mode",
        "Heat mode, 72 degrees",
        "Set thermostat to 72 for heat",
        "Could you warm it up to 72?",
    ],
    4: [
        "What's the status of everything?",
        "Give me a full home status",
        "Show me all device states",
        "What are all my devices doing?",
        "Status of all devices please",
        "Can you check all the devices?",
        "What's going on with everything?",
        "Give me a summary of the house",
        "Check everything for me",
        "Full status report",
    ],
    5: [
        "Enable movie night",
        "Set up for movie night",
        "Put on movie night",
        "Activate movie night please",
        "Can you put on movie night mode?",
        "Start movie night",
        "Movie night time",
        "I want movie night mode",
        "Switch to movie night",
        "Set the movie night scene",
    ],
    6: [
        "Unlock garage",
        "The garage door should be unlocked",
        "Open the garage door",
        "Can you unlock the garage?",
        "Please unlock the garage door",
        "I need the garage open",
        "Unlock the garage for me",
        "Garage door unlock please",
        "Open up the garage",
        "Let me into the garage",
    ],
    7: [
        "Bedroom light status?",
        "Check if bedroom lights are on",
        "Are the bedroom lights on?",
        "What's the status of the bedroom lights?",
        "Is the bedroom light on?",
        "Can you check the bedroom lights?",
        "Tell me about the bedroom lights",
        "Status of bedroom lighting",
        "Are the bedroom lights currently on?",
        "Check bedroom lights for me",
    ],
    8: [
        "Set AC to 74 degrees",
        "I want it cooler, 74 degrees",
        "Cool it down to 74",
        "Set cooling to 74",
        "Can you set the AC to 74?",
        "74 degrees cooling mode",
        "I need it cooled to 74",
        "Cool the house to 74 degrees",
        "Set the temperature to 74 cool",
        "AC at 74 please",
    ],
    9: [
        "I'm leaving the house for the day",
        "I'm going out, set up the house",
        "Leaving now, prepare the house",
        "Set everything up for me leaving",
        "I'm heading out",
    ],
    10: [
        "Lock the back door and switch off the office lights",
        "Back door locked and office lights off please",
        "Secure the back door, kill the office lights",
        "Back door lock and office lights out",
    ],
    11: [
        "Turn all lights on",
        "All lights on please",
        "Switch on every light in the house",
        "Lights on everywhere",
        "Can you turn on all the lights?",
    ],
    12: [
        "Turn it off",
        "Off please",
        "Switch it off please",
        "Can you turn it off?",
        "Please switch it off",
    ],
    13: [
        "Actually leave the hallway light off",
        "Keep hallway dark",
        "Hallway light should stay off",
        "Don't turn on the hallway light",
        "Actually the hallway one should be off",
    ],
    14: [
        "Make it 2 degrees warmer",
        "Increase temperature by 2",
        "Bump the heat up 2 degrees",
        "Add 2 degrees to the thermostat",
        "Raise it 2 degrees please",
    ],
    15: [
        "Unlock the first door",
        "Undo the first one",
        "The first door, unlock it",
        "Go back and unlock the first one",
        "Unlock the one I mentioned first",
    ],
    16: [
        "Set the lights to half brightness",
        "Can you dim the bedroom lights?",
        "Lower the brightness to 40%",
        "Dim the kitchen lights please",
        "Reduce the hallway lights to 30%",
        "Make the lights dimmer",
        "Can you dim the lights in the living room?",
    ],
    17: [
        "Can you order some food?",
        "I want to order a pizza",
        "Call a cab for me",
        "Book a restaurant for tonight",
        "Play some music please",
        "What's the weather like?",
        "Set a timer for 30 minutes",
    ],
    18: [
        "Switch it on",
        "Can you put it on?",
        "Turn that on",
        "Switch on the thing",
        "I need it on",
        "Turn that device on",
    ],
    19: [
        "I want it to be comfortable",
        "Set it up nicely",
        "Make the temperature perfect",
        "Can you adjust it?",
        "Make the house feel better",
        "Fix the temperature",
    ],
    # --- toggle_lights: on, new rooms ---
    20: [
        "Living room lights on",
        "Switch on the living room lights",
        "Could you turn on the living room lights?",
        "Please put the living room lights on",
        "Living room needs lights",
    ],
    21: [
        "Turn on living room lights",
        "Can you put the living room lights on?",
        "Living room light on",
        "Switch living room lights on please",
        "Get the living room lights on",
    ],
    22: [
        "Turn on the living room lights",
        "Living room lights on please",
        "Switch on living room lights",
        "Living room light on",
        "Please turn on the living room light",
    ],
    23: [
        "Bedroom lights on",
        "Switch on the bedroom lights",
        "Could you turn on the bedroom lights?",
        "Please put the bedroom lights on",
        "Get the bedroom lights on",
    ],
    24: [
        "Turn on the bedroom lights",
        "Can you switch on the bedroom lights?",
        "Bedroom light on",
        "Please turn on the bedroom light",
        "Switch the bedroom lights on",
    ],
    25: [
        "Turn on the bedroom lights",
        "Bedroom lights on please",
        "Switch on bedroom lights",
        "Bedroom light on",
        "Please turn on the bedroom light",
    ],
    26: [
        "Bathroom lights on",
        "Switch on the bathroom light",
        "Please turn on the bathroom light",
        "Get the bathroom light on",
        "Bathroom light on please",
    ],
    27: [
        "Turn on the bathroom light",
        "Bathroom lights on",
        "Switch on bathroom light",
        "Can you turn on the bathroom light?",
        "Please put the bathroom light on",
    ],
    28: [
        "Can you turn the bathroom light on?",
        "The bathroom is dark, turn on the light",
        "I can't see in the bathroom",
        "Bathroom needs more light",
        "Turn on bathroom light please",
    ],
    29: [
        "Hallway lights on",
        "Switch on the hallway light",
        "Please turn on the hallway light",
        "Get the hallway light on",
        "Hallway light on please",
    ],
    30: [
        "Turn on the hallway light",
        "Hallway light on",
        "Switch the hallway light on",
        "Can you turn on the hallway light?",
        "Please put the hallway light on",
    ],
    31: [
        "Turn on the hallway light",
        "Hallway lights on please",
        "Switch on hallway light",
        "Hallway light on",
        "Please turn on the hallway light",
    ],
    32: [
        "Office lights on",
        "Switch on the office lights",
        "Please turn on the office lights",
        "Get the office lights on",
        "Office light on please",
    ],
    33: [
        "Turn on the office lights",
        "Office lights on",
        "Switch the office lights on",
        "Can you turn on the office lights?",
        "Please put the office lights on",
    ],
    34: [
        "Can you turn the office lights on?",
        "Turn on office light, I need to work",
        "Office is too dark",
        "Light up the office please",
        "Office lights on please",
    ],
    # --- toggle_lights: off, all rooms ---
    35: [
        "Kitchen lights off",
        "Switch off the kitchen lights",
        "Please turn off the kitchen light",
        "Kill the kitchen lights",
        "Kitchen light off please",
    ],
    36: [
        "Turn off the kitchen lights",
        "Kitchen lights off",
        "Switch the kitchen lights off",
        "Can you turn off the kitchen lights?",
        "Please put the kitchen lights off",
    ],
    37: [
        "Turn off the kitchen lights",
        "Kitchen lights off please",
        "Switch off kitchen lights",
        "Kitchen light off",
        "Please turn off the kitchen light",
    ],
    38: [
        "Living room lights off",
        "Switch off the living room lights",
        "Please turn off the living room lights",
        "Kill the living room lights",
        "Living room light off please",
    ],
    39: [
        "Turn off the living room lights",
        "Living room lights off",
        "Switch the living room lights off",
        "Can you turn off the living room lights?",
        "Please turn off living room light",
    ],
    40: [
        "Turn off the living room lights",
        "Living room lights off please",
        "Switch off living room lights",
        "The living room is empty, turn off lights",
        "Nobody in the living room, lights off",
    ],
    41: [
        "Bedroom lights off",
        "Switch off the bedroom lights",
        "Please turn off the bedroom lights",
        "Kill the bedroom lights",
        "Bedroom light off please",
    ],
    42: [
        "Turn off the bedroom lights",
        "Bedroom lights off",
        "Switch the bedroom lights off",
        "Can you turn off the bedroom lights?",
        "Please turn off the bedroom light",
    ],
    43: [
        "Turn off the bedroom lights",
        "Bedroom lights off please",
        "Switch off bedroom lights",
        "Bedroom light off",
        "Please turn off the bedroom light",
    ],
    44: [
        "Bathroom light off",
        "Switch off the bathroom light",
        "Please turn off the bathroom light",
        "Kill the bathroom light",
        "Bathroom light off please",
    ],
    45: [
        "Turn off the bathroom light",
        "Bathroom light off",
        "Switch the bathroom light off",
        "Can you turn off the bathroom light?",
        "Please turn off bathroom light",
    ],
    46: [
        "Hallway light off",
        "Switch off the hallway light",
        "Please turn off the hallway light",
        "Kill the hallway light",
        "Hallway light off please",
    ],
    47: [
        "Turn off the hallway light",
        "Hallway light off please",
        "Switch off hallway light",
        "The hallway light is on for no reason, turn it off",
        "Hallway light is unnecessary, switch it off",
    ],
    48: [
        "Office lights off",
        "Switch off the office lights",
        "Please turn off the office lights",
        "Kill the office lights",
        "Office light off please",
    ],
    49: [
        "Turn off the office lights",
        "Office lights off",
        "Switch the office lights off",
        "Can you turn off the office lights?",
        "Please turn off the office light",
    ],
    50: [
        "Turn off the office lights",
        "Office lights off please",
        "Switch off office lights",
        "Office light off",
        "Please turn off the office light",
    ],
    # --- adversarial toggle_lights ---
    51: [
        "It's pretty dark in here",
        "Could use some more light in here",
        "Can't really see anything",
        "Really dark right now",
        "I need some light",
    ],
    52: [
        "I'm done in the bathroom",
        "Just finished in the bathroom",
        "Leaving the bathroom now",
        "Done with the bathroom",
        "I walked out of the bathroom",
    ],
    # --- lock_door: new combos ---
    53: [
        "Please unlock the front door",
        "Open the front door",
        "Front door unlocked please",
        "I need the front door open",
        "Unlock front",
    ],
    54: [
        "Unlock the front door",
        "Can you unlock the front door?",
        "Please unlock the front door",
        "I need the front door unlocked",
        "Open the front door",
    ],
    55: [
        "Unlock the front door",
        "Front door unlock please",
        "Please unlock the front door",
        "Open the front door for me",
        "I need the front door open",
    ],
    56: [
        "Please unlock the back door",
        "Open the back door",
        "Back door unlocked please",
        "I need the back door open",
        "Unlock back door",
    ],
    57: [
        "Unlock the back door",
        "Can you unlock the back door?",
        "Please unlock the back door",
        "I need the back door unlocked",
        "Open the back door",
    ],
    58: [
        "Unlock the back door",
        "Back door unlock please",
        "Please unlock the back door",
        "Open the back door for me",
        "I need the back door open",
    ],
    59: [
        "Lock the garage",
        "Garage door locked please",
        "Please lock the garage door",
        "Secure the garage",
        "Garage door lock",
    ],
    60: [
        "Lock the garage door",
        "Can you lock the garage?",
        "Please lock the garage door",
        "I need the garage locked",
        "Garage door, lock it",
    ],
    61: [
        "Lock the garage door please",
        "Can you lock the garage?",
        "I want the garage secured",
        "The garage should be locked",
        "Make sure the garage door is locked",
    ],
    62: [
        "Lock the side door",
        "Side door locked please",
        "Please lock the side door",
        "Secure the side door",
        "Side door lock",
    ],
    63: [
        "Lock the side door",
        "Can you lock the side door?",
        "Please lock the side door",
        "I need the side door locked",
        "Side door, secure it",
    ],
    64: [
        "Lock the side door",
        "Side door lock please",
        "Please lock the side door",
        "Can you lock the side door for me?",
        "I need the side door locked",
    ],
    65: [
        "Please unlock the side door",
        "Open the side door",
        "Side door unlocked please",
        "I need the side door open",
        "Unlock side door",
    ],
    66: [
        "Unlock the side door",
        "Can you unlock the side door?",
        "Please unlock the side door",
        "I need the side door unlocked",
        "Open the side door",
    ],
    67: [
        "Unlock the side door",
        "Side door unlock please",
        "Please unlock the side door",
        "Open the side door for me",
        "I need the side door open",
    ],
    68: [
        "Make sure everything is locked up",
        "Lock all the doors before leaving",
        "Secure the house",
        "Everything locked?",
        "Lock up the house",
    ],
    # --- set_thermostat: auto mode + new temps ---
    69: [
        "Auto mode, 70 degrees",
        "Set temperature to 70 auto",
        "70 degrees in automatic mode please",
        "Auto thermostat at 70",
        "Temperature 70, auto mode",
    ],
    70: [
        "Set the thermostat to 70 in auto mode",
        "Auto mode 70 degrees please",
        "70 auto",
        "Temperature 70 automatic",
        "Set auto to 70",
    ],
    71: [
        "Set the thermostat to 70 in auto mode",
        "70 degrees auto please",
        "Auto mode at 70 degrees",
        "Temperature 70, auto",
        "Can you put it on auto at 70?",
    ],
    72: [
        "Warm the house to 65",
        "65 degrees heat please",
        "Set heating to 65",
        "Heat mode at 65 degrees",
        "Can you heat to 65?",
    ],
    73: [
        "Heat the house to 65 degrees",
        "65 heat",
        "Set the heat to 65",
        "Heating to 65 please",
        "65 degrees heating",
    ],
    74: [
        "Cool the house to 78",
        "78 degrees cool please",
        "Set cooling to 78",
        "AC at 78 degrees",
        "Can you cool to 78?",
    ],
    75: [
        "Cool the house to 78 degrees",
        "78 cool",
        "Set the AC to 78",
        "Cooling to 78 please",
        "78 degrees cooling",
    ],
    76: [
        "It's too warm in here",
        "Can you cool things down?",
        "I'm sweating, cool the house",
        "Way too hot, turn on the AC",
        "House is too warm",
    ],
    77: [
        "It's too cold in here",
        "Can you warm things up?",
        "I'm cold, heat the house",
        "Way too chilly, turn on the heat",
        "House is too cold",
    ],
    # --- get_device_status: missing coverage ---
    78: [
        "What's the thermostat at?",
        "Check the thermostat",
        "Tell me the thermostat setting",
        "Thermostat reading?",
        "What temperature is the thermostat set to?",
    ],
    79: [
        "What is the thermostat set to?",
        "Check thermostat status",
        "Tell me the current temperature setting",
        "What's the thermostat showing?",
        "Thermostat check",
    ],
    80: [
        "What is the thermostat set to?",
        "Thermostat reading please",
        "Check the thermostat for me",
        "What's the temperature setting?",
        "Thermostat info",
    ],
    81: [
        "Check the door locks",
        "Are the doors secure?",
        "Tell me which doors are locked",
        "Door lock status?",
        "Which doors are locked right now?",
    ],
    82: [
        "Are all the doors locked?",
        "Check door locks for me",
        "Tell me the door status",
        "Which doors are locked?",
        "Door status check",
    ],
    83: [
        "Are all the doors locked?",
        "Check door status",
        "Door lock check please",
        "Tell me about the doors",
        "Door status",
    ],
    84: [
        "Check the living room lights",
        "What's the living room light status?",
        "Tell me if the living room lights are on",
        "Living room lights status?",
        "Are the living room lights currently on?",
    ],
    85: [
        "Are the kitchen lights on?",
        "Check the kitchen light",
        "What's the status of the kitchen light?",
        "Kitchen light status?",
        "Tell me if the kitchen light is on",
    ],
    86: [
        "Is the bathroom light on?",
        "What's the bathroom light status?",
        "Tell me if the bathroom light is on",
        "Bathroom light status?",
        "Check the bathroom light for me",
    ],
    # --- set_scene: 3 missing scenes ---
    87: [
        "Bedtime mode please",
        "Set bedtime scene",
        "Switch to bedtime mode",
        "Enable bedtime",
        "Activate the bedtime scene",
    ],
    88: [
        "Activate bedtime mode",
        "Bedtime mode on",
        "Set up bedtime",
        "Switch to bedtime scene",
        "Enable the bedtime scene",
    ],
    89: [
        "Time for bed, set everything up",
        "I'm heading to bed",
        "Going to sleep, prepare the house",
        "Bedtime setup please",
        "Set the house for sleeping",
    ],
    90: [
        "Morning scene on",
        "Set up morning mode",
        "Switch to morning scene",
        "Enable morning",
        "Start the morning scene",
    ],
    91: [
        "Activate morning mode",
        "Morning scene on",
        "Set up morning",
        "Switch to morning mode",
        "Enable the morning scene",
    ],
    92: [
        "Start the morning routine",
        "Morning time, set up the house",
        "Just woke up, get things going",
        "Morning setup please",
        "Set the house for morning",
    ],
    93: [
        "Party mode on",
        "Set up party scene",
        "Switch to party mode",
        "Enable party",
        "Start the party scene",
    ],
    94: [
        "Activate party mode",
        "Party scene on",
        "Set up party mode",
        "Switch to party scene",
        "Enable the party scene",
    ],
    95: [
        "Party time, set the mood",
        "Guests are coming, prepare the house",
        "Setting up for a party",
        "Party setup please",
        "Get the house ready for guests",
    ],
    # --- intent_unclear: additional + boundary ---
    96: [
        "Switch on the patio light",
        "Patio lights please",
        "Can you turn on the patio light?",
        "Turn on lights outside on the patio",
        "Patio light on",
    ],
    97: [
        "Set temperature to 100 degrees",
        "I want it at 90 degrees",
        "Set thermostat to 85",
        "Make it 95 degrees",
        "Heat to 110 degrees",
    ],
    98: [
        "Unlock my car",
        "Lock the car doors",
        "Can you lock the car?",
        "Car door unlock please",
        "Secure my vehicle",
    ],
    99: [
        "Is it going to rain today?",
        "What's the forecast?",
        "Will it be sunny?",
        "What's the weather like outside?",
        "Check the weather for me",
    ],
    100: [
        "Make it better in here",
        "Improve things",
        "Set it up right",
        "Can you improve the atmosphere?",
        "Make it more pleasant",
    ],
    101: [
        "Turn the stuff off",
        "Switch off the device",
        "Turn off that thing",
        "Power off the thingamajig",
        "Switch that off",
    ],
}

# ---------------------------------------------------------------------------
# History variants for tasks with pronoun/back-reference resolution
# ---------------------------------------------------------------------------

TASK_HISTORY_VARIANTS = {
    12: [
        {
            "history": [
                {"role": "user",      "content": "switch on the bedroom light"},
                {"role": "assistant", "content": "The bedroom light has been turned on."},
            ],
            "target_room": "bedroom",
        },
        {
            "history": [
                {"role": "user",      "content": "turn on the office light"},
                {"role": "assistant", "content": "The office light has been turned on."},
            ],
            "target_room": "office",
        },
        {
            "history": [
                {"role": "user",      "content": "kitchen lights on please"},
                {"role": "assistant", "content": "The kitchen light has been turned on."},
            ],
            "target_room": "kitchen",
        },
        {
            "history": [
                {"role": "user",      "content": "switch on the hallway light"},
                {"role": "assistant", "content": "The hallway light has been turned on."},
            ],
            "target_room": "hallway",
        },
    ],
    15: [
        {
            "history": [
                {"role": "user",      "content": "lock the front door"},
                {"role": "assistant", "content": "The front door has been locked."},
                {"role": "user",      "content": "and the garage too"},
                {"role": "assistant", "content": "The garage door has been locked."},
            ],
            "target_door": "front",
        },
        {
            "history": [
                {"role": "user",      "content": "lock the back door"},
                {"role": "assistant", "content": "The back door has been locked."},
                {"role": "user",      "content": "and the side door as well"},
                {"role": "assistant", "content": "The side door has been locked."},
            ],
            "target_door": "back",
        },
        {
            "history": [
                {"role": "user",      "content": "can you lock the garage door"},
                {"role": "assistant", "content": "The garage door has been locked."},
                {"role": "user",      "content": "also lock the front door"},
                {"role": "assistant", "content": "The front door has been locked."},
            ],
            "target_door": "garage",
        },
    ],
}

# ---------------------------------------------------------------------------
# Call-quality filter
# ---------------------------------------------------------------------------

_VALID_ENUMS = {
    "toggle_lights": {
        "room":  {"living_room", "bedroom", "kitchen", "bathroom", "office", "hallway"},
        "state": {"on", "off"},
    },
    "set_thermostat": {
        "mode": {"heat", "cool", "auto"},
    },
    "lock_door": {
        "door":  {"front", "back", "garage", "side"},
        "state": {"lock", "unlock"},
    },
    "get_device_status": {
        "device_type": {"lights", "thermostat", "door", "all"},
    },
    "set_scene": {
        "scene": {"movie_night", "bedtime", "morning", "away", "party"},
    },
    "intent_unclear": {
        "reason": {"ambiguous", "off_topic", "incomplete", "unsupported_device"},
    },
}

_TASK_MAX_CALLS = {
    1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3,
    9: 4, 10: 5, 11: 9, 12: 4, 13: 9, 14: 4, 15: 4,
    16: 2, 17: 2, 18: 2, 19: 2,
    # toggle_lights on/off (easy single-tool)
    20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3,
    29: 3, 30: 3, 31: 3, 32: 3, 33: 3, 34: 3,
    35: 3, 36: 3, 37: 3, 38: 3, 39: 3, 40: 3, 41: 3, 42: 3, 43: 3,
    44: 3, 45: 3, 46: 3, 47: 3, 48: 3, 49: 3, 50: 3,
    # adversarial toggle_lights
    51: 4, 52: 4,
    # lock_door combos
    53: 3, 54: 3, 55: 3, 56: 3, 57: 3, 58: 3, 59: 3, 60: 3, 61: 3,
    62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3,
    # adversarial lock_door
    68: 4,
    # set_thermostat
    69: 3, 70: 3, 71: 3, 72: 3, 73: 3, 74: 3, 75: 3,
    76: 4, 77: 4,
    # get_device_status
    78: 3, 79: 3, 80: 3, 81: 3, 82: 3, 83: 3, 84: 3, 85: 3, 86: 3,
    # set_scene
    87: 3, 88: 3, 89: 4, 90: 3, 91: 3, 92: 4, 93: 3, 94: 3, 95: 4,
    # intent_unclear
    96: 2, 97: 2, 98: 2, 99: 2, 100: 2, 101: 2,
}


def _check_call_quality(tool_calls: list[dict], task_id: int) -> bool:
    """Return False if the trace has too many calls or hallucinated enum values."""
    max_calls = _TASK_MAX_CALLS.get(task_id, 10)
    if len(tool_calls) > max_calls:
        return False
    for call in tool_calls:
        valid_enums = _VALID_ENUMS.get(call["name"], {})
        for param, valid_vals in valid_enums.items():
            val = call["args"].get(param)
            if val is not None and val not in valid_vals:
                return False
    return True


# ---------------------------------------------------------------------------
# Verifier factories for history variants
# ---------------------------------------------------------------------------

def _make_pronoun_verifier(room: str):
    """Verifier for task 12 variants: 'it' refers to the given room."""
    def verifier(tool_calls, duration, state):
        from benchmark.run import TaskResult
        call = _find_last_call(tool_calls, "toggle_lights")
        passed = state["lights"][room]["state"] == "off"
        return TaskResult(12, "Turn off light (pronoun reference)", "hard",
                          passed, call["name"] if call else None, passed, duration)
    return verifier


def _make_first_door_verifier(door: str):
    """Verifier for task 15 variants: unlock the first door mentioned in history."""
    def verifier(tool_calls, duration, state):
        from benchmark.run import TaskResult
        door_calls = [c for c in _find_all_calls(tool_calls, "lock_door")
                      if c["args"].get("door") == door]
        call = door_calls[-1] if door_calls else None
        passed = state["doors"][door] == "unlocked"
        return TaskResult(15, "Unlock first door (back-reference)", "hard",
                          passed, call["name"] if call else None, passed, duration)
    return verifier


# ---------------------------------------------------------------------------
# Paraphrase generation and deduplication
# ---------------------------------------------------------------------------

def generate_paraphrases(task_prompt: str, existing: list[str], n: int = 10) -> list[str]:
    existing_block = "\n".join(f"- {p}" for p in existing) if existing else "(none)"
    user = f"""\
Generate {n} paraphrases of the following home-assistant user request.

Original: "{task_prompt}"

Already-existing phrases (DO NOT repeat or closely echo these):
{existing_block}

Requirements:
- Vary register: casual, clipped, formal, frustrated, polite, terse.
- Vary phrasing: imperatives, questions, hints, indirect requests.
- Include some with typos or dropped words, as a real user might type.
- Include some very short (2-4 words) and some longer ones.
- Every phrase must have the same underlying intent as the original.
- Output ONLY the phrases, one per line, no numbering, no explanation.
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a linguistic diversity assistant."},
            {"role": "user",   "content": user},
        ],
        temperature=1.0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content or ""
    lines = [ln.strip().strip("-").strip() for ln in raw.splitlines()]
    return [ln for ln in lines if ln]


def jaccard_dedup(examples: list[dict], threshold: float = 0.8) -> list[dict]:
    def bigrams(text: str) -> set[tuple[str, str]]:
        words = text.lower().split()
        return set(zip(words, words[1:])) if len(words) >= 2 else {(w, "") for w in words}

    def user_message(ex: dict) -> str:
        for msg in ex["messages"]:
            if msg["role"] == "user":
                return msg["content"]
        return ""

    kept: list[dict] = []
    kept_bigrams: list[set] = []
    for ex in examples:
        bg = bigrams(user_message(ex))
        if not any(len(prev & bg) / len(prev | bg) >= threshold
                   for prev in kept_bigrams if prev | bg):
            kept.append(ex)
            kept_bigrams.append(bg)
    return kept


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def collect_example(task, prompt, backend="openai",
                    history_override=None, verifier_override=None,
                    temperature: float = 0.7):
    home_state.clear()
    home_state.update(copy.deepcopy(_DEFAULT_STATE))

    if task.id in _RANDOMIZED_TASKS:
        randomize_state()

    initial_state = copy.deepcopy(home_state)

    tool_calls_seen = []
    messages_out = []

    def capture(name, args, result):
        tool_calls_seen.append({"name": name, "args": args})

    history = history_override if history_override is not None else task.history
    run_agent(prompt, history=history, backend=backend,
              on_tool_call=capture, messages_out=messages_out,
              temperature=temperature)

    final_state = copy.deepcopy(home_state)
    verifier = verifier_override if verifier_override is not None else task.verifier

    # Task 14 verifier accepts initial_state for relative temperature check
    if task.id == 14 and verifier_override is None:
        result = verifier(tool_calls_seen, 0.0, final_state, initial_state)
    else:
        result = verifier(tool_calls_seen, 0.0, final_state)

    if not result.passed:
        return None

    if not _check_call_quality(tool_calls_seen, task.id):
        return None

    return {
        "task_id":    task.id,
        "difficulty": task.difficulty,
        "messages":   messages_out,
        "tools":      TOOL_SCHEMAS,
    }


def generate_dataset(runs=20, backend="openai", temperature=0.7, n_paraphrases=10):
    examples = []
    stats = {}

    for task in TASKS:
        variants = TASK_HISTORY_VARIANTS.get(task.id) or [None]
        static = TASK_PARAPHRASES.get(task.id, [])
        if backend == "openai" and n_paraphrases > 0:
            generated = generate_paraphrases(task.prompt,
                                             existing=[task.prompt] + static,
                                             n=n_paraphrases)
        else:
            generated = []
        prompts = [task.prompt] + static + generated
        passed = failed = 0

        for variant in variants:
            history_override = variant["history"] if variant else None
            if variant and task.id == 12:
                verifier_override = _make_pronoun_verifier(variant["target_room"])
            elif variant and task.id == 15:
                verifier_override = _make_first_door_verifier(variant["target_door"])
            else:
                verifier_override = None

            for prompt in prompts:
                for _ in range(runs):
                    ex = collect_example(task, prompt, backend,
                                         history_override=history_override,
                                         verifier_override=verifier_override,
                                         temperature=temperature)
                    if ex:
                        examples.append(ex)
                        passed += 1
                    else:
                        failed += 1

        stats[task.id] = (task.name, len(prompts) * len(variants), passed, failed)

    return examples, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate golden SFT dataset using gpt-4o-mini.")
    parser.add_argument("--runs",            type=int,   default=5,      help="Runs per prompt (default 5)")
    parser.add_argument("--temperature",     type=float, default=0.7,    help="Sampling temperature for agent (default 0.7)")
    parser.add_argument("--n-paraphrases",   type=int,   default=10,     help="LLM-generated paraphrases per task (default 10)")
    parser.add_argument("--dedup-threshold", type=float, default=0.8,    help="Jaccard similarity threshold for dedup (default 0.8)")
    parser.add_argument("--backend",         default="openai", choices=["local", "openai"])
    args = parser.parse_args()

    print(f"Generating dataset: {args.runs} runs per prompt, backend={args.backend}, "
          f"temperature={args.temperature}, n_paraphrases={args.n_paraphrases}")
    examples, stats = generate_dataset(
        runs=args.runs, backend=args.backend,
        temperature=args.temperature, n_paraphrases=args.n_paraphrases,
    )

    before = len(examples)
    examples = jaccard_dedup(examples, threshold=args.dedup_threshold)
    print(f"Dedup removed {before - len(examples)} near-duplicate examples.")

    print(f"\n{'ID':<4} {'Task':<45} {'Prompts':<8} {'Passed':<8} {'Failed'}")
    print("-" * 78)
    for tid, (name, n_prompts, passed, failed) in stats.items():
        print(f"{tid:<4} {name:<45} {n_prompts:<8} {passed:<8} {failed}")
    total = sum(p + f for _, _, p, f in stats.values())
    print("-" * 78)
    print(f"Total attempts: {total}   Saved examples: {len(examples)}   "
          f"Pass rate: {100 * len(examples) / total:.0f}%")

    Path("benchmark/datasets").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path("benchmark/datasets") / f"{timestamp}_golden_dataset.jsonl"
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nDataset saved to {out_path}")
