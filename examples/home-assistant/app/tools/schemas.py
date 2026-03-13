TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "toggle_lights",
            "description": "Turn lights on or off in a specific room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room":  {"type": "string", "enum": ["living_room", "bedroom", "kitchen", "bathroom", "office", "hallway"]},
                    "state": {"type": "string", "enum": ["on", "off"]},
                },
                "required": ["room", "state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_thermostat",
            "description": "Set the thermostat temperature (60-80 F) and operating mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "integer", "minimum": 60, "maximum": 80},
                    "mode":        {"type": "string", "enum": ["heat", "cool", "auto"]},
                },
                "required": ["temperature", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lock_door",
            "description": "Lock or unlock a specific door.",
            "parameters": {
                "type": "object",
                "properties": {
                    "door":  {"type": "string", "enum": ["front", "back", "garage", "side"]},
                    "state": {"type": "string", "enum": ["lock", "unlock"]},
                },
                "required": ["door", "state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_device_status",
            "description": "Get the current status of home devices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_type": {"type": "string", "enum": ["lights", "thermostat", "door", "all"]},
                    "room":        {"type": "string", "description": "Optional room name to filter lights status."},
                },
                "required": ["device_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_scene",
            "description": "Activate a preset home scene that adjusts multiple devices at once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene": {"type": "string", "enum": ["movie_night", "bedtime", "morning", "away", "party"]},
                },
                "required": ["scene"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "intent_unclear",
            "description": (
                "Call this tool instead of responding with text when the request cannot be fulfilled. "
                "Reason 'ambiguous': request could mean multiple things. "
                "Reason 'off_topic': completely outside home automation (ordering food, weather, music, etc.). "
                "Reason 'incomplete': pronoun or reference with no prior context (e.g. 'turn it on'). "
                "Reason 'unsupported_device': within home domain but feature unavailable (brightness, TV, cameras)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "enum": ["ambiguous", "off_topic", "incomplete", "unsupported_device"]},
                },
                "required": ["reason"],
            },
        },
    },
]
