def get_flight_status(flight_id: str):
    """Get flight status for a given flight id.

    Args:
        flight_id: The ID to get flight status for.
    """
    if flight_id == '123':
        return {"status": "landed"}
    elif flight_id == '456':
        return {"status": "delayed"}
    else:
        return {"status": "not-available"}

def get_weather(location: str):
    """Get current weather for a location.

    Args:
        location: The location to get weather for.
    """
    # Mock weather data
    return {
        "location": location,
        "temperature": 72,
        "unit": "fahrenheit",
        "conditions": "partly cloudy",
        "humidity": 65,
        "wind_speed": 8
    }

# list of JSON schemas for the available tools
tool_definitions = [{
    "type": "function",
    "function": {
        "name": "get_flight_status",
        "description": "Get flight status for a given flight id.",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_id": {"type": "string", "description": "The ID to get flight status for."}
            },
            "required": ["flight_id"]
        }
    }
},
{
    "type": "function",
    "function": {
    "name": "get_weather",
    "description": "Get current weather information for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
        "location": {
            "type": "string",
            "description": "The location to get weather for"
        },
        },
        "required": ["location"]
    }
    }
}]

# maps tool names to the function that executes it.
tool_callers = {
    "get_flight_status": get_flight_status,
    "get_weather": get_weather
}