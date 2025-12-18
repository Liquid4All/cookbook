curl http://localhost:8080/v1/chat/completions -d '{
    "model": "gpt-3.5-turbo",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature",
                "description": "Gets the temperature at a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The location to get the temperature for."}
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Gets the current time at a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The location to get the current time for."}
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }
            }
        }
    ],
    "messages": [
        {
        "role": "user",
        "content": "What'\''s the current temperature in Paris?"
        }
    ]
}'

echo ""
echo "================================"
echo ""

curl http://localhost:8080/v1/chat/completions -d '{
    "model": "gpt-3.5-turbo",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature",
                "description": "Gets the temperature at a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The location to get the temperature for."}
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Gets the current time at a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The location to get the current time for."}
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }
            }
        }
    ],
    "messages": [
        {
        "role": "user",
        "content": "What time is it in Tokyo right now?"
        }
    ]
}'

echo ""
echo "================================"
echo ""

curl http://localhost:8080/v1/chat/completions -d '{
    "model": "gpt-3.5-turbo",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature",
                "description": "Gets the temperature at a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The location to get the temperature for."}
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Gets the current time at a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The location to get the current time for."}
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }
            }
        }
    ],
    "messages": [
        {
        "role": "user",
        "content": "Can you tell me the current time and temperature in New York?"
        }
    ]
}'