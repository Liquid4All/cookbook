def scenario_tool_call(port):
    import openai

    client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")

    def ask(messages, tools=None):
        model = client.models.list().data[0].id
        print(f"model={model}")

        print("--------------------------------")
        print("messages:")
        for message in messages:
            print(message)

        kwargs = {
            "model": model,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        print("raw response:")
        print(f"Content: {message.content}")
        print(f"Tool calls: {message.tool_calls}")
        print("--------------------------------")
        return message

    def assistant(message):
        return {"role": "assistant", "content": message}

    # Define the tools matching the Rust definitions
    tools = [
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
                    "additionalProperties": False,
                },
            },
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
                    "additionalProperties": False,
                },
            },
        },
    ]

    # Example 1: Ask for temperature
    user1 = {"role": "user", "content": "What's the current temperature in Paris?"}
    _response1 = ask([user1], tools=tools)

    # Example 2: Ask for time
    user2 = {"role": "user", "content": "What time is it in Tokyo right now?"}
    _response2 = ask([user2], tools=tools)

    # Example 3: Ask for both
    user3 = {"role": "user", "content": "Can you tell me the current time and temperature in New York?"}
    _response3 = ask([user3], tools=tools)


scenario_tool_call(8080)