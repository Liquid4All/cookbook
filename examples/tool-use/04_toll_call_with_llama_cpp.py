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

        # breakpoint()

        return message

    def assistant(message):
        return {"role": "assistant", "content": message}

    from tools import tool_definitions

    user_prompts = [
        "Get status for flight 123",
        "Get status for flight 456",
        "Get status for flight 789",
        "What is the weather like in San Francisco?",
        "How much money do I have in my savings account?"
    ]

    for i, prompt in enumerate(user_prompts, 1):

        print(f"\n{'='*80}")
        print(f"Test Case {i}/{len(user_prompts)}")
        print(f"{'='*80}")
        print(f"User: {prompt}")
        
        messages = [
            {"role": "system", "content": "Follow json schema."},
            {"role": "user", "content": prompt}
        ]
        response = ask(messages, tools=tool_definitions)

        print(f"Final model response: ", response)
        print(f"{'='*80}")

scenario_tool_call(8080)