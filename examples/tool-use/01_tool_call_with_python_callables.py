import json
import re

from transformers import AutoModelForCausalLM, AutoTokenizer

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

# maps tool names to tool callers
TOOLS = {"get_flight_status": get_flight_status, "get_weather": get_weather}

def ask_model(messages: list[dict], skip_special_tokens: bool) -> str:
    """
    Given message history and a set of available tool, generate text completion

    Args:
        messages: history of previous messages

    Return:
        str: model response, possibly containing a tool that needs to be called
    """
    inputs = tokenizer.apply_chat_template(messages, tools=[get_flight_status, get_weather], add_generation_prompt=True, return_dict=True, return_tensors="pt")
    outputs = model.generate(**inputs.to(model.device), max_new_tokens=256)
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=skip_special_tokens)
    return response

def parse_model_output(response) -> list[dict]:
    """Extract tool calls from model response."""
    # Extract content between <|tool_call_start|> and <|tool_call_end|>
    match = re.search(r'<\|tool_call_start\|>(.*?)<\|tool_call_end\|>', response)
    if not match:
        return []

    tool_call_str = match.group(1).strip()
    # Remove surrounding brackets: [get_status(id="123")] -> get_status(id="123")
    tool_call_str = tool_call_str.strip('[]')

    # Parse function name and arguments
    # Format: function_name(arg1="val1", arg2="val2")
    func_match = re.match(r'(\w+)\((.*)\)', tool_call_str)
    if not func_match:
        return []

    func_name = func_match.group(1)
    args_str = func_match.group(2)

    # Parse arguments (simple key="value" format)
    args = {}
    for arg in re.findall(r'(\w+)="([^"]*)"', args_str):
        args[arg[0]] = arg[1]

    return [{
        "type": "function",
        "function": {
            "name": func_name,
            "arguments": json.dumps(args)
        }
    }]

def execute_tool_calls(
    tool_calls: str,
    messages: list[dict]
) -> list[dict]:
    """
    Runs one or many tool calls and appends each of the results as a message with
    """
    # Add IDs to tool calls (required by chat template)
    for i, tool_call in enumerate(tool_calls):
        tool_call["id"] = f"call_{i}"

    # Add assistant message with tool calls (content can be null or empty when there are tool calls)
    messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

    # Execute each tool call
    for tool_call in tool_calls:
        function_name = tool_call["function"]["name"]
        function_args = json.loads(tool_call["function"]["arguments"])
        tool_call_id = tool_call["id"]

        print(f"\nExecuting {function_name} with args: {function_args}")

        # Call the function
        function_to_call = TOOLS[function_name]
        function_result = function_to_call(**function_args)

        print(f"Tool output: {function_result}")

        # Add tool result to messages
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(function_result)
        })

    return messages


def run_examples(model_id: str):
    """
    One-turn interaction with tool calling
    """
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        print(f"Model: {model_id}")
        print(f"{'-'*80}")

        messages = [{"role": "user", "content": prompt}]

        # generate a text completion with the model
        response = ask_model(messages, skip_special_tokens=False)
        print(f"Raw model response: ", response)

        # parse model response 
        tool_calls = parse_model_output(response)
        print('Parsed tool call: ', tool_calls)

        # execute tool calls and append new messages with results
        messages = execute_tool_calls(tool_calls, messages)

        # generate a new text complection with the model
        response = ask_model(messages, skip_special_tokens=True)
        print(f"Final model response: ", response)

        print(f"{'='*80}")

if __name__ == '__main__':
    from fire import Fire
    Fire(run_examples)