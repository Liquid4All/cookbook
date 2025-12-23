import json
import re
from typing import Callable

from transformers import AutoModelForCausalLM, AutoTokenizer

from tools import (
    get_flight_status,
    get_weather,
    tool_definitions,
    tool_callers,
)

class ToolAugmentedLM:
    """
    """
    def __init__(
        self,
        model_id: str,
        tool_definitions: list[dict],
        tool_callers: dict[str, Callable],
    ):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tool_definitions = tool_definitions
        self.tool_callers = tool_callers

    def act(self, messages: list[dict]) -> list[dict]:
        """
        Given a history of messages, generates a text completion.
        If this completion includes tool calls, parse them, execute them, and genearte
        a new text completion.
        The loop ends if the text completion contains no tool calls
        """
        while True:
            response = self._get_text_completion(messages, skip_special_tokens=False)
            tool_calls = self._parse_model_output(response)
            
            if not tool_calls:
                # no tool calls, exit loop and return the latest response from the model
                response = self._get_text_completion(messages, skip_special_tokens=True)
                break
            
            # executes tool calls and appends messages with roles "assistant" and "tool" 
            messages = self._execute_tool_calls(tool_calls, messages)

        return response


    def _get_text_completion(self, messages: list[dict], skip_special_tokens: bool) -> str:
        """
        Given message history and a set of available tool, generate text completion

        Args:
            messages: history of previous messages

        Return:
            str: model response, possibly containing a tool that needs to be called
        """
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_definitions,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        outputs = self.model.generate(**inputs.to(self.model.device), max_new_tokens=256)
        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=skip_special_tokens
        )
        return response
    
    def _parse_model_output(self, response) -> list[dict]:
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

    def _execute_tool_calls(
        self,
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
            function_to_call = self.tool_callers[function_name]
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
    """
    tool_augmented_lm = ToolAugmentedLM(
        model_id,
        tool_definitions=tool_definitions,
        tool_callers=tool_callers,    
    )

    user_prompts = [
        # "Get status for flight 123",
        # "Get status for flight 456",
        # "Get status for flight 789",
        # "What is the weather like in San Francisco?",
        # "How much money do I have in my savings account?"
        "What is the weather like in San Francisco and the status of flight 456?"
    ]

    for i, prompt in enumerate(user_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}/{len(user_prompts)}")
        print(f"{'='*80}")
        print(f"User: {prompt}")
        print(f"Model: {model_id}")
        print(f"{'-'*80}")

        messages = [{"role": "user", "content": prompt}]
        response = tool_augmented_lm.act(messages)
        print(f"Final model response: ", response)

        print(f"{'='*80}")
    


if __name__ == '__main__':
    from fire import Fire
    Fire(run_examples)