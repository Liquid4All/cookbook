# LFM2.5-1.2B-Thinking demo

## Flight search assistant

A Python CLI that answers all flight-related questions you may have, for example

```
What is the chepeast flight Barcelona-Belgrade for January 19th?
How much does the cheapest flight cost from Barcelona to Belgrade in the next 7 days?
What is the fastest route between New York and Paris?
Can you build a route from Istnbul to Tampa, with a layover in New York?
```

These examples showcase the thinking + tool calling capabilities of the model.


## Quickstart

1. Start `llama-server` with the GGUF checkpoint
    ```
    llama-server -m $MODEL_GGUF --jinja --temp 0 --seed 41 --port $LLAMA_SERVER_PORT
    ```

2. Run assistant with a few queries:
    ```
    uv run flight_search.py --question "What is the chepeast flight from BAR to BEG for 2026-01-19?"
    uv run flight_search.py --question "How much does the cheapest flight cost from Barcelona to Belgrade in the next 7 days?"
    ```

## TODOs

- [ ] Check `flight_search.py` works when using GPT-4
    Design 10 questions from easy to hard and check frontier model solves it perfectly using only 2 tools.
    
- [ ] Make it work when using LFM2.5-1.2B-Instruct.







## Vibe checks with the GGUF checkpoint

```
MODEL_GGUF=$PWD/tim_grpo1.2b_from268012_d=olympgsmdiverse_rm=unified_tis=2.0_opt=adamw_lr=1e-6_bs=256_rl=7168_n=16_ep=10_capture1k_mb256_nodes2_maskzeroadv_step340_282712_HF-GGUF/tim_grpo1.2b_from268012_d=olympgsmdiverse_rm=unified_tis=2.0_opt=adamw_lr=1e-1.3B-6_bs=256_rl=7168_n=16_ep=10_capture1k_mb256_nodes2_maskzeroadv_step340_282712_HF-Q4_0.gguf
```

### Thinking traces
Does the model output thinking traces between `<think></think>` ?
```
llama-cli -m $MODEL_GGUF -p "Who are you?"
llama-cli -m $MODEL_GGUF -p "How to train a Language Model, step by step"
```

### Tool calling
Does tool calling work when serving the model with `llama-server`?

1. Start the `llama-server`
    ```
    llama-server -m $MODEL_GGUF --jinja --temp 0
    ```

2. Run experiments
    ```bash
    # one-shoot
    uv run python flight_search.py --port $LLAMA_SERVER_PORT --prompt "Find flights from NYC to London on 2026-03-15"

    # interactive mode
    uv run python flight_search.py --model gpt-4 --prompt "Find flights from NYC to London on 2026-03-15" 
    
    ```







# Idea -> Minimal ReAct Agent Implementation

A minimal from-scratch implementation of the **ReAct** (Reasoning + Acting) paradigm for building AI agents.

## Steps

- [x] Check `react_agent_with_llm.py` works when using Sonnet 
- [ ] Replace Sonnet with a local LM using Transformers as the inference engine.
    - Encapsulate llm into a common interface, so I can easily swap between model and inference engines.


## What is ReAct?

ReAct is a prompting paradigm that interleaves **reasoning traces** with **task-specific actions**, allowing language models to:
1. **Reason** about what to do next
2. **Act** by calling tools/functions
3. **Observe** the results
4. Repeat until reaching a final answer

## Architecture

```
┌─────────────┐
│   Question  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  ReAct Loop (max iterations)        │
│                                      │
│  1. Build prompt with:               │
│     - Question                       │
│     - Available tools                │
│     - Conversation history           │
│                                      │
│  2. Call LLM → Get response:         │
│     Thought: [reasoning]             │
│     Action: tool_name(arg)           │
│                                      │
│  3. Parse & Execute action           │
│                                      │
│  4. Add observation to history       │
│     Observation: [result]            │
│                                      │
│  5. Check for final answer           │
│     Answer: [final answer]           │
│                                      │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│ Final Answer│
└─────────────┘
```

## Core Components

### 1. ReActAgent Class
The main agent that orchestrates the reasoning loop:
- `__init__(llm_call, tools)` - Initialize with LLM and available tools
- `run(question)` - Execute the ReAct loop
- `_build_prompt()` - Format the prompt with instructions
- `_parse_action()` - Extract tool calls from LLM response
- `_parse_answer()` - Extract final answer from LLM response

### 2. Tools
Functions that the agent can call:
```python
def calculator(expression: str) -> float:
    """Evaluates a mathematical expression."""
    return eval(expression)

def search(query: str) -> str:
    """Searches for information."""
    return search_results
```

### 3. LLM Integration
Any callable that takes a prompt string and returns a response:
```python
def llm_call(prompt: str) -> str:
    # Call your favorite LLM API
    return response
```

## Usage

### Basic Example
```python
from react_agent import ReActAgent, calculator, search

# Define tools
tools = {
    "calculator": calculator,
    "search": search,
}

# Create agent with your LLM
agent = ReActAgent(llm_call=your_llm_function, tools=tools)

# Run agent
result = agent.run("What is 15 * 24?")
```

### With Real LLM APIs

```python
from anthropic import Anthropic

def create_claude_llm():
    client = Anthropic()
    def llm_call(prompt: str) -> str:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    return llm_call

llm = create_claude_llm()
agent = ReActAgent(llm_call=llm, tools=tools)
```

## Example ReAct Trace

**Question:** What is 15 * 24?

**Iteration 1:**
```
Thought: I need to calculate 15 * 24
Action: calculator(15 * 24)
Observation: 360
```

**Iteration 2:**
```
Thought: I now know the final answer
Answer: 15 multiplied by 24 equals 360
```

## Key Features

✅ **Minimal dependencies** - Pure Python with regex
✅ **LLM-agnostic** - Works with any LLM API
✅ **Extensible tools** - Easy to add new capabilities
✅ **Verbose mode** - Debug the reasoning process
✅ **Error handling** - Gracefully handles tool errors

## Files

- `react_agent.py` - Core implementation with mock LLM for demo
- `react_agent_with_llm.py` - Examples of real LLM integration
- `README.md` - This file

## Extending

### Add New Tools

```python
def wikipedia(query: str) -> str:
    """Searches Wikipedia for information."""
    # Your implementation
    return result

tools = {
    "calculator": calculator,
    "search": search,
    "wikipedia": wikipedia,  # New tool
}
```

### Customize Prompt Format

Modify `_build_prompt()` to change the instruction format or add few-shot examples.

### Add Tool Validation

Extend `_parse_action()` to validate tool arguments before execution.

## Limitations

This is a **minimal implementation** for educational purposes. Production agents should add:
- Better error recovery
- Tool result validation
- Context window management
- Multi-turn conversation support
- Parallel tool execution
- Cost tracking and token limits
- Streaming responses

## References

- [ReAct Paper](https://arxiv.org/abs/2210.03629) - Yao et al., 2022
- [LangChain ReAct](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)

## License

MIT - Feel free to use and modify!

-----

## Archived
Let's built a real-world (aka useful) Python using `LFM2.5-1.2B-Instruct`, that plays on its
strong tool calling capabilities.

The tool to use are the ones provided by the Tavily API:

- Search the web: `user question -> answer`
    ```
    from tavily import TavilyClient

    tavily_client = TavilyClient(api_key="tvly-YOUR_API_KEY")
    response = tavily_client.search("Who is Leo Messi?")

    print(response)
    ```

- Extract webpage content: `url -> raw content`
    ```
    from tavily import TavilyClient

    tavily_client = TavilyClient(api_key="tvly-YOUR_API_KEY")
    response = tavily_client.extract("https://en.wikipedia.org/wiki/Artificial_intelligence")

    print(response)
    ```

- Crawl
    ```
    from tavily import TavilyClient

    tavily_client = TavilyClient(api_key="tvly-YOUR_API_KEY")
    response = tavily_client.crawl("https://docs.tavily.com", instructions="Find all pages on the Python SDK")

    print(response)
    ```

## Ideas

- Flight finder
- Product News tracker

## Implementation steps

- [ ] Build single-turn agent based on [this](https://github.com/tavily-ai/tavily-cookbook/blob/main/cookbooks/search/product_news_tracker.ipynb).
- 


### 1. Build single-turn tool calling with LFM2-2.6B and the transformers library

Use as a reference this working code:
```
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
```

In our case, the tools are from the tavily API:
```

```

## Idea 2. Product News tracker

```
# tool.py
def search_product_updates(company_name: str, domains: list):
    """Search for product updates from a company.
    
    Args:
        company_name: Company to search for
        domains: Company domains for self-reported news (e.g., ["openai.com"])
    
    Returns:
        List of results with 'search_type' field indicating source
    """
    all_results = []

    # Self-reported news from company domains
    company_results = client.search(
        query=f"{company_name} product news, updates, releases, and announcements",
        search_depth="basic",
        max_results=10,
        include_domains=domains,
    )

    for result in company_results["results"]:
        result["search_type"] = "Self-reported News"
        all_results.append(result)

    # Third-party coverage from news sources
    news_results = client.search(
        query=f"{company_name} product news, updates, releases, and announcements",
        search_depth="basic",
        max_results=10,
        time_range="month",
        topic="news",
    )

    for result in news_results["results"]:
        result["search_type"] = "Third-party Coverage"
        all_results.append(result)

    return all_results
```