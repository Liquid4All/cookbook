# LFM2.5-1.2B-Thinking demo

## Flight search assistant

A Python CLI that helps you find and book flight tickets:

```
What is the chepeast flight Barcelona-Belgrade for January 19th?
```

These examples showcase the thinking + tool calling capabilities of the model.


## Quickstart

1. Start `llama-server` with the GGUF checkpoint
    ```
    llama-server -m $MODEL_GGUF --jinja --temp 0 --seed 41 --port $LLAMA_SERVER_PORT
    ```

2. Run assistant with a few queries:
    ```
    # 1 tool call
    # OK!
    uv run flight_search.py --model gpt-4o --prompt "What flights are available from New York to Paris on 2026-01-19?"
    uv run flight_search.py --port $LLAMA_SERVER_PORT --prompt "What flights are available from New York to Paris on 2026-01-19?"

    # 1 tool call
    # OK
    uv run flight_search.py --model gpt-4o --prompt "Are there any direct flights from Belgrade to Istanbul on 2026-02-18?"
    uv run flight_search.py --port $LLAMA_SERVER_PORT --prompt "Are there any direct flights from Belgrade to Istanbul on 2026-02-18?"
    
    # 1 tool call
    uv run flight_search.py --model gpt-4o --prompt "Book flight AA495 for 2026-02-04"
    uv run flight_search.py --port $LLAMA_SERVER_PORT --prompt "Book flight AA495 for 2026-02-04"

    # GPT-4o works, LFM2 does not
    uv run flight_search.py --model gpt-4o --prompt "Book the cheapest flight from Barcelona to Belgrade on 2026-01-31"
    uv run flight_search.py --port $LLAMA_SERVER_PORT --prompt "Book the cheapest flight from Barcelona to Belgrade on 2026-01-31"
    
    # GPT-4o works, LFM2 does not
    uv run flight_search.py --model gpt-4o --prompt "I'm planning a trip from Barcelona. On 2026-02-14 I want to fly to either Amsterdam or Istanbul. Which destination has the cheaper flight, and what's the flight duration for each option?"
    uv run flight_search.py --port $LLAMA_SERVER_PORT --prompt "I'm planning a trip from Barcelona. On 2026-02-14 I want to fly to either Amsterdam or Istanbul. Which destination has the cheaper flight, and what's the flight duration for each option?"
    ```



## TODOs
- [x] Check `flight_search.py` works when using GPT-4
    Design 10 questions from easy to hard and check frontier model solves it perfectly using only 2 tools.
    
- [x] Make it work when using LFM2.5-1.2B-Instruct.

- [ ] Adjust `llama-server` command to use `--hf` flag to download model.
- [ ] Extract configuratino in yamls that include
    - model_id
    - system_prompt
