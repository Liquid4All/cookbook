# Python CLI tool find and book cheap plane tickets using LFM2.5-1.2B-Thinking

[![Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.com/invite/liquid-ai)

This is a minimal Python CLI that helps you find and book plane tickets using tool calling and reasoning to solve multi-step workflows.

This project showcases the power of LFM2.5-1.2B-Thinking, a small Language Model that
excells at tasks that require reasoning, logic and strong tool calling skills.

![See it in action](./media/demo.gif)

### Table of contents
- [Quickstart](#quickstart)
- [How does it work?](#how-does-it-work)
- [Need help?](#need-help)

## Quickstart

1. Make sure you have `uv` installed in your system
    ```
    uv --version
    ```
    If the previous command fails, install `uv` following [these instructions](https://docs.astral.sh/uv/getting-started/installation/).


2. Build the project
    ```
    uv sync
    ```

3. Ask the flight search assistant to help you find and book plane tickets. For example:
    ```
    # single tool call to `search_flights`
    uv run flight_search.py --query "What flights are available from New York to Paris on 2026-01-19?"
    
    # single tool call to `book_flight`
    uv run flight_search.py --query "Book flight AA495 for 2026-02-04"

    # 2-step sequential tool call to `search_flights` and then `book_flight`
    uv run flight_search.py --query "Book the cheapest flight from Barcelona to Belgrade on 2026-01-31"

    # N-step sequential tool call
    uv run flight_search.py --query "Book the cheapest flight from Barcelona to a US city on the East Coast that is not NYC on 2026-02-14"
    ```

## How does it work?

The model has access to 2 tools:

- `search_flights` -> to retrieve contextual information
- `book_flights` -> to act on the outside world

Given a user request, for example `Book the cheapest flight from Barcelona to Belgrade on 2026-01-31` the model iteratively

- Generate a response possibly with tool calls
- Executes any tool calls
- Regenerates response

The model is served fully locally using llama.cpp using the 

## Next steps

- [ ] Add an evaluation dataset and loop.
- [ ] If necessary, boost model performance with GRPO fine-tuning with verifiable rewards.


## Need help?
Join the Liquid AI Discord Community and ask.

[![Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.com/invite/liquid-ai)

