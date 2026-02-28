## Intent

Create basic implementation of an agentic coding, following the design of
- Claude Code https://code.claude.com/docs/en/how-claude-code-works
- opencode implementation in https://github.com/anomalyco/opencode

Using a minimal set of tools that a coding agent requires to
- read content from a file
- save content to a file
- run bash commands
- etc.

I am not 100% sure about the exact set of minimal tools to use. Make suggestions/give advice based on this documentation https://code.claude.com/docs/en/how-claude-code-works

Keep the code simple, and do not overcomplicate things. The purpose of this project is to present it to an audience that is learning this for the first time.
I don't want to build something mega polished. The intent here is to create a design that is functional but not too complex to scare people. Remember, this is supposed to be kind of a "Hello, World!" tutorial for software engineers in this new era.

I want the code to be modular into

- tools
- context managment logic (like compactation, truncation), keep it simple remember
- language model, which can be either
    - from Anthropic or
    - local model running with llama.cpp. We will use LFM2-24B-A2B Q4_0 quantization as the default one

    I am interested in using the local model, but I added Antrhopic so I can validate with a very strong model like Sonnet that the agent loop and tools are good enough for the job. Once we are confident the agentic loop, including context management and tool definitions, are ok, I will switch to the locally running model.

Design the code so that the agentic loop is model independent, and I can swith from Sonnet to local model without changing the code, but just updating the configuration.

The user interacts with this tool from the command line, as Claude Code does.

The terminal UI experience is similar to Claude Code. Do not overcomplicate.