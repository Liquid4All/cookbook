from .config import Config
from .context import ContextManager, SYSTEM_PROMPT
from .llm.base import LLMClient
from .tools import TOOLS, execute_tool


class Agent:
    """The agentic loop."""

    def __init__(self, llm: LLMClient, config: Config) -> None:
        self._llm = llm
        self._context = ContextManager(max_messages=config.max_context_messages)

    def run_turn(self, user_input: str) -> None:
        """Process one user message, running the inner loop until end_turn."""
        self._context.add({"role": "user", "content": user_input})

        tools_used = 0
        nudged = False

        while True:
            if self._context.should_compact():
                self._context.compact()
                print("[context compacted]")

            response = self._llm.chat(
                messages=self._context.get_messages(),
                tools=TOOLS,
                system=SYSTEM_PROMPT,
            )

            # Add assistant response to history
            self._context.add({"role": "assistant", "content": response.content})

            tool_calls = [b for b in response.content if b["type"] == "tool_use"]

            if not tool_calls:
                # If the model produced text without ever calling a tool, nudge it once.
                if tools_used == 0 and not nudged:
                    nudged = True
                    self._context.add({
                        "role": "user",
                        "content": (
                            "You must call a tool to complete this request. "
                            "Do not describe or simulate the result — call the appropriate tool directly."
                        ),
                    })
                    continue
                # End of turn — print the final text response
                for block in response.content:
                    if block["type"] == "text":
                        print(block["text"])
                break

            # Execute all tool calls and collect results
            tools_used += len(tool_calls)
            tool_results = []
            for call in tool_calls:
                args_preview = ", ".join(f"{k}={v!r}" for k, v in call["input"].items())
                print(f"  [tool] {call['name']}({args_preview})")
                result = execute_tool(call["name"], call["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": result,
                })

            # Feed results back as a user message and loop
            self._context.add({"role": "user", "content": tool_results})
