import json

from meeting_intelligence.tools import TOOLS

SYSTEM_PROMPT = f"""\
You are a Meeting Intelligence Agent. You have tools to accomplish tasks — always use them.
Output function calls as JSON.

List of tools: {json.dumps(TOOLS)}

Rules:
- ALWAYS call the appropriate tool to fulfill a request. Never say you cannot do something that a tool enables.
- Use only the tools needed for the current request — not all tasks require all tools.
- Never describe or narrate what a tool would do — call the tool directly. Do not simulate tool results in text.
- Only produce a text response to summarize what was accomplished after the tools have been executed.
- If a due date is not mentioned, default to one week from today.
- If an owner is not mentioned, use "unassigned".
"""


class ContextManager:
    """
    Manages the conversation history passed to the model.

    Simple compaction strategy: when message count exceeds the limit,
    keep the first 2 messages (original task context) and the most recent
    half, inserting a notice where messages were dropped.
    """

    def __init__(self, max_messages: int = 40) -> None:
        self._messages: list[dict] = []
        self._max_messages = max_messages

    def add(self, message: dict) -> None:
        self._messages.append(message)

    def get_messages(self) -> list[dict]:
        return self._messages.copy()

    def should_compact(self) -> bool:
        return len(self._messages) > self._max_messages

    def compact(self) -> None:
        """Drop the middle of the history, preserving head and tail."""
        if not self.should_compact():
            return
        keep_recent = self._max_messages // 2
        head = self._messages[:2]
        tail = self._messages[-keep_recent:]
        notice = {
            "role": "user",
            "content": "[Context compacted: older messages removed to stay within limits]",
        }
        self._messages = head + [notice] + tail
