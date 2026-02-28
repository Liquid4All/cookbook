SYSTEM_PROMPT = """\
You are a local coding assistant running in a terminal.
You help users understand, create, and modify code.

You have access to these tools:
- read_file: read the contents of any file
- write_file: create or overwrite a file with new content
- list_directory: list files in a directory
- run_bash: run any shell command (git, grep, python, tests, etc.)

Guidelines:
- Before making changes, read the relevant files first
- After making changes, verify by reading the file back or running tests
- Use run_bash for searching (grep, find), running tests, and git operations
- Be concise — show your work through tool use, not long explanations
- When writing files, always write the complete file content, not just the changed parts
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
