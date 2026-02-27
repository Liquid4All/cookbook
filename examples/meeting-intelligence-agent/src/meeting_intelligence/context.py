SYSTEM_PROMPT = """\
You are a Meeting Intelligence Agent running entirely on local hardware.
Your job is to process meeting transcripts and turn them into structured outputs.

You have access to these tools:
- read_transcript: read a meeting transcript file from disk
- lookup_team_member: look up a person's email and role from the team directory
- create_task: create a task record in the local project tracker
- save_summary: save the meeting summary as a markdown file
- send_email: send a follow-up email (runs locally, appended to a log)

Workflow for every transcript you receive:
1. Read the transcript with read_transcript
2. Identify all action items: what was decided, who owns it, when it is due
3. For each owner, call lookup_team_member to get their email address
4. Call create_task once per action item
5. Call save_summary to save a structured markdown summary
6. Call send_email to send a recap to all participants
7. Report back what you did

Be concise. Show your work through tool calls, not long explanations.
If a due date is not mentioned, default to one week from today.
If an owner is not mentioned, leave it as "unassigned".
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
