# Tool Calling Findings: LFM Models

Observations from building and debugging the LFM Home Assistant demo. Useful for anyone
integrating LFM models with tool calling via llama-server and the OpenAI SDK.

---

## 1. llama-server handles special tokens transparently

Pass `tools=` to the OpenAI SDK pointed at `llama-server` and you get back structured
`message.tool_calls` objects, exactly like the OpenAI API. The model's internal special
tokens (e.g. tool call delimiters) are parsed by llama-server and never surface in Python.
No custom parsing is needed on the application side.

## 2. Small models may call tools when no tool applies

Tested with LFM2-1.2B-Tool. When asked "What time is it?" (no `get_time` tool available),
the model returned `get_weather` with `finish_reason: tool_calls` instead of responding
directly. This is expected behavior for a 1.2B model: it learned to reach for tools and
struggles to decide when NOT to use them.

Mitigation: add explicit instruction to the system prompt.

```python
"Only call a tool when it is directly needed to fulfill the user's request. "
"If no tool applies, respond directly with text."
```

## 3. Without a repeat-call guard, the agent loop will spin to max_iter

When the model calls a tool whose result does not help it answer the question, it will
call the same tool again on the next turn with identical arguments. This repeats until
`max_iter` is hit, producing a poor user experience ("Max iterations reached.").

Fix: track seen `(tool, args)` pairs and break early when a repeat is detected.

```python
seen_calls: set[str] = set()

call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
if call_key in seen_calls:
    return message.content or "I'm not sure how to help with that using the available tools."
seen_calls.add(call_key)
```

This turns a 4-iteration loop with a confusing error into a single call with a graceful
fallback message.

## 4. System prompt phrasing matters more than you expect at small scale

At 1.2B parameters the model is sensitive to system prompt wording. A bare
`"You are a helpful home assistant AI."` is not enough to prevent spurious tool calls.
Adding a single sentence about when to call tools versus when to respond directly
significantly reduced off-topic tool use across test prompts.

## 5. The benchmark and the server share the same agent loop

Because `benchmark/run.py` imports `app.agent.run_agent` directly, any improvement to
the agent (system prompt, guards) is immediately reflected in benchmark scores. There is
no divergence between what is benchmarked and what runs in the browser.

## 6. After a successful tool call, small models re-issue the same call instead of summarising

Observed with LFM2-1.2B-Tool on the second turn of a multi-step interaction (e.g. "turn on
the bedroom light" followed by "switch it off"). The model correctly calls `toggle_lights`
on the first iteration and receives the tool result. On the second iteration it calls
`toggle_lights` again with identical arguments instead of generating a text confirmation.
This triggers the repeat-call guard, which at the time returned `message.content or fallback`.
Because a tool-call-only message has `content = None`, the user always saw the fallback string
"I'm not sure how to help with that using the available tools." even though the action succeeded.

Root cause: small LFMs do not reliably learn the "stop calling tools and summarise" behaviour
when fine-tuned on limited instruction data.

Fix: move the duplicate check to BEFORE appending the assistant message, `break` instead of
returning early, then do one final forced-text API call with `tool_choice="none"` after the
loop. The model sees the full conversation including tool results and must respond in prose.

```python
# check before appending so messages stays valid for the fallback call
duplicate = any(
    f"{tc.function.name}:{json.dumps(json.loads(tc.function.arguments), sort_keys=True)}"
    in seen_calls
    for tc in message.tool_calls
)
if duplicate:
    break

# ... execute tools, append results ...

# forced text-only summary (reached on duplicate or max_iter)
final = client.chat.completions.create(
    model="local",
    messages=messages,
    tools=TOOL_SCHEMAS,
    tool_choice="none",
    temperature=0.1,
    max_tokens=256,
)
return final.choices[0].message.content or "Done."
```

This pattern is safer than returning early: it guarantees a human-readable response while
keeping the messages list in a consistent state (no orphaned tool-call messages without
matching tool results).

## 7. Without conversation history the model cannot resolve pronouns across turns

Each call to `run_agent` rebuilt `messages` from scratch (system + current user message only).
On the second turn, "switch it off" gave the model zero context about which room was mentioned
in the previous turn. The model guessed a room — consistently picking `living_room` — even
though the user had just said "turn on the bedroom light".

Fix: maintain a `conversation_history` list in the server and prepend it to `messages` on
every call. Only plain `user`/`assistant` text turns belong in the history; internal tool-call
and tool-result messages from the agent loop are not included, keeping the history concise and
avoiding protocol errors.

```python
# server.py
conversation_history: list[dict] = []

text = run_agent(req.message, history=conversation_history, on_tool_call=on_tool_call)
conversation_history.append({"role": "user",      "content": req.message})
conversation_history.append({"role": "assistant", "content": text})

# agent.py
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    *(history or []),
    {"role": "user", "content": user_message},
]
```

## 8. Port conflicts are silent on macOS

When running the Vite dev server (from a prior project in the same directory) alongside
uvicorn on the same port, macOS lets both processes bind without error. The first process
to bind wins all incoming connections. The FastAPI server receives nothing and the browser
gets 404s with no obvious indication of why. Always check `lsof -i :<port>` before
debugging application logic.
