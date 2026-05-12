"""Parse the pipe-delimited function-call strings the fine-tuned model emits.

Format: `FuncName|$arg1=val1|$arg2=val2`. A bare function name with no
arguments (e.g. `HassGetCurrentTime`) is also valid.

This is the canonical parser for both eval scoring and the demo wrapper.
"""

from __future__ import annotations


def parse_call(text: str) -> tuple[str, dict[str, str]] | None:
    """Parse `FuncName|$arg=val|...` into (function_name, args_dict).

    Returns None if the string isn't a well-formed function call. The function
    name must be a valid identifier (letters, digits, underscores) so that
    error strings or plain ASR text don't accidentally count as parseable.
    Argument values may contain `=` (split on the first `=` only) and
    whitespace.
    """
    text = text.strip()
    if not text:
        return None
    parts = text.split("|")
    fn = parts[0].strip()
    if not fn or not fn.replace("_", "").isalnum():
        return None
    args: dict[str, str] = {}
    for p in parts[1:]:
        if "=" not in p:
            return None
        k, v = p.split("=", 1)
        k = k.strip()
        if not k.startswith("$") or not k[1:].replace("_", "").isalnum():
            return None
        args[k] = v.strip()
    return fn, args
