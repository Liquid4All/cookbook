"""Pin the contract of the shared function-call parser.

The eight cases below match the PRD's testing decisions: bare function name,
single arg, multiple args, `=` inside a value, whitespace inside a value,
empty input, plain ASR text, malformed argument.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _parser import parse_call  # noqa: E402


def test_bare_function_name() -> None:
    assert parse_call("HassGetCurrentTime") == ("HassGetCurrentTime", {})


def test_single_argument() -> None:
    assert parse_call("HassStartTimer|$minutes=5") == (
        "HassStartTimer",
        {"$minutes": "5"},
    )


def test_multiple_arguments() -> None:
    assert parse_call("HassLightSet|$area=bedroom|$brightness=70") == (
        "HassLightSet",
        {"$area": "bedroom", "$brightness": "70"},
    )


def test_argument_value_contains_equals_sign() -> None:
    assert parse_call("HassBroadcast|$message=2+2=4") == (
        "HassBroadcast",
        {"$message": "2+2=4"},
    )


def test_argument_value_contains_whitespace() -> None:
    assert parse_call("HassLightSet|$area=living room") == (
        "HassLightSet",
        {"$area": "living room"},
    )


def test_empty_string_returns_none() -> None:
    assert parse_call("") is None
    assert parse_call("   ") is None


def test_plain_asr_text_returns_none() -> None:
    assert parse_call("Dinner is ready.") is None


def test_malformed_argument_returns_none() -> None:
    assert parse_call("HassLightSet|area=bedroom") is None
    assert parse_call("HassLightSet|$area") is None
    assert parse_call("HassStartTimer|") is None
